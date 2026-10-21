"""
src/pipeline.py
Core tracking and face-association pipeline wiring together capture, person tracking,
face detection/recognition, bipartite mapping, 3-state identity machine, simulated centring,
overlay rendering, and per-frame logging.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.capture import VideoCaptureWrapper
from src.detector import PersonDetectorTracker, Track
from src.face import FaceEngine, FaceDetection
from src.mapper import FaceBodyMapper
from src.identity import IdentityManager, State
from src.centring import VirtualPanController
from src.overlay import VisualOverlay
from src.logger import ExperimentLogger


class TrackingPipeline:
    """
    Complete end-to-end Person Tracking & Face Association Pipeline.
    """

    def __init__(self, config: dict, session_name: Optional[str] = None):
        self.config = config

        # Set fixed random seed for full reproducibility
        seed = config.get("seed", 42)
        np.random.seed(seed)

        # 1. Capture config
        cap_cfg = config.get("capture", {})
        self.frame_w = cap_cfg.get("width", 640)
        self.frame_h = cap_cfg.get("height", 480)
        self.target_fps = cap_cfg.get("target_fps", 30)

        # 2. Person detector & tracker
        det_cfg = config.get("detector", {})
        self.detector = PersonDetectorTracker(
            backend=det_cfg.get("backend", "yolo"),
            model_path=det_cfg.get("model_path", "models/yolov8n.pt"),
            tracker_config=det_cfg.get("tracker_config", "bytetrack.yaml"),
            conf_threshold=det_cfg.get("conf_threshold", 0.40),
            iou_threshold=det_cfg.get("iou_threshold", 0.50),
            detect_interval=det_cfg.get("detect_interval", 2),
        )

        # 3. Face engine
        face_cfg = config.get("face", {})
        self.face_engine = FaceEngine(
            yunet_path=face_cfg.get("yunet_model_path", "models/face_detection_yunet_2023mar.onnx"),
            sface_path=face_cfg.get("sface_model_path", "models/face_recognition_sface_2021dec.onnx"),
            score_threshold=face_cfg.get("score_threshold", 0.60),
            nms_threshold=face_cfg.get("nms_threshold", 0.30),
            top_k=face_cfg.get("top_k", 5000),
            min_face_size=face_cfg.get("min_face_size", 20),
            match_threshold=face_cfg.get("match_threshold", 0.363),
            enrolled_dir=face_cfg.get("enrolled_dir", "data/enrolled"),
        )
        self.match_threshold = face_cfg.get("match_threshold", 0.363)

        # 4. Face-to-body mapper
        map_cfg = config.get("mapper", {})
        self.mapper = FaceBodyMapper(
            upper_body_ratio=map_cfg.get("upper_body_ratio", 0.40),
            min_face_width_ratio=map_cfg.get("min_face_width_ratio", 0.12),
            max_face_width_ratio=map_cfg.get("max_face_width_ratio", 0.65),
            min_assignment_score=map_cfg.get("min_assignment_score", 0.40),
            overlap_iou_threshold=map_cfg.get("overlap_iou_threshold", 0.25),
            weights=map_cfg.get("weights", None),
        )

        # 5. Identity manager
        id_cfg = config.get("identity", {})
        self.identity_mgr = IdentityManager(
            half_life_s=id_cfg.get("half_life_s", 2.0),
            confidence_floor=id_cfg.get("confidence_floor", 0.30),
            hysteresis_k=id_cfg.get("hysteresis_k", 2),
            track_lost_frames=det_cfg.get("track_lost_frames", 30),
            ambiguous_decay_multiplier=id_cfg.get("ambiguous_decay_multiplier", 1.5),
            ambiguous_penalty=map_cfg.get("ambiguous_penalty", 0.05),
            body_continuity_enabled=id_cfg.get("body_continuity_enabled", True),
            body_retention_weight=id_cfg.get("body_retention_weight", 0.75),
            body_match_threshold=id_cfg.get("body_match_threshold", 0.45),
            min_body_confidence=id_cfg.get("min_body_confidence", 0.60),
        )

        # 6. Centring controller
        cent_cfg = config.get("centring", {})
        self.target_name = cent_cfg.get("target_name", "")
        self.centring = VirtualPanController(
            target_name=self.target_name,
            kp=cent_cfg.get("kp", 1.5),
            deadband=cent_cfg.get("deadband", 0.03),
            pan_max=cent_cfg.get("pan_max", 30.0),
            max_angle=cent_cfg.get("max_angle", 45.0),
            smoothing_alpha=cent_cfg.get("smoothing_alpha", 0.35),
        )

        # 7. Overlay
        over_cfg = config.get("overlay", {})
        self.overlay = VisualOverlay(
            show_debug=over_cfg.get("show_debug", False),
            show_fps=over_cfg.get("show_fps", True),
            show_virtual_pan=over_cfg.get("show_virtual_pan", True),
        )

        # 8. Logger
        log_cfg = config.get("logging", {})
        self.log_dir = log_cfg.get("log_dir", "logs")
        self.video_dir = Path(log_cfg.get("video_dir", "videos"))
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ExperimentLogger(
            output_dir=self.log_dir,
            session_name=session_name,
            config_snapshot=config,
        )

        # State / Performance tracking
        self.fps_rolling: float = 30.0
        self.stage_latencies: Dict[str, float] = {
            "capture": 0.0,
            "person_track": 0.0,
            "face_detect": 0.0,
            "embedding": 0.0,
            "mapping": 0.0,
            "state_update": 0.0,
            "draw": 0.0,
        }
        self.is_recording: bool = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.scenario_tag: str = "default"

    def set_scenario_tag(self, tag: str):
        self.scenario_tag = tag

    def toggle_recording(self, fps: float = 30.0):
        self.is_recording = not self.is_recording
        if self.is_recording:
            vid_path = self.video_dir / f"{self.logger.session_name}_raw.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (self.frame_w, self.frame_h))
            print(f"[Recording STARTED] Output: {vid_path}")
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            print("[Recording STOPPED]")

    def step_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_idx: int,
        event_marker: Optional[str] = None,
        is_mirrored: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """
        Execute one full processing iteration on a single frame.
        """
        frame_start_t = time.perf_counter()

        if self.is_recording and self.video_writer is not None and frame is not None:
            self.video_writer.write(frame)

        # Stage 1: Person detect / track
        t0 = time.perf_counter()
        body_tracks: List[Track] = self.detector.detect_and_track(frame, frame_idx)
        t_detect = (time.perf_counter() - t0) * 1000.0
        self.stage_latencies["person_track"] = t_detect

        # Stage 2: Face detect
        t0 = time.perf_counter()
        detected_faces: List[FaceDetection] = self.face_engine.detect_faces(frame)
        t_face = (time.perf_counter() - t0) * 1000.0
        self.stage_latencies["face_detect"] = t_face

        # Stage 3: Embedding extraction & matching
        t0 = time.perf_counter()
        for f in detected_faces:
            self.face_engine.extract_and_match(frame, f)
        t_emb = (time.perf_counter() - t0) * 1000.0
        self.stage_latencies["embedding"] = t_emb

        # Stage 4: Face-to-body mapping
        t0 = time.perf_counter()
        face_boxes = [f.bbox_xywh for f in detected_faces]
        track_tuples = [(t.id, t.bbox_xywh) for t in body_tracks]
        face_assignments, ambiguous_track_ids = self.mapper.match(face_boxes, track_tuples)
        t_map = (time.perf_counter() - t0) * 1000.0
        self.stage_latencies["mapping"] = t_map

        # Build face evidence dictionary for identity machine:
        # track_id -> (matched_name, similarity)
        face_evidence: Dict[int, Tuple[Optional[str], float]] = {}
        for tid, f_idx in face_assignments.items():
            if 0 <= f_idx < len(detected_faces):
                f_det = detected_faces[f_idx]
                face_evidence[tid] = (f_det.matched_name, f_det.similarity)

        # Stage 5: State machine update
        t0 = time.perf_counter()
        active_tids = [t.id for t in body_tracks]
        body_box_map = {t.id: t.bbox_xywh for t in body_tracks}
        active_identities = self.identity_mgr.step(
            current_time=timestamp,
            active_track_ids=active_tids,
            face_assignments=face_evidence,
            match_threshold=self.match_threshold,
            ambiguous_track_ids=ambiguous_track_ids,
            frame=frame,
            body_bboxes=body_box_map,
        )

        # Map tracks for centring controller: track_id -> (TrackIdentity, bbox)
        tracks_for_centring = {}
        for tid, ident in active_identities.items():
            tracks_for_centring[tid] = (ident, body_box_map.get(tid, (0, 0, 0, 0)))

        # Update centring controller
        err, pan_cmd, pan_angle, pan_status = self.centring.update(
            current_time=timestamp,
            frame_width=self.frame_w,
            tracks=tracks_for_centring,
        )
        t_state = (time.perf_counter() - t0) * 1000.0
        self.stage_latencies["state_update"] = t_state

        # Total pipeline time for frame
        frame_elapsed = max(time.perf_counter() - frame_start_t, 1e-4)
        current_fps = 1.0 / frame_elapsed
        self.fps_rolling = 0.90 * self.fps_rolling + 0.10 * current_fps

        # Log frame data
        track_log_entries = []
        for tid in active_tids:
            ident = active_identities.get(tid)
            bbox = body_box_map.get(tid, (0, 0, 0, 0))
            is_face_vis = tid in face_assignments
            sim = face_evidence.get(tid, (None, 0.0))[1] if is_face_vis else 0.0
            track_log_entries.append({
                "track_id": tid,
                "bbox": bbox,
                "state": ident.state.value if ident else "Unknown",
                "name": ident.name if ident else "Unknown",
                "confidence": ident.confidence if ident else 0.0,
                "similarity": sim,
                "face_visible": is_face_vis,
                "ambiguous": tid in ambiguous_track_ids,
            })

        self.logger.log_frame(
            frame_idx=frame_idx,
            timestamp=timestamp,
            track_entries=track_log_entries,
            centring_error=err,
            pan_cmd=pan_cmd,
            pan_angle=pan_angle,
            fps=self.fps_rolling,
            event_marker=event_marker,
        )

        # Stage 6: Draw overlay
        t0 = time.perf_counter()
        hud_info = {
            "fps": self.fps_rolling,
            "t_half": self.identity_mgr.half_life_s,
            "floor": self.identity_mgr.confidence_floor,
            "scenario": self.scenario_tag,
            "is_recording": self.is_recording,
            "is_mirrored": is_mirrored,
            "target_name": self.centring.target_name,
            "latencies": self.stage_latencies,
        }
        centring_tuple = (
            err,
            pan_cmd,
            pan_angle,
            pan_status,
            self.centring.target_cx,
            self.centring.target_cy,
            self.centring.locked_track_id,
        )

        annotated_frame = self.overlay.render(
            frame=frame,
            tracks=tracks_for_centring,
            faces=detected_faces,
            face_assignments=face_assignments,
            ambiguous_track_ids=ambiguous_track_ids,
            centring_data=centring_tuple,
            hud_data=hud_info,
        )
        t_draw = (time.perf_counter() - t0) * 1000.0
        self.stage_latencies["draw"] = t_draw

        status_dict = {
            "fps": self.fps_rolling,
            "centring_error": err,
            "pan_cmd": pan_cmd,
            "pan_angle": pan_angle,
            "pan_status": pan_status,
            "active_tracks": len(body_tracks),
            "latencies": self.stage_latencies,
        }

        return annotated_frame, status_dict

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
