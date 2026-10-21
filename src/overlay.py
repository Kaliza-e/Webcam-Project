"""
src/overlay.py
Visual overlay rendering bounding boxes, state-dependent colors, face-to-body link lines,
virtual centring crosshair/error arrows, and rich diagnostic HUD.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from src.identity import TrackIdentity, State
from src.face import FaceDetection


STATE_COLORS = {
    State.CONFIRMED: (46, 204, 113),   # BGR Vibrant Green
    State.TEMPORARY: (0, 165, 255),    # BGR Warm Amber
    State.UNKNOWN: (130, 130, 130),    # BGR Neutral Grey
}


class VisualOverlay:
    """
    Renders diagnostic HUD, track states, face associations, and virtual-pan crosshairs.
    """

    def __init__(
        self,
        show_debug: bool = False,
        show_fps: bool = True,
        show_virtual_pan: bool = True,
    ):
        self.show_debug = show_debug
        self.show_fps = show_fps
        self.show_virtual_pan = show_virtual_pan

    def toggle_debug(self):
        self.show_debug = not self.show_debug

    def render(
        self,
        frame: np.ndarray,
        tracks: Dict[int, Tuple[TrackIdentity, Tuple[float, float, float, float]]],
        faces: List[FaceDetection],
        face_assignments: Dict[int, int],  # track_id -> face_idx
        ambiguous_track_ids: set,
        centring_data: Tuple[float, float, float, str, Optional[float], Optional[float], Optional[int]],
        # (err, pan_cmd, pan_angle, status, target_cx, target_cy, locked_tid)
        hud_data: Dict[str, any],
    ) -> np.ndarray:
        """
        Draw all overlays onto a copy of the frame.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        err, pan_cmd, pan_angle, pan_status, target_cx, target_cy, locked_tid = centring_data

        # 1. Draw Body Tracks & Associated Faces
        for tid, (identity, bbox) in tracks.items():
            bx, by, bw, bh = [int(v) for v in bbox]
            state = identity.state
            color = STATE_COLORS.get(state, (130, 130, 130))

            # Draw body box
            thickness = 2 if tid != locked_tid else 3
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, thickness)

            # Draw upper head region guide (subtle dashed/thin line at 40% height)
            head_h = int(by + bh * 0.40)
            cv2.line(canvas, (bx, head_h), (bx + bw, head_h), (color[0] // 2, color[1] // 2, color[2] // 2), 1)

            # Build label: ID | name | state | conf
            if state == State.CONFIRMED:
                label = f"ID:{tid} | {identity.name} | CONFIRMED | {identity.confidence:.2f}"
            elif state == State.TEMPORARY:
                if getattr(identity, "is_body_locked", False):
                    label = f"ID:{tid} | {identity.name} | TEMPORARY [BODY LOCK] ({identity.confidence:.2f})"
                else:
                    label = f"ID:{tid} | {identity.name} | TEMPORARY ({identity.confidence:.2f})"
            else:
                label = f"ID:{tid} | Unknown | UNKNOWN | 0.00"

            if tid in ambiguous_track_ids:
                label += " [AMBIG]"

            # Label banner
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (bx, max(0, by - 20)), (bx + lw + 8, max(20, by)), color, -1)
            cv2.putText(
                canvas,
                label,
                (bx + 4, max(15, by - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0) if state != State.UNKNOWN else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # If face is assigned to this track, draw face box & link line
            if tid in face_assignments:
                f_idx = face_assignments[tid]
                if 0 <= f_idx < len(faces):
                    face_det = faces[f_idx]
                    fx, fy, fw, fh = [int(v) for v in face_det.bbox_xywh]
                    fcx, fcy = int(fx + fw / 2), int(fy + fh / 2)

                    # Face box
                    cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
                    # Line from face center to body top center
                    body_top_cx = int(bx + bw / 2)
                    cv2.line(canvas, (fcx, fcy), (body_top_cx, by), (255, 255, 0), 1, cv2.LINE_AA)

        # 2. Draw Unassigned Faces (if any)
        assigned_face_indices = set(face_assignments.values())
        for f_idx, face_det in enumerate(faces):
            if f_idx not in assigned_face_indices:
                fx, fy, fw, fh = [int(v) for v in face_det.bbox_xywh]
                cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (180, 180, 180), 1)
                cv2.putText(
                    canvas,
                    f"Face {face_det.score:.2f} (Unassigned)",
                    (fx, max(15, fy - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

        # 3. Virtual Centring Display
        if self.show_virtual_pan:
            cx_frame = w // 2
            cy_frame = h // 2

            # Camera center crosshair (with simulated angle offset shift)
            # Shifts virtual reticle by simulated_pan_angle scaled to pixels
            pan_offset_x = int(pan_angle * 4.0)
            v_cx = int(np.clip(cx_frame - pan_offset_x, 30, w - 30))

            # Virtual center reticle
            reticle_color = (0, 255, 255) if pan_status == "TRACKING" else (160, 160, 160)
            cv2.line(canvas, (v_cx - 15, cy_frame), (v_cx + 15, cy_frame), reticle_color, 1)
            cv2.line(canvas, (v_cx, cy_frame - 15), (v_cx, cy_frame + 15), reticle_color, 1)
            cv2.circle(canvas, (v_cx, cy_frame), 8, reticle_color, 1)

            # If target active, draw error vector
            if target_cx is not None and target_cy is not None and pan_status == "TRACKING":
                cv2.arrowedLine(
                    canvas,
                    (cx_frame, cy_frame),
                    (int(target_cx), cy_frame),
                    (0, 165, 255),
                    2,
                    tipLength=0.15,
                )
                cv2.circle(canvas, (int(target_cx), int(target_cy)), 6, (0, 165, 255), -1)

            # Centring status badge bottom center
            badge_text = f"PAN: {pan_status} | Cmd: {pan_cmd:+.1f} deg/s | Angle: {pan_angle:+.1f} deg | Err: {err:+.2f}"
            (bw_t, bh_t), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_x = (w - bw_t) // 2
            cv2.rectangle(canvas, (badge_x - 6, h - 35), (badge_x + bw_t + 6, h - 10), (20, 20, 20), -1)
            cv2.putText(
                canvas,
                badge_text,
                (badge_x, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255) if pan_status == "TRACKING" else (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        # 4. HUD Banner (Top Left)
        fps = hud_data.get("fps", 0.0)
        t_half = hud_data.get("t_half", 2.0)
        floor = hud_data.get("floor", 0.3)
        scenario = hud_data.get("scenario", "live")
        is_recording = hud_data.get("is_recording", False)
        target_name = hud_data.get("target_name", "Auto")

        # Top semi-transparent banner
        hud_h = 100 if self.show_debug else 52
        overlay_banner = canvas[0:hud_h, 0:w].copy()
        cv2.rectangle(overlay_banner, (0, 0), (w, hud_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay_banner, 0.65, canvas[0:hud_h, 0:w], 0.35, 0, canvas[0:hud_h, 0:w])

        # Row 1
        fps_color = (0, 255, 0) if fps >= 15.0 else (0, 165, 255)
        cv2.putText(canvas, f"FPS: {fps:4.1f}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"T_half: {t_half:.1f}s", (125, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Floor: {floor:.2f}", (240, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Target: {target_name or 'Auto'}", (345, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Tag: {scenario}", (480, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Recording Indicator (Top Right)
        if is_recording:
            cv2.circle(canvas, (w - 75, 20), 6, (0, 0, 255), -1)
            cv2.putText(canvas, "REC", (w - 62, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            rec_tag = "[MIRROR]" if hud_data.get("is_mirrored", False) else "[PAUSED]"
            cv2.putText(canvas, rec_tag, (w - 85, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if hud_data.get("is_mirrored", False) else (120, 120, 120), 1, cv2.LINE_AA)

        # Row 2 (Quick Key Guide)
        key_help = "Keys: 'q' quit | 'm' mirror | 'c' cam | 'f' full | 'a' fill/fit | 'r' rec | 't' target | '[' / ']' T_half"
        cv2.putText(canvas, key_help, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)

        # Row 3 & 4 (Per-stage Latency Debug Breakdown)
        if self.show_debug:
            latencies = hud_data.get("latencies", {})
            lat_str = " | ".join([f"{k}: {v:.1f}ms" for k, v in latencies.items()])
            cv2.putText(canvas, f"Latency: {lat_str}", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1, cv2.LINE_AA)
            markers_guide = "1:head_turn_start  2:head_turn_end  3:occl_start  4:occl_end  5:cross_start  6:cross_end"
            cv2.putText(canvas, f"Markers: {markers_guide}", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 180, 255), 1, cv2.LINE_AA)

        return canvas
