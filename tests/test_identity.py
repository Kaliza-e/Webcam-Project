"""
tests/test_identity.py
Unit tests verifying exponential decay math, all transitions, and Honesty Rules 1 through 6.
"""

import math
import pytest
from src.identity import TrackIdentity, IdentityManager, State


def test_decay_math_and_half_life():
    """
    Verify exponential decay formula:
    conf(t) = conf_at_seen * exp(-lambda * dt), lambda = ln(2) / T_half.
    At dt = T_half, confidence must be EXACTLY half of conf_at_seen.
    At dt = 2 * T_half, confidence must be EXACTLY a quarter.
    """
    half_life = 2.0
    track = TrackIdentity(track_id=1, half_life_s=half_life, confidence_floor=0.1)
    track.conf_at_seen = 0.80
    track.t_last_seen = 10.0

    # At t = 10.0 (dt = 0)
    c0 = track.compute_decay(current_time=10.0)
    assert math.isclose(c0, 0.80, abs_tol=1e-5)

    # At t = 12.0 (dt = T_half) -> should be 0.40
    c1 = track.compute_decay(current_time=12.0)
    assert math.isclose(c1, 0.40, abs_tol=1e-5)

    # At t = 14.0 (dt = 2 * T_half) -> should be 0.20
    c2 = track.compute_decay(current_time=14.0)
    assert math.isclose(c2, 0.20, abs_tol=1e-5)

    # At t = 16.0 (dt = 3 * T_half) -> should be 0.10
    c3 = track.compute_decay(current_time=16.0)
    assert math.isclose(c3, 0.10, abs_tol=1e-5)


def test_state_transitions():
    """
    Test transition cycle:
    Unknown -> Confirmed (hysteresis) -> Temporary (face lost) -> Confirmed (face recovered) -> Unknown (decay).
    """
    mgr = IdentityManager(half_life_s=2.0, confidence_floor=0.30, hysteresis_k=2)

    # 1. First observation (k=1): remains Unknown
    tracks = mgr.step(
        current_time=1.0,
        active_track_ids=[1],
        face_assignments={1: ("Alice", 0.90)},
        match_threshold=0.363,
    )
    assert tracks[1].state == State.UNKNOWN

    # 2. Second observation (k=2): promoted to Confirmed
    tracks = mgr.step(
        current_time=1.033,
        active_track_ids=[1],
        face_assignments={1: ("Alice", 0.90)},
        match_threshold=0.363,
    )
    assert tracks[1].state == State.CONFIRMED
    assert tracks[1].name == "Alice"
    assert math.isclose(tracks[1].confidence, 0.90, abs_tol=1e-4)

    # 3. Face lost: transition to Temporary
    tracks = mgr.step(
        current_time=1.5,
        active_track_ids=[1],
        face_assignments={},  # no face
        match_threshold=0.363,
    )
    assert tracks[1].state == State.TEMPORARY
    assert tracks[1].name == "Alice"
    assert tracks[1].confidence < 0.90

    # 4. Face recovered: returns to Confirmed
    tracks = mgr.step(
        current_time=1.6,
        active_track_ids=[1],
        face_assignments={1: ("Alice", 0.85)},
        match_threshold=0.363,
    )
    assert tracks[1].state == State.CONFIRMED
    assert math.isclose(tracks[1].confidence, 0.85, abs_tol=1e-4)

    # 5. Face lost for a long duration -> drops below floor (0.30) to Unknown
    # dt > 2 * T_half * ln(0.85/0.30)
    tracks = mgr.step(
        current_time=10.0,
        active_track_ids=[1],
        face_assignments={},
        match_threshold=0.363,
    )
    assert tracks[1].state == State.UNKNOWN
    assert tracks[1].name == "Unknown"
    assert tracks[1].confidence == 0.0


def test_honesty_rule_1_never_present_temporary_as_confirmed():
    """
    Honesty Rule 1: Never present Temporary as Confirmed.
    Distinct enum and labels.
    """
    mgr = IdentityManager(half_life_s=2.0, confidence_floor=0.30, hysteresis_k=1)
    # Confirm
    mgr.step(current_time=1.0, active_track_ids=[1], face_assignments={1: ("Alice", 0.80)}, match_threshold=0.363)
    assert mgr.tracks[1].state == State.CONFIRMED

    # Drop face
    mgr.step(current_time=2.0, active_track_ids=[1], face_assignments={}, match_threshold=0.363)
    assert mgr.tracks[1].state == State.TEMPORARY
    assert mgr.tracks[1].state != State.CONFIRMED
    assert mgr.tracks[1].state.value == "Temporary"


def test_honesty_rule_2_contradicting_face_drops_to_unknown():
    """
    Honesty Rule 2: If a face mapped to a track matches a DIFFERENT person
    (or no one), immediately drop that track to Unknown (catches identity swaps).
    """
    mgr = IdentityManager(half_life_s=2.0, confidence_floor=0.30, hysteresis_k=2)
    # Establish Alice
    mgr.step(current_time=1.0, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)
    mgr.step(current_time=1.033, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)
    assert mgr.tracks[1].state == State.CONFIRMED
    assert mgr.tracks[1].name == "Alice"

    # Bob's face is now detected on Track 1!
    mgr.step(current_time=1.066, active_track_ids=[1], face_assignments={1: ("Bob", 0.88)}, match_threshold=0.363)
    # Immediately drops to Unknown (does not immediately become Bob without hysteresis)
    assert mgr.tracks[1].state == State.UNKNOWN
    assert mgr.tracks[1].name == "Unknown"

    # Test unrecognized face (< threshold, e.g. covered/occluded face) transitions Confirmed to Temporary
    mgr.tracks[1].state = State.CONFIRMED
    mgr.tracks[1].name = "Alice"
    mgr.tracks[1].conf_at_seen = 0.90
    mgr.tracks[1].t_last_seen = 2.0
    mgr.step(current_time=2.1, active_track_ids=[1], face_assignments={1: (None, 0.15)}, match_threshold=0.363)
    assert mgr.tracks[1].state == State.TEMPORARY
    assert mgr.tracks[1].name == "Alice"


def test_honesty_rule_3_track_lost_timeout():
    """
    Honesty Rule 3: If a track disappears for more than track_lost_frames, delete its identity.
    """
    mgr = IdentityManager(track_lost_frames=5, hysteresis_k=1)
    mgr.step(current_time=1.0, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)
    assert 1 in mgr.tracks

    # Track 1 disappears for 5 frames
    for f in range(5):
        mgr.step(current_time=1.0 + f * 0.033, active_track_ids=[], face_assignments={}, match_threshold=0.363)
    assert 1 in mgr.tracks  # Still within 5 frames

    # Frame 6 (> 5): purged
    mgr.step(current_time=2.0, active_track_ids=[], face_assignments={}, match_threshold=0.363)
    assert 1 not in mgr.tracks


def test_honesty_rule_4_duplicate_identity_conflict():
    """
    Honesty Rule 4: If two tracks claim the same identity, only the one with higher confidence
    keeps it; the other goes to Unknown.
    """
    mgr = IdentityManager(hysteresis_k=1)
    # Track 1 confirmed as Alice with 0.85
    mgr.step(current_time=1.0, active_track_ids=[1], face_assignments={1: ("Alice", 0.85)}, match_threshold=0.363)
    # Track 1 enters Temporary
    mgr.step(current_time=1.5, active_track_ids=[1], face_assignments={}, match_threshold=0.363)
    conf_track1 = mgr.tracks[1].confidence

    # Now Track 2 appears claiming Alice with fresh high confidence 0.92
    mgr.step(
        current_time=1.6,
        active_track_ids=[1, 2],
        face_assignments={2: ("Alice", 0.92)},
        match_threshold=0.363,
    )
    # Track 2 wins, Track 1 dropped to Unknown
    assert mgr.tracks[2].state == State.CONFIRMED
    assert mgr.tracks[2].name == "Alice"
    assert mgr.tracks[1].state == State.UNKNOWN
    assert mgr.tracks[1].name == "Unknown"


def test_honesty_rule_5_ambiguous_overlap():
    """
    Honesty Rule 5: When two bodies overlap ambiguously and face is not visible,
    do not extend/restore Temporary identity, and apply extra confidence penalty / faster decay.
    """
    mgr = IdentityManager(half_life_s=2.0, hysteresis_k=1, ambiguous_decay_multiplier=1.5, ambiguous_penalty=0.05)
    mgr.step(current_time=1.0, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)

    # Face lost under ambiguous overlap
    mgr.step(
        current_time=2.0,
        active_track_ids=[1, 2],
        face_assignments={},
        match_threshold=0.363,
        ambiguous_track_ids={1, 2},
    )
    # Should decay faster than regular decay
    conf_ambiguous = mgr.tracks[1].confidence
    normal_decay = 0.90 * math.exp(-(math.log(2) / 2.0) * 1.0)
    assert conf_ambiguous < normal_decay


def test_honesty_rule_6_hysteresis():
    """
    Honesty Rule 6: Use hysteresis (require K consecutive matched frames, default 2)
    before promoting Unknown to Confirmed.
    """
    mgr = IdentityManager(hysteresis_k=3)
    # Frame 1
    mgr.step(current_time=1.0, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)
    assert mgr.tracks[1].state == State.UNKNOWN

    # Frame 2
    mgr.step(current_time=1.033, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)
    assert mgr.tracks[1].state == State.UNKNOWN

    # Frame 3: reaches K=3 -> Confirmed!
    mgr.step(current_time=1.066, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)
    assert mgr.tracks[1].state == State.CONFIRMED
    assert mgr.tracks[1].name == "Alice"


def test_body_tracking_continuity():
    """
    Verify security-grade whole-person body tracking continuity:
    When face is occluded/lost, unbroken body tracking and torso appearance match
    sustains identity in Temporary state with [BODY LOCK], maintaining high confidence.
    """
    import numpy as np
    mgr = IdentityManager(
        half_life_s=2.0,
        confidence_floor=0.30,
        hysteresis_k=1,
        body_continuity_enabled=True,
        body_retention_weight=0.75,
        min_body_confidence=0.60,
    )
    # Synthetic frame with a distinct torso patch
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    frame[50:350, 150:400] = [30, 80, 200]
    bbox = (150, 50, 250, 300)

    # 1. Confirm Alice with face and body frame
    mgr.step(
        current_time=1.0,
        active_track_ids=[1],
        face_assignments={1: ("Alice", 0.90)},
        match_threshold=0.363,
        frame=frame,
        body_bboxes={1: bbox},
    )
    assert mgr.tracks[1].state == State.CONFIRMED
    assert mgr.tracks[1].name == "Alice"
    assert mgr.tracks[1].body_signature is not None
    assert mgr.tracks[1].is_body_locked

    # 2. Face covered / occluded for 3.0 seconds (dt=3s > T_half)
    mgr.step(
        current_time=4.0,
        active_track_ids=[1],
        face_assignments={},  # face covered
        match_threshold=0.363,
        frame=frame,
        body_bboxes={1: bbox},
    )
    # Body lock should remain active
    assert mgr.tracks[1].state == State.TEMPORARY
    assert mgr.tracks[1].name == "Alice"
    assert mgr.tracks[1].is_body_locked
    # Sustained confidence floor (0.90 * 0.60 * 1.0 = 0.54 >= 0.50)
    assert mgr.tracks[1].confidence >= 0.50
