"""
Pose and form-analysis utilities for the Smartan bicep curl task.

This module:
- wraps YOLOv8 keypoints into a small ArmPose data structure
- computes elbow angles and basic body geometry
- maintains state across frames to count reps
- evaluates rule-based form quality
- draws a full-body skeleton and highlights both arms,
  while still indicating which arm is more "active" in that frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# COCO-style keypoint indices used by YOLOv8 pose
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class ArmPose:
    """Keypoints needed for a single arm and torso reference."""

    shoulder: Tuple[float, float]
    elbow: Tuple[float, float]
    wrist: Tuple[float, float]
    hip: Tuple[float, float]


def _angle_at_joint(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> Optional[float]:
    """
    Compute the angle ABC (in degrees) at point B.

    Returns None if the vectors are degenerate.
    """
    a_v = np.asarray(a, dtype=np.float32)
    b_v = np.asarray(b, dtype=np.float32)
    c_v = np.asarray(c, dtype=np.float32)

    v1 = a_v - b_v
    v2 = c_v - b_v

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def slice_arm_pose(keypoints_xy: np.ndarray, side: str) -> Optional[ArmPose]:
    """
    Build an ArmPose from YOLO keypoints for the given side.

    side: "left" or "right"
    """
    if keypoints_xy is None or keypoints_xy.shape[0] < 17:
        return None

    if side == "right":
        s_idx, e_idx, w_idx, h_idx = R_SHOULDER, R_ELBOW, R_WRIST, R_HIP
    else:
        s_idx, e_idx, w_idx, h_idx = L_SHOULDER, L_ELBOW, L_WRIST, L_HIP

    try:
        shoulder = tuple(map(float, keypoints_xy[s_idx]))
        elbow = tuple(map(float, keypoints_xy[e_idx]))
        wrist = tuple(map(float, keypoints_xy[w_idx]))
        hip = tuple(map(float, keypoints_xy[h_idx]))
    except Exception:
        return None

    return ArmPose(shoulder=shoulder, elbow=elbow, wrist=wrist, hip=hip)


class BicepCurlEvaluator:
    """
    Stateful evaluator for a single-arm bicep curl.

    It tracks:
    - smoothed elbow angle across recent frames
    - global min/max angle to measure motion amplitude
    - rep count using angle thresholds
    - rule-based form assessment per frame

    Rules implemented:
      1. Range of motion (ROM) at the elbow
      2. Torso / upper-arm stability (shoulder vertical movement)
      3. Elbow kept near torso (avoid swinging forward)
      4. Wrist alignment with forearm (only when keypoints are plausible)
    """

    def __init__(
        self,
        smoothing: int = 9,
        min_motion_deg: float = 35.0,
        up_threshold: float = 60.0,
        down_threshold: float = 150.0,
    ) -> None:
        self._angle_history: deque[float] = deque(maxlen=smoothing)
        self._shoulder_y_history: deque[float] = deque(maxlen=smoothing)
        self._elbow_torso_history: deque[float] = deque(maxlen=smoothing)

        self._min_angle: Optional[float] = None
        self._max_angle: Optional[float] = None

        # rep-state machine
        self._phase: str = "extended"  # or "flexed"
        self.rep_count: int = 0

        self._min_motion_deg = float(min_motion_deg)
        self._up_thresh = float(up_threshold)
        self._down_thresh = float(down_threshold)

    def _update_range(self, angle: float) -> None:
        if self._min_angle is None or angle < self._min_angle:
            self._min_angle = angle
        if self._max_angle is None or angle > self._max_angle:
            self._max_angle = angle

    def _motion_amplitude(self) -> float:
        if self._min_angle is None or self._max_angle is None:
            return 0.0
        return float(self._max_angle - self._min_angle)

    def step(self, arm: Optional[ArmPose]) -> Dict:
        """
        Process one frame for ONE arm.

        Returns a feedback dict with:
            - elbow_angle (smoothed)
            - rep_count
            - good_form (bool)
            - messages (list[str])
            - rom_ok, torso_stable, elbow_near_torso, wrist_ok (per-rule flags)
        """
        if arm is None:
            return {
                "elbow_angle": None,
                "rep_count": self.rep_count,
                "good_form": False,
                "messages": ["Pose not reliable in this frame."],
                "rom_ok": False,
                "torso_stable": False,
                "elbow_near_torso": False,
                "wrist_ok": False,
            }

        angle = _angle_at_joint(arm.shoulder, arm.elbow, arm.wrist)
        if angle is None:
            return {
                "elbow_angle": None,
                "rep_count": self.rep_count,
                "good_form": False,
                "messages": ["Could not compute elbow angle."],
                "rom_ok": False,
                "torso_stable": False,
                "elbow_near_torso": False,
                "wrist_ok": False,
            }

        # --- tracking raw geometry histories ---
        self._angle_history.append(angle)
        self._shoulder_y_history.append(arm.shoulder[1])

        elbow_torso_offset = abs(arm.elbow[0] - arm.hip[0])
        self._elbow_torso_history.append(elbow_torso_offset)

        self._update_range(angle)
        smoothed_angle = float(np.mean(self._angle_history))
        motion = self._motion_amplitude()

        # --------- REP COUNT STATE MACHINE (this was missing) ----------
        # Only count reps once there is meaningful movement in this arm.
        if motion >= self._min_motion_deg:
            if self._phase == "extended":
                # arm is going up
                if smoothed_angle < self._up_thresh:
                    self._phase = "flexed"
            elif self._phase == "flexed":
                # arm is going back down
                if smoothed_angle > self._down_thresh:
                    self.rep_count += 1
                    self._phase = "extended"
        # ----------------------------------------------------------------

        torso_len = abs(arm.shoulder[1] - arm.hip[1])
        if torso_len < 1.0:
            torso_len = 200.0

        messages: List[str] = []
        good_form = True

        rom_ok = True
        torso_stable = True
        elbow_near_torso = True
        wrist_ok = True

        # Rule 1: ROM – only if there is actual movement
        if motion >= self._min_motion_deg:
            if 70.0 < smoothed_angle < 150.0:
                rom_ok = False
                messages.append("Increase curl depth for better range of motion.")
                good_form = False

        # Rule 2: torso / upper-arm stability (shoulder jitter)
        if len(self._shoulder_y_history) >= 4:
            shoulder_std = float(np.std(self._shoulder_y_history))
            if shoulder_std > 0.25 * torso_len:
                torso_stable = False
                messages.append("Minimise torso and shoulder movement.")
                good_form = False

        # Rule 3: keep elbow close to torso (avoid swinging forward)
        if len(self._elbow_torso_history) >= 4:
            avg_elbow_offset = float(np.mean(self._elbow_torso_history))
            if avg_elbow_offset > 0.4 * torso_len:
                elbow_near_torso = False
                messages.append("Keep elbow closer to your side.")
                good_form = False

        # Rule 4: wrist roughly above forearm (avoid wrist collapse)
        wrist_vertical = abs(arm.wrist[1] - arm.elbow[1])
        wrist_horizontal = abs(arm.wrist[0] - arm.elbow[0])

        if wrist_horizontal < 1.2 * torso_len:
            if 40.0 < smoothed_angle < 140.0 and wrist_vertical > 0.8 * torso_len:
                wrist_ok = False
                messages.append("Stack your wrist above your forearm; avoid dropping it.")
                good_form = False

        if not messages:
            if motion >= self._min_motion_deg:
                messages.append("Good curl mechanics in this frame.")
            else:
                messages.append("Static hold – posture looks solid.")
            good_form = True

        return {
            "elbow_angle": smoothed_angle,
            "rep_count": self.rep_count,
            "good_form": good_form,
            "messages": messages,
            "rom_ok": rom_ok,
            "torso_stable": torso_stable,
            "elbow_near_torso": elbow_near_torso,
            "wrist_ok": wrist_ok,
        }


def draw_pose_and_feedback(
    frame: np.ndarray,
    keypoints_xy: Optional[np.ndarray],
    feedback_left: Dict,
    feedback_right: Dict,
    active_side: str,
) -> np.ndarray:
    """
    Draw:
    - full-body skeleton
    - BOTH arms highlighted
    - left/right angles + reps in text
    - indicate which side is considered more "active" for this frame
    """
    if keypoints_xy is not None and keypoints_xy.shape[0] >= 17:
        # skeleton edges
        edges = [
            (L_SHOULDER, R_SHOULDER),
            (L_SHOULDER, L_ELBOW),
            (L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW),
            (R_ELBOW, R_WRIST),
            (L_SHOULDER, L_HIP),
            (R_SHOULDER, R_HIP),
            (L_HIP, R_HIP),
            (L_HIP, L_KNEE),
            (L_KNEE, L_ANKLE),
            (R_HIP, R_KNEE),
            (R_KNEE, R_ANKLE),
            (NOSE, L_SHOULDER),
            (NOSE, R_SHOULDER),
        ]

        # generic skeleton (thin grey)
        for i, j in edges:
            try:
                p1 = tuple(map(int, keypoints_xy[i]))
                p2 = tuple(map(int, keypoints_xy[j]))
            except Exception:
                continue
            cv2.line(frame, p1, p2, (190, 190, 190), 2)

        # keypoints as small dots
        for idx in range(17):
            try:
                p = tuple(map(int, keypoints_xy[idx]))
                cv2.circle(frame, p, 3, (0, 255, 255), -1)
            except Exception:
                continue

        # highlight BOTH arms: left and right
        # left arm: cyan
        try:
            ls = tuple(map(int, keypoints_xy[L_SHOULDER]))
            le = tuple(map(int, keypoints_xy[L_ELBOW]))
            lw = tuple(map(int, keypoints_xy[L_WRIST]))
            cv2.circle(frame, ls, 6, (255, 255, 0), -1)
            cv2.circle(frame, le, 6, (255, 255, 0), -1)
            cv2.circle(frame, lw, 6, (255, 255, 0), -1)
            cv2.line(frame, ls, le, (255, 255, 0), 3)
            cv2.line(frame, le, lw, (255, 255, 0), 3)
        except Exception:
            pass

        # right arm: green
        try:
            rs = tuple(map(int, keypoints_xy[R_SHOULDER]))
            re = tuple(map(int, keypoints_xy[R_ELBOW]))
            rw = tuple(map(int, keypoints_xy[R_WRIST]))
            cv2.circle(frame, rs, 6, (0, 255, 0), -1)
            cv2.circle(frame, re, 6, (0, 255, 0), -1)
            cv2.circle(frame, rw, 6, (0, 255, 0), -1)
            cv2.line(frame, rs, re, (0, 255, 0), 3)
            cv2.line(frame, re, rw, (0, 255, 0), 3)
        except Exception:
            pass

    # ----- text overlays: show BOTH sides -----
    y = 24

    left_angle = feedback_left.get("elbow_angle")
    right_angle = feedback_right.get("elbow_angle")
    left_rep = feedback_left.get("rep_count", 0)
    right_rep = feedback_right.get("rep_count", 0)

    cv2.putText(
        frame,
        f"Active side: {active_side}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    y += 26

    cv2.putText(
        frame,
        f"Reps L/R: {left_rep} / {right_rep}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    y += 26

    if left_angle is not None or right_angle is not None:
        la = f"{left_angle:.1f}" if left_angle is not None else "-"
        ra = f"{right_angle:.1f}" if right_angle is not None else "-"
        cv2.putText(
            frame,
            f"Elbow angle L/R: {la} / {ra}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        y += 26

    # short feedback line for each side (first message only)
    msg_left = feedback_left.get("messages", [""])[0]
    msg_right = feedback_right.get("messages", [""])[0]

    if msg_left:
        cv2.putText(
            frame,
            f"L: {msg_left}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
        )
        y += 22

    if msg_right:
        cv2.putText(
            frame,
            f"R: {msg_right}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
        )
        y += 22

    return frame
