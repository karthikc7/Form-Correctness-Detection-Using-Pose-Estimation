# rules_bicep_curl.py
#
# Logic for per–frame bicep–curl analysis:
#   - per–arm elbow angle
#   - simple posture rules
#   - independent rep counter for left & right
#
# This file is written from scratch to avoid plagiarism.

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import math
import numpy as np


# COCO indices used by YOLO pose
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12


def _get_xy(keypoints: np.ndarray, idx: int) -> Optional[np.ndarray]:
    """
    Safe accessor for a 2D point from YOLO keypoints.
    Supports shapes (17, 2) or (17, 3).
    Returns None if index is invalid or coordinates look empty.
    """
    if keypoints is None:
        return None
    if idx < 0 or idx >= keypoints.shape[0]:
        return None

    p = keypoints[idx]
    if p is None:
        return None

    # YOLO: (x, y, conf) or (x, y)
    if len(p) >= 2:
        x, y = float(p[0]), float(p[1])
        # crude check against missing values
        if x == 0.0 and y == 0.0:
            return None
        return np.array([x, y], dtype=np.float32)
    return None


def _angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle in degrees at joint b between segments (a-b) and (c-b).
    Returns value in [0, 180].
    """
    ba = a - b
    bc = c - b

    # avoid zero-length vectors
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return 180.0

    cos_ang = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    cos_ang = max(-1.0, min(1.0, cos_ang))
    return math.degrees(math.acos(cos_ang))


@dataclass
class ArmFeedback:
    elbow_angle: float
    rep_count: int
    rom_ok: bool
    torso_stable: bool
    elbow_near_torso: bool
    wrist_ok: bool
    good_form: bool
    messages: str


class _SingleArmRepTracker:
    """
    Simple, robust rep counter:
      - 'down'  = arm almost straight (large angle)
      - 'up'    = arm flexed (small angle)
      - one full down→up→down cycle = +1 rep
    """

    def __init__(self, up_thresh: float = 60.0, down_thresh: float = 140.0):
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.state = "down"   # 'down' or 'up'
        self.rep_count = 0

    def update(self, angle: Optional[float]) -> Tuple[int, bool]:
        """
        Update internal state with the current elbow angle.
        Returns (current_rep_count, rep_incremented_this_frame).
        """
        if angle is None:
            return self.rep_count, False

        inc = False

        if self.state == "down":
            # arm flexes upward
            if angle < self.up_thresh:
                self.state = "up"
        elif self.state == "up":
            # arm extends again -> finished one rep
            if angle > self.down_thresh:
                self.rep_count += 1
                inc = True
                self.state = "down"

        return self.rep_count, inc


class BicepCurlEvaluator:
    """
    High-level evaluator for a single person doing curls.
    - Tracks LEFT and RIGHT arms independently.
    - Provides angles, multiple form checks, and rep counts.
    - Does NOT depend on 'active_side' to count reps.
    """

    def __init__(
        self,
        up_thresh_deg: float = 60.0,
        down_thresh_deg: float = 140.0,
        torso_tolerance_deg: float = 15.0,
        elbow_torso_ratio_max: float = 0.45,
        wrist_drift_ratio_max: float = 0.6,
    ):
        self.left_tracker = _SingleArmRepTracker(up_thresh_deg, down_thresh_deg)
        self.right_tracker = _SingleArmRepTracker(up_thresh_deg, down_thresh_deg)

        self.up_thresh_deg = up_thresh_deg
        self.down_thresh_deg = down_thresh_deg
        self.torso_tolerance_deg = torso_tolerance_deg
        self.elbow_torso_ratio_max = elbow_torso_ratio_max
        self.wrist_drift_ratio_max = wrist_drift_ratio_max

        # Baseline torso orientation from the first valid frame
        self._baseline_torso_angle: Optional[float] = None

        # For "which arm is moving more" visual highlight
        self._prev_left_angle: Optional[float] = None
        self._prev_right_angle: Optional[float] = None

    # ---- Core helpers -----------------------------------------------------

    def _torso_angle(self, kps: np.ndarray) -> Optional[float]:
        l_sh = _get_xy(kps, L_SHOULDER)
        r_sh = _get_xy(kps, R_SHOULDER)
        l_hip = _get_xy(kps, L_HIP)
        r_hip = _get_xy(kps, R_HIP)

        if any(p is None for p in (l_sh, r_sh, l_hip, r_hip)):
            return None

        shoulder_mid = 0.5 * (l_sh + r_sh)
        hip_mid = 0.5 * (l_hip + r_hip)
        vec = hip_mid - shoulder_mid  # torso vector

        if np.linalg.norm(vec) < 1e-6:
            return None

        # angle of torso vector relative to vertical axis (0 = perfectly upright)
        # vertical axis is (0, 1)
        dot = float(np.dot(vec, np.array([0.0, 1.0], dtype=np.float32)))
        cos_ang = dot / (np.linalg.norm(vec))
        cos_ang = max(-1.0, min(1.0, cos_ang))
        angle = math.degrees(math.acos(cos_ang))
        return angle

    def _torso_stable(self, kps: np.ndarray) -> bool:
        ang = self._torso_angle(kps)
        if ang is None:
            return True  # if we can't tell, don't penalise

        if self._baseline_torso_angle is None:
            self._baseline_torso_angle = ang
            return True

        return abs(ang - self._baseline_torso_angle) <= self.torso_tolerance_deg

    def _elbow_near_torso(self, shoulder: np.ndarray, elbow: np.ndarray) -> bool:
        upper = elbow - shoulder
        upper_len = float(np.linalg.norm(upper)) + 1e-6
        horizontal_dist = abs(elbow[0] - shoulder[0])
        return (horizontal_dist / upper_len) <= self.elbow_torso_ratio_max

    def _wrist_control_ok(self, elbow: np.ndarray, wrist: np.ndarray) -> bool:
        upper = wrist - elbow
        upper_len = float(np.linalg.norm(upper)) + 1e-6

        # How far does wrist drift forward/back relative to elbow?
        horizontal_drift = abs(wrist[0] - elbow[0])
        return (horizontal_drift / upper_len) <= self.wrist_drift_ratio_max

    # ---- Public API -------------------------------------------------------

    def evaluate_frame(
        self,
        keypoints: np.ndarray,
        dt: float,
    ) -> Dict[str, Any]:
        """
        Main entry point used by your video loop.

        Returns dict:
        {
          "active_side": "left" | "right" | "none",
          "left": ArmFeedback,
          "right": ArmFeedback,
        }
        """
        # ----- Torso & shared checks -----
        torso_stable = self._torso_stable(keypoints)

        # ----- Left arm -----
        l_sh = _get_xy(keypoints, L_SHOULDER)
        l_el = _get_xy(keypoints, L_ELBOW)
        l_wr = _get_xy(keypoints, L_WRIST)
        left_angle = None
        left_rom_ok = False
        left_elbow_near_torso = False
        left_wrist_ok = False
        left_msgs = []

        if l_sh is not None and l_el is not None and l_wr is not None:
            left_angle = _angle_at_joint(l_sh, l_el, l_wr)

            # ROM check
            left_rom_ok = (
                left_angle < self.up_thresh_deg
                or left_angle > self.down_thresh_deg
            )

            # elbow close to torso
            left_elbow_near_torso = self._elbow_near_torso(l_sh, l_el)

            # wrist drift
            left_wrist_ok = self._wrist_control_ok(l_el, l_wr)

            if not left_rom_ok:
                left_msgs.append(
                    "Increase curl depth for better range of motion."
                )
            if not left_elbow_near_torso:
                left_msgs.append("Keep elbow closer to your side.")
            if not left_wrist_ok:
                left_msgs.append("Keep the wrist neutral, avoid bending.")
        else:
            left_msgs.append("Left arm keypoints not reliable in this frame.")

        # rep logic (always run if we have an angle)
        left_rep_count, _ = self.left_tracker.update(left_angle)
        left_good_form = bool(
            left_rom_ok and torso_stable and left_elbow_near_torso and left_wrist_ok
        )
        if left_good_form and not left_msgs:
            left_msgs.append("Good curl mechanics in this frame.")

        # ----- Right arm -----
        r_sh = _get_xy(keypoints, R_SHOULDER)
        r_el = _get_xy(keypoints, R_ELBOW)
        r_wr = _get_xy(keypoints, R_WRIST)
        right_angle = None
        right_rom_ok = False
        right_elbow_near_torso = False
        right_wrist_ok = False
        right_msgs = []

        if r_sh is not None and r_el is not None and r_wr is not None:
            right_angle = _angle_at_joint(r_sh, r_el, r_wr)

            right_rom_ok = (
                right_angle < self.up_thresh_deg
                or right_angle > self.down_thresh_deg
            )

            right_elbow_near_torso = self._elbow_near_torso(r_sh, r_el)
            right_wrist_ok = self._wrist_control_ok(r_el, r_wr)

            if not right_rom_ok:
                right_msgs.append(
                    "Increase curl depth for better range of motion."
                )
            if not right_elbow_near_torso:
                right_msgs.append("Keep elbow closer to your side.")
            if not right_wrist_ok:
                right_msgs.append("Keep the wrist neutral, avoid bending.")
        else:
            right_msgs.append("Right arm keypoints not reliable in this frame.")

        right_rep_count, _ = self.right_tracker.update(right_angle)
        right_good_form = bool(
            right_rom_ok and torso_stable and right_elbow_near_torso and right_wrist_ok
        )
        if right_good_form and not right_msgs:
            right_msgs.append("Good curl mechanics in this frame.")

        # ----- Decide which side is "active" for drawing highlight -----
        active_side = "none"
        if left_angle is not None or right_angle is not None:
            # use angle *change* to detect movement
            dl = (
                abs(left_angle - self._prev_left_angle)
                if (left_angle is not None and self._prev_left_angle is not None)
                else 0.0
            )
            dr = (
                abs(right_angle - self._prev_right_angle)
                if (right_angle is not None and self._prev_right_angle is not None)
                else 0.0
            )
            if dl > dr and dl > 1.0:
                active_side = "left"
            elif dr > dl and dr > 1.0:
                active_side = "right"

        self._prev_left_angle = left_angle if left_angle is not None else self._prev_left_angle
        self._prev_right_angle = right_angle if right_angle is not None else self._prev_right_angle

        return {
            "active_side": active_side,
            "left": ArmFeedback(
                elbow_angle=float(left_angle) if left_angle is not None else float("nan"),
                rep_count=left_rep_count,
                rom_ok=bool(left_rom_ok),
                torso_stable=bool(torso_stable),
                elbow_near_torso=bool(left_elbow_near_torso),
                wrist_ok=bool(left_wrist_ok),
                good_form=left_good_form,
                messages=" ".join(left_msgs),
            ),
            "right": ArmFeedback(
                elbow_angle=float(right_angle) if right_angle is not None else float("nan"),
                rep_count=right_rep_count,
                rom_ok=bool(right_rom_ok),
                torso_stable=bool(torso_stable),
                elbow_near_torso=bool(right_elbow_near_torso),
                wrist_ok=bool(right_wrist_ok),
                good_form=right_good_form,
                messages=" ".join(right_msgs),
            ),
        }

    # For older calls if your main file uses evaluator.evaluate(...)
    def evaluate(self, keypoints: np.ndarray, dt: float) -> Dict[str, Any]:
        return self.evaluate_frame(keypoints, dt)
