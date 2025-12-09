"""
End-to-end pipeline for the Smartan bicep curl form-analysis task.

Now evaluates BOTH arms per frame:
- chooses main person via bounding box area
- runs a separate BicepCurlEvaluator for left and right arms
- overlays feedback for BOTH arms (angles + reps),
  while still indicating which side is more "active"
- writes per-arm + per-rule metrics to CSV
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from pose_utils import (
    BicepCurlEvaluator,
    draw_pose_and_feedback,
    slice_arm_pose,
)


def _pick_person(
    keypoints_xy: np.ndarray,
    boxes_xyxy: Optional[np.ndarray],
) -> int:
    """
    Decide which person to track.

    Strategy:
    - if multiple persons are present, pick the one with the largest bounding box
      (closest to camera)
    """
    num_people = keypoints_xy.shape[0]

    if boxes_xyxy is not None and len(boxes_xyxy) == num_people:
        areas = []
        for x1, y1, x2, y2 in boxes_xyxy:
            areas.append(float((x2 - x1) * (y2 - y1)))
        person_idx = int(np.argmax(areas))
    else:
        person_idx = 0

    return person_idx


def analyse_bicep_curl_video(
    input_path: str,
    output_video_path: str,
    output_csv_path: str,
    model_path: str = "yolov8n-pose.pt",
) -> None:
    """Main driver function for the assignment."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    print(f"[INFO] Analysing video: {input_path}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    evaluator_left = BicepCurlEvaluator()
    evaluator_right = BicepCurlEvaluator()

    tracked_person_idx: Optional[int] = None
    records = []
    frame_idx = 0

    print("[INFO] Starting inference over frames...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, verbose=False)

        body_kp = None
        feedback_left = {
            "elbow_angle": None,
            "rep_count": evaluator_left.rep_count,
            "good_form": False,
            "messages": ["No person detected."],
            "rom_ok": False,
            "torso_stable": False,
            "elbow_near_torso": False,
            "wrist_ok": False,
        }
        feedback_right = {
            "elbow_angle": None,
            "rep_count": evaluator_right.rep_count,
            "good_form": False,
            "messages": ["No person detected."],
            "rom_ok": False,
            "torso_stable": False,
            "elbow_near_torso": False,
            "wrist_ok": False,
        }
        active_side = "right"

        if results and len(results) > 0:
            res = results[0]

            if res.keypoints is not None and len(res.keypoints) > 0:
                kp_all = res.keypoints.xy  # (num_people, 17, 2)
                kp_all = (
                    kp_all.cpu().numpy()
                    if hasattr(kp_all, "cpu")
                    else np.asarray(kp_all)
                )

                boxes = None
                if res.boxes is not None and len(res.boxes) > 0:
                    bx = res.boxes.xyxy
                    boxes = (
                        bx.cpu().numpy()
                        if hasattr(bx, "cpu")
                        else np.asarray(bx)
                    )

                if tracked_person_idx is None:
                    tracked_person_idx = _pick_person(kp_all, boxes)
                    print(f"[INFO] Tracking person #{tracked_person_idx}")

                person_index = min(tracked_person_idx, kp_all.shape[0] - 1)
                body_kp = kp_all[person_index]

                arm_left = slice_arm_pose(body_kp, "left")
                arm_right = slice_arm_pose(body_kp, "right")

                feedback_left = evaluator_left.step(arm_left)
                feedback_right = evaluator_right.step(arm_right)

                angle_left = feedback_left.get("elbow_angle")
                angle_right = feedback_right.get("elbow_angle")

                if angle_left is not None and angle_right is not None:
                    active_side = "left" if angle_left <= angle_right else "right"
                elif angle_left is not None:
                    active_side = "left"
                elif angle_right is not None:
                    active_side = "right"
                else:
                    active_side = "right"

        # ✅ draw BOTH arms + text for both sides
        frame = draw_pose_and_feedback(
            frame,
            body_kp,
            feedback_left,
            feedback_right,
            active_side,
        )
        video_writer.write(frame)

        records.append(
            {
                "frame_idx": frame_idx,
                "time_sec": frame_idx / fps,
                "active_side": active_side,
                # left arm metrics
                "left_elbow_angle": feedback_left.get("elbow_angle"),
                "left_rep_count": feedback_left.get("rep_count"),
                "left_good_form": feedback_left.get("good_form"),
                "left_rom_ok": feedback_left.get("rom_ok"),
                "left_torso_stable": feedback_left.get("torso_stable"),
                "left_elbow_near_torso": feedback_left.get("elbow_near_torso"),
                "left_wrist_ok": feedback_left.get("wrist_ok"),
                "left_messages": "; ".join(feedback_left.get("messages", [])),
                # right arm metrics
                "right_elbow_angle": feedback_right.get("elbow_angle"),
                "right_rep_count": feedback_right.get("rep_count"),
                "right_good_form": feedback_right.get("good_form"),
                "right_rom_ok": feedback_right.get("rom_ok"),
                "right_torso_stable": feedback_right.get("torso_stable"),
                "right_elbow_near_torso": feedback_right.get("elbow_near_torso"),
                "right_wrist_ok": feedback_right.get("wrist_ok"),
                "right_messages": "; ".join(feedback_right.get("messages", [])),
            }
        )

        frame_idx += 1

    cap.release()
    video_writer.release()

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    csv_path_used = output_csv_path
    try:
        df.to_csv(output_csv_path, index=False)
    except PermissionError:
        alt = output_csv_path.replace(".csv", "_run2.csv")
        df.to_csv(alt, index=False)
        csv_path_used = alt
        print(
            f"[WARN] Could not overwrite {output_csv_path} (file in use). "
            f"Saved metrics to {alt} instead."
        )

    print("[INFO] Finished.")
    print(f"[INFO] Annotated video saved at: {output_video_path}")
    print(f"[INFO] Metrics CSV saved at: {csv_path_used}")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # change this to sample_bicep_curl_04.mp4 if you want that by default
    input_video = os.path.join(root_dir, "data", "videos", "sample_bicep_curl_04.mp4")
    output_video = os.path.join(
        root_dir, "data", "output", "annotated_bicep_curl.mp4"
    )
    output_csv = os.path.join(root_dir, "data", "output", "bicep_curl_metrics.csv")

    analyse_bicep_curl_video(input_video, output_video, output_csv)
