# **Bicep Curl Form Analysis – Smartan.AI Internship Task**

This repository contains a complete computer-vision pipeline for analysing **bicep curl exercise form** using **YOLOv8 Pose**.
The system extracts human keypoints, evaluates biomechanics through rule-based logic, counts repetitions for **both arms**, renders real-time feedback, and exports per-frame metrics.

---

## **Features**

* Full-body pose estimation (YOLOv8 Pose)
* Independent **left & right arm** evaluation
* Elbow-angle calculation using joint geometry
* **4 posture rules** per arm: ROM depth, torso stability, elbow-torso alignment, wrist alignment
* Rep counting for each arm
* Real-time on-screen feedback (angles, reps, coaching messages)
* Annotated output video
* CSV export of all metrics (frame-wise)
* Automatic primary-person tracking in multi-person frames

---

## **Project Structure**

```
smartan_pose_task/
├─ src/
│  ├─ main_bicep_curl.py
│  ├─ pose_utils.py
│  ├─ rules_bicep_curl.py
├─ data/
│  ├─ videos/
│  └─ output/
├─ requirements.txt
└─ README.md
```

---

## **Installation**

```sh
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## **Usage**

Place your input video inside:

```
data/videos/
```

Run the pipeline:

```sh
python src/main_bicep_curl.py
```

Outputs are saved to:

```
data/output/annotated_bicep_curl.mp4
data/output/bicep_curl_metrics.csv
```

---

## **How It Works**

1. YOLOv8 Pose detects the person and extracts 17 keypoints.
2. Shoulder–elbow–wrist coordinates are used to compute elbow angles.
3. Four posture rules evaluate the quality of each curl.
4. Independent rep counters track left and right repetitions.
5. The system highlights both arms and overlays angles + feedback.
6. CSV logs include rep counts, angles, rule flags, and messages for every frame.

---

## **Posture Rules Implemented**

* **Range of Motion (ROM):** full extension → full flexion
* **Torso Stability:** detects torso sway / upper-body movement
* **Elbow Near Torso:** prevents elbow drifting forward
* **Wrist Alignment:** prevents wrist collapsing or bending

---

## **Deliverables Included**

* Python source code (clean, modular, no plagiarism)
* Full-body annotated output video
* Frame-wise analytics CSV
* Rule-based evaluation logic
* Independent rep-counting per arm

---


Just tell me **"generate the GitHub info"**.
