<div align="center">

# Smart Car Distance Measurement

A computer vision system that detects vehicles in dashcam footage and calculates real-time distance to the car in front. Built as a final year project at Universiti Malaysia Pahang, awarded Silver Medal at CITREX 2024.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv)](https://opencv.org)
[![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)](https://ultralytics.com)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)]()

</div>

---

## The Problem

Drivers have blind spots. At certain angles and distances, it is genuinely difficult to judge how far the car in front is. Dashcams record the footage but do nothing with it. This system processes that footage in real time, detects the vehicle ahead, and overlays the estimated distance directly on the frame.

---

## What It Does

| Step | What Happens |
|---|---|
| Frame capture | Video is read frame by frame via OpenCV VideoCapture |
| Vehicle detection | YOLOv8 identifies cars within a trapezium Region of Interest |
| Car extraction | Detected car bounding box is cropped and isolated |
| Grayscale conversion | RGB to greyscale to simplify edge processing |
| Sobel edge detection | Horizontal and vertical gradients highlight car boundaries |
| Contour extraction | Largest contour is selected as the car outline |
| Centre point | Centroid of bounding box is calculated and marked |
| Distance calculation | Perspective projection formula using known car width (1.67m) and focal length (683.43) |
| Status overlay | Colour-coded bounding box and status message (Normal / Precaution / Caution) displayed on frame |

**Distance formula:**
```
Distance = (Car Width in pixels × Focal Length) / Actual Car Width (1.67m)
```

---

## Stack

**Language** — Python 3.x

**Libraries** — OpenCV, NumPy, Ultralytics (YOLOv8), cvzone

**Hardware** — 70mai M300 dashcam, AMD Ryzen 3 laptop

**Dataset** — Self-recorded dashcam footage at Pekan City, sunny conditions, car-only subjects

---

## Project Structure

```
smart-car-distance/
├── main.py                  # Entry point, frame loop, orchestration
├── Yolo-Weights/
│   └── yolov8l.pt           # YOLOv8 large model weights
├── datasets/
│   └── dataset_*.mp4        # Dashcam video files
└── output_result/
    ├── videos/              # Processed output videos (.avi)
    ├── frames/              # Saved frames per dataset
    ├── ori_image/           # Extracted car images per frame
    └── distance_data/       # Distance logs in CSV format
```

---

## Running Locally

### Prerequisites

- Python 3.8+
- pip

### Install dependencies

```bash
pip install opencv-python numpy ultralytics cvzone
```

### Run

```bash
# Edit the dataset path and dir_name at the bottom of main.py first
# cap = cv2.VideoCapture('datasets/dataset_3.mp4')
# dir_name = "dataset3"

python main.py
```

Output files are saved automatically to `output_result/`.

**Keyboard controls while running:**
```
q   Quit
p   Pause / unpause
```

---

## Results

### Object Tracking Accuracy (MOTA)

| Dataset | MOTA (%) | MOTP (%) | Misses (%) | FP (%) |
|---|---|---|---|---|
| dataset_1 | 83 | 82.43 | 12 | 3 |
| dataset_2 | 97 | 80.52 | 2 | 0 |
| dataset_3 | 84 | 93.76 | 7 | 8 |
| dataset_5 | 96 | 90.94 | 3 | 1 |
| dataset_7 | 98 | 81.15 | 1 | 1 |

Best: dataset_7 at 98% MOTA. Lowest: dataset_1 at 83%, mainly due to occlusion and crowded scenes.

### Distance Measurement Accuracy

| Dataset | Angle | Measured (m) | Actual (m) | Error (%) |
|---|---|---|---|---|
| dataset_10 | 98° | 6.24 | 6.22 | 0.32 |
| dataset_11 | 134° | 3.22 | 4.00 | 24.22 |
| dataset_12 | 120° | 4.14 | 4.64 | 12.08 |
| dataset_13 | 72° | 4.17 | 4.58 | 9.83 |
| dataset_14 | 66° | 2.58 | 2.85 | 10.47 |

Error range: 0.32% to 24.22%. Higher error at wide angles where the car width in frame deviates significantly from the calibrated reference.

---

## Demo
## Demo

[![Demo Video](https://img.youtube.com/vi/ukzGumsVJ34/maxresdefault.jpg)](https://www.youtube.com/watch?v=ukzGumsVJ34)

---

## What I Learned

- Sobel is computationally lighter than Canny and fast enough for frame-by-frame processing on a low-spec laptop, but it is noise-sensitive and produces thicker edges which affects contour precision
- YOLOv8 confidence threshold at 0.6 was the sweet spot — lower caused false positives on background objects, higher caused missed detections at distance
- The perspective projection formula works well when the car is facing squarely toward the camera. Angle deviation above roughly 120 degrees causes significant error because the visible width no longer represents the full car width
- Contour selection by maximum area is a simple heuristic that works most of the time but fails when background objects have larger contours than the car

## What Needs Improving

- The focal length (683.43) was manually calibrated for one specific camera setup. A proper camera calibration step would make this portable to other dashcams
- Wide-angle detection accuracy is poor, a depth estimation model or stereo camera setup would handle this better
- No night or rain condition testing, the system was only validated in sunny daytime conditions
- Processing is frame-by-frame with no temporal tracking, so the distance reading flickers when detection briefly fails. A Kalman filter for temporal smoothing would help
- Low-spec hardware (Ryzen 3) caused slow processing. The pipeline is not optimised for real-time use yet

---

## Awards

- Silver Medal, CITREX 2024, UMP Innovation and Research Exhibition
- Top 20 Finalist, FYPro-Com 2024, national final year project competition

---

## Docs

| Document | File |
|---|---|
| Full Thesis | `fyp.pptx` |
| Source Code | `main.py` |

---

<div align="center">
Final Year Project, Bachelor of Computer Science (Software Engineering), UMP 2024.
</div>
