## MAKITEST : Suspicious Exam Behaviour Detection System

This serves as a guide to the usage, setup, and system framework and architecture of the dual gaze tracking system of MAKITEST. The system monitors exam takers via webcam, tracking head pose and eye gaze to detect and log suspicious behaviours in real time.

---

## Step-by-Step Setup

### Prerequisites
- **Python 3.10 or higher** — [Download Python](https://www.python.org/downloads/)
- **A webcam** connected and accessible (index `0` or `1`)
- **Windows OS** recommended (the `keyboard` library requires it for global hotkey hooks)

---

### Step 1 — Clone or Download the Project

Download all project files into a single folder. Ensure the following files are present in the same directory:

```
main_trackerprocess.py          ← entry point (run this)
dataset_builder.py              ← build ML features from saved sessions
train_model.py                  ← train and save the suspicion classifier
predict_session.py              ← classify one saved session
heatmap_feature_extractor.py    ← extract heatmap and behaviour features
face_trackprocessor.py
face_distanceprocessor.py
face_axisprocessor.py
eye_trackprocessor.py
eye_gazeprocessor.py
eye_calibrationprocessor.py
eye_screenposprocessor.py
gaze_directionprocessor.py
suspicion_scoringprocessor.py
heatmap_processor.py
keypress_trackprocessor.py
frame_bufferprocessor.py
face_landmarker.task          ← required MediaPipe model file
requirements.txt
```

> **Important:** The `face_landmarker.task` model file must be present in the same directory as the scripts. Without it, face and eye detection will not function.

---

### Step 2 — Create a Virtual Environment (Recommended)

Open a terminal in the project folder and run:

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

---

### Step 3 — Install Dependencies

With the virtual environment active, install all required packages:

```bash
pip install -r requirements.txt
```

This installs the following libraries:

| Package | Purpose |
|---|---|
| `mediapipe` | Face and eye landmark detection model |
| `opencv-python` / `opencv-contrib-python` | Webcam capture and frame rendering |
| `numpy` | Numerical computation and array operations |
| `matplotlib` | Plotting and heatmap visualisation |
| `keyboard` | Global keyboard hotkey monitoring |
| `sounddevice` | Audio support (reserved for future alerts) |
| `pillow` | Image processing utilities |
| `pandas` | Tabular feature construction and model input |
| `scikit-learn` | Random forest training, calibration, validation, and metrics |
| `scipy` | Efficient heatmap colour-map lookup with `cKDTree` |
| `joblib` | Save and load the trained model |

---

### Step 4 — Verify Webcam Index

Open `main_trackerprocess.py` and locate this line near the top:

```python
cap = cv2.VideoCapture(1)
```

Change the index to match your camera:
- `0` — default built-in webcam
- `1` — secondary or external webcam

---

### Step 5 — Run the Program

Run the main tracker from the terminal:

```bash
python main_trackerprocess.py
```

Before tracking starts, select the session type when prompted:
- `1` — non-cheating session
- `2` — cheating session

Session files are saved in a uniquely named folder under the selected category:

```
sessions/
├── non_cheating/
│   └── session_<id>/
└── cheating/
      └── session_<id>/
```

Two windows will open:
- **Main feed** — shows the live webcam with face/eye landmarks overlaid
- **Screen position debug view** — shows the estimated gaze position on a virtual screen grid

---

### Step 6 — Calibrate the System

Calibration must be performed before gaze tracking is accurate. Follow the on-screen calibration stages:

| Key | Action |
|---|---|
| `{` | Advance to the next calibration stage (look at the indicated target point) |
| `}` | Exit the program |

**Calibration stages (in order):**
1. **Stage 0** — Look straight at the camera (center baseline)
2. **Stage 1** — Look up
3. **Stage 2** — Look down
4. **Stage 3** — Look left
5. **Stage 4** — Look right

At each stage, press `{` while holding your gaze on the target. The system collects 60 frames of samples per stage and saves the result to `calibration_data.json` automatically. On subsequent runs, saved calibration data is loaded from this file.

---

### Step 7 — Review Session Output

After the session ends (`}` to quit), the following output files are generated in the project folder:

| File | Description |
|---|---|
| `session_log_<timestamp>.csv` | Event-based log of all gaze directions, violations, and durations |
| `session_heatmap_<timestamp>.png` | Colour-mapped heatmap of screen gaze positions over the session |
| `<timestamp>_violation_<reason>.mp4` | Video evidence with pre-roll and post-roll around a suspicious event |

Each session is self-contained under its category directory, which allows the
same files to be reviewed manually or used by the machine-learning pipeline.

---

## System Architecture

The system is composed of **13 modules**, each responsible for a distinct processing stage. Data flows from the webcam through detection, tracking, calibration, scoring, and output layers.

```
Webcam → FrameBufferProcessor
           ↓
     FaceTrackProcessor → FaceDistanceProcessor
           ↓                       ↓
     FaceAxisProcessor       EyeTrackProcessor
           ↓                       ↓
     GazeDirectionProcessor ← EyeGazeProcessor
           ↓                       ↓
     EyeScreenPosProcessor ← EyeCalibrationProcessor
           ↓
     SuspicionScoringProcessor → CSV log / Video clips
           ↓
     HeatmapProcessor → Heatmap image
     KeypressTrackProcessor → Suspicious key log
```

---

## Module Reference

### `main_trackerprocess.py` — Main Tracker Process *(entry point)*
The entry point of the application. Initialises OpenCV's camera feed, instantiates every processor module, and runs the main loop that reads frames, calls each processor in sequence, and renders the results. Press `{` to step through calibration and `}` to exit.

---

### `face_trackprocessor.py` — Face Track Processor
Loads the `face_landmarker.task` MediaPipe model and runs face landmark detection on each video frame. Detects up to 1 face and returns the full set of 478 3D facial landmarks. Also tracks a smoothed gaze ray direction by averaging recent face normal vectors, which is fed into the axis and gaze processors.

---

### `face_distanceprocessor.py` — Face Distance Processor
Estimates the physical distance (in centimetres) between the subject and the camera. It uses a second-degree polynomial fitted (`numpy.polyfit`) to a set of manually measured iris-pixel-width vs. distance calibration samples. Given the pixel distance between the left and right iris centers, it returns a centimetre distance estimate.

<img width="720" height="540" alt="image" src="https://github.com/user-attachments/assets/8a1238e4-8496-4b4b-9eac-b3f6588d700e" />
<br><br>

Using the second degree of polynomial gives the process a great balance between efficiency and accuracy. This is implemented by gathering the distance of the irises with their corresponding distance from the camera. The process then uses `polyfit()` to predict the values of input data outside the given samples.

---

### `face_axisprocessor.py` — Face Axis Processor
Computes the **yaw** (left/right rotation) and **pitch** (up/down tilt) of the head in degrees using five key facial landmarks: chin, nose tip, forehead, left cheek, and right cheek. Calibration offsets are applied so that looking straight ahead reads as 0°/0°. The resulting angles are mapped to an estimated screen position using configurable degree limits (default: ±25° yaw, ±12° pitch).

#### Face-Position Anchor Offset
When converting head-pose angles to a screen coordinate, the processor also accounts for where the face sits within the camera frame. Each frame, `update_anchor_from_face_position()` receives the face centre in pixels and the frame dimensions. It normalises the face position to a `[-0.5, 0.5]` range relative to the frame centre, then scales that deviation directly onto screen space (1920×1080):

```
norm_x = (face_center_x / frame_width)  - 0.5
norm_y = (face_center_y / frame_height) - 0.5

anchor_offset_x = norm_x * 1920
anchor_offset_y = norm_y * 1080
```

These offsets are added to the angle-derived screen position inside `get_estimated_screen_position()` before the result is clamped to the screen bounds. A face perfectly centred in the camera frame produces zero offset; a face displaced to the right shifts the estimated gaze position right proportionally, correcting for the parallax-like error that would otherwise occur when the subject is not centred in front of the camera.

---

### `eye_trackprocessor.py` — Eye Track Processor
Uses a second instance of the MediaPipe `FaceLandmarker` model to isolate and track detailed eye geometry. Extracts upper/lower eyelid contours, iris boundary points, and pupil center positions for both eyes. Passes the raw eye geometry to `EyeGazeProcessor` for normalised gaze computation, and draws the eye bounding boxes and landmark points on the frame.

---

### `eye_gazeprocessor.py` — Eye Gaze Processor
Takes the raw eye landmark coordinates from `EyeTrackProcessor` and computes two key measurements per eye:
- **Iris position** — the normalised horizontal position of the iris center within the eye bounding box (0 = far left, 1 = far right)
- **Iris box height** — the pixel distance between the upper and lower eyelid markers, used to infer vertical gaze and detect blinks

These values are returned as `raw_eye_data` which feeds into the calibration and gaze direction processors.

---

### `eye_calibrationprocessor.py` — Eye Calibration Processor
Manages a multi-stage calibration sequence that records baseline eye measurements for center, up, down, left, and right gaze positions. For each stage it collects a rolling window of 60 samples and averages them into threshold values. The calibrated thresholds are stored in a dictionary and saved to `calibration_data.json` so recalibration is not required on every run.

---

### `eye_screenposprocessor.py` — Eye Screen Position Processor
Maps the calibrated iris position values to absolute screen pixel coordinates (default: 1920×1080). Applies exponential moving average (EMA) smoothing with a configurable time constant (`smoothing_tau = 0.18 s`) to eliminate jitter. Also classifies the screen into a 3×3 grid of named regions (e.g. "Up Left", "Center Center", "Down Right") for use by the suspicion scorer.

---

### `gaze_directionprocessor.py` — Gaze Direction Processor
Fuses the face-axis-based screen position and the eye-based screen position into a single weighted estimate. Default weights are 45% face axis and 55% eye gaze. Also classifies the combined gaze into a discrete direction string (`"Up"`, `"Down"`, `"Left"`, `"Right"`, `"Center"`) using calibrated thresholds and dead zones to prevent jitter.

---

### `suspicion_scoringprocessor.py` — Suspicion Scoring Processor
The behavioural analysis engine. Monitors gaze direction and raises suspicion events based on configurable rules:

| Trigger | Default Threshold |
|---|---|
| Looking to the side | > 3.0 seconds |
| Looking down | > 5.0 seconds |
| Face off-screen | > 1.5 seconds |
| Rapid gaze shifts | > 6 shifts / 10 seconds |

When a violation is detected, it:
- Logs the event with timestamps to a CSV file (`session_log_<timestamp>.csv`)
- Saves a video clip (3-second pre-roll + post-roll) to an AVI file
- Tracks and reports suspicious keyboard shortcuts detected by `KeypressTrackProcessor`

---

### `heatmap_processor.py` — Heatmap Processor
Accumulates all gaze screen-position points recorded during the session. On exit, generates a Gaussian-blurred heatmap image showing where the subject looked most frequently. The output is saved as a colour-mapped PNG file (`session_heatmap_<timestamp>.png`).

---

### `keypress_trackprocessor.py` — Keypress Track Processor
Registers global keyboard hooks (using the `keyboard` library) to detect suspicious hotkey combinations without needing focus on the application window. Monitored combos include:

- `Alt+Tab`, `Ctrl+C/V/X` — switching and clipboard
- `Windows`, `Ctrl+Esc` — start menu / app switching
- `Ctrl+T/N/W`, `F11`, `Esc` — browser actions
- `Print Screen`, `Win+Shift+S` — screen capture
- `Alt+F4` — application evasion

Detections are stored in a thread-safe buffer and retrieved each frame by the main loop.

---

### `frame_bufferprocessor.py` — Frame Buffer Processor
Runs a dedicated background thread that continuously reads frames from the OpenCV `VideoCapture` object and stores the latest frame in a thread-safe buffer. This decouples frame acquisition from the processing loop, preventing the main loop from blocking on camera I/O and ensuring the freshest possible frame is always available.

---

## Machine-Learning Pipeline

The repository includes a session-level classifier that combines spatial gaze
patterns from the heatmap with behavioural events from the suspicion-scoring
CSV. Sessions must contain both a `session_heatmap_*.png` file and a
`session_log_*.csv` file to be included.

### Build the feature dataset

After collecting labelled sessions under `sessions/cheating/` and
`sessions/non_cheating/`, run:

```bash
python dataset_builder.py
```

`dataset_builder.py` scans both category folders and writes `features.csv`.
Each row represents one session and contains the extracted features, the
session ID, and the numeric label (`1` for cheating, `0` for non-cheating).
Sessions missing either required artifact are skipped.

### Train the model

```bash
python train_model.py
```

Training uses a calibrated `RandomForestClassifier` with five-fold stratified
cross-validation. It prints cross-validated accuracy, ROC-AUC, and a
classification report, then saves:

| File | Description |
|---|---|
| `suspicion_model.joblib` | Trained calibrated classifier used for inference |
| `feature_columns.json` | Feature names and the order expected by the model |

Run training again whenever `features.csv` changes.

### Predict one session

```bash
python predict_session.py sessions/cheating/session_<id>
```

The predictor loads the saved model and feature-column order, extracts the
session features, supplies them as a named pandas `DataFrame`, and prints the
predicted label and confidence. It returns `(label, confidence)` when called
from Python, or `None` when the session is missing a heatmap or CSV log.

## Feature Extraction Reference

The functions in `heatmap_feature_extractor.py` can also be imported directly:

| Function | Purpose |
|---|---|
| `build_jet_lut()` | Build the lookup table for OpenCV's JET colour map |
| `recover_intensity_map(heatmap_img)` | Recover a density surrogate from a saved heatmap image |
| `extract_heatmap_features(intensity_map, grid_size=8)` | Calculate centroid, spread, elongation, entropy, peak, coverage, and grid-cell features |
| `extract_csv_features(csv_path)` | Calculate violation counts, gaze transitions, non-centre time, and violation rate |
| `extract_session_features(session_dir)` | Combine heatmap and CSV features for one session |

The generated heatmap features include normalised centroid coordinates,
normalised spread on both axes, an elongation ratio, entropy, peak ratio,
coverage ratio, and an 8x8 occupancy grid. CSV features include counts for
`frantic_eye_movement`, `forbidden_key`, `off_screen`, and `duration`, plus
transition count, percentage of non-centre gaze time, and violation rate.

## Python API Summary

The main callable entry points are:

| Function | Module | Returns |
|---|---|---|
| `build_dataset()` | `dataset_builder.py` | A pandas `DataFrame`, or `None` when no usable sessions are found |
| `train_model()` | `train_model.py` | Saves the trained model and feature-column metadata |
| `predict_session(session_dir)` | `predict_session.py` | `(label, confidence)`, or `None` for incomplete sessions |
| `FaceAxisProcessor.process(avg_direction)` | `face_axisprocessor.py` | Calibrated yaw and pitch |
| `FaceAxisProcessor.get_estimated_screen_position()` | `face_axisprocessor.py` | Clamped `(x, y)` screen coordinates |
| `GazeDirectionProcessor.weighted_screen_position(face, eye)` | `gaze_directionprocessor.py` | Blended screen position and 3x3 grid label |
| `HeatmapProcessor.add_point(screen_pos)` | `heatmap_processor.py` | Adds a bounded gaze point to the session |
| `HeatmapProcessor.generate_heatmap()` | `heatmap_processor.py` | Saved heatmap path, or `None` when no points exist |
| `SuspicionScoringProcessor.update(...)` | `suspicion_scoringprocessor.py` | Current recording and violation state |

