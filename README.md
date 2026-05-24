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
| `heatmap_<timestamp>.png` | Visual heatmap of screen gaze positions over the session |
| `clip_<timestamp>.avi` | Video clip recorded around any detected suspicious behaviour event |

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
Accumulates all gaze screen-position points recorded during the session. On exit, generates a Gaussian-blurred heatmap image showing where the subject looked most frequently. The output is saved as a colour-mapped PNG file (`heatmap_<timestamp>.png`).

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

