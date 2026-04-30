import cv2
import math
import numpy as np
import face_trackprocessor as ftp
import face_axisprocessor as fap
import eye_trackprocessor as etp
import eye_calibrationprocessor as ecp
import gaze_directionprocessor as gdp
import suspicion_scoringprocessor as ssp

processor = ftp.FaceLandmarkerProcessor()
axis_processor = fap.FaceAxisProcessor()
eye_processor = etp.EyeLandmarkerProcessor()
eye_calibrator = ecp.EyeCalibrationProcessor()
gaze_processor = gdp.GazeDirectionProcessor()
scoring_processor = ssp.SuspicionScoringProcessor()

cap = cv2.VideoCapture(0)
avg_direction = None
raw_eye_data = None
current_gaze = 'Center'
screen_x = None
screen_y = None

# Heatmap accumulator for gaze visualization
heatmap_sw, heatmap_sh = 1920 // 2, 1080 // 2
heatmap_accumulator = np.zeros((heatmap_sh, heatmap_sw), dtype=np.float32)
HEATMAP_DECAY = 0.997
HEATMAP_INTENSITY = 0.03

# DEBUGGING PURPOSES: This loop processes the video feed from the webcam, detects face and eye landmarks, 
# estimates head pose, and displays the results in real-time. 
# It also allows for calibration of the head pose estimation by pressing the 'c' key. 
# The estimated screen position is visualized on a separate frame for debugging purposes. 

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and avg_direction is not None and raw_eye_data is not None:
        if eye_calibrator.calibration_stage == -1:
            eye_calibrator.next_stage()
            axis_processor.calibrate(avg_direction)
            continue
        
        print("Calibrated! Current pose set as zero.")
        
        samples = []
        eye_calibrator.sample_count = 70  # Collect 70 samples for each calibration position to ensure stable calibration values.
        # Collect 70 samples for each calibration position (center, up, down, left, right) to ensure stable calibration values.
        while len(samples) < eye_calibrator.sample_count:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1280, 720))
            eye_frame = frame.copy()
            eye_results = eye_processor.process_frame(eye_frame)
            eye_frame, raw_eye_data = eye_processor._draw_landmarks(eye_frame, eye_results)
            
            if raw_eye_data is not None:
                samples.append(raw_eye_data)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        eye_calibrator.calibrate(samples)
        
        print(f"Calibration stage {eye_calibrator.calibration_stage} complete.")
        eye_calibrator.next_stage()
    

    elif key == ord('q'):
        break

    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1280, 720))

    # ====================== FACE PROCESSING ======================
    face_frame = frame.copy()
    results = processor.process_frame(face_frame)
    output_frame, avg_direction = processor._draw_landmarks(face_frame, results)
    if avg_direction is not None:
        screen_x, screen_y = None, None
        yaw, pitch = axis_processor.process(avg_direction)
        if yaw is not None and pitch is not None:
            cv2.putText(output_frame, f"Yaw: {yaw:.2f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Pitch: {pitch:.2f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Landmarks", output_frame)
    # ===================== END OF FACE PROCESSING ======================
    
    # ====================== EYE PROCESSING ======================
    # Process eyes BEFORE screen position so fresh eye data is available for combined gaze.
    eye_frame = frame.copy()
    eye_results = eye_processor.process_frame(eye_frame)
    eye_frame, raw_eye_data = eye_processor._draw_landmarks(eye_frame, eye_results)
    cv2.imshow("Eye Landmarks", eye_frame)
    # ===================== END OF EYE PROCESSING =====================

    # ===================== ESTIMATED SCREEN POSITION DEBUGGING =========
    if avg_direction is not None:
        sw, sh = heatmap_sw, heatmap_sh

        # Compute screen position: use combined head+eye after calibration, head-only before
        if eye_calibrator.calibration_stage == 5 and raw_eye_data is not None:
            screen_x, screen_y = axis_processor.get_estimated_screen_position(
                raw_eye_data, eye_calibrator.calibrated_thresholds
            )
            tracking_mode = "Head + Eye"
        else:
            screen_x, screen_y = axis_processor.get_estimated_screen_position()
            tracking_mode = "Head Only"

        # --- Heatmap accumulation ---
        if screen_x is not None and screen_y is not None:
            gx = max(0, min(sw - 1, screen_x // 2))
            gy = max(0, min(sh - 1, screen_y // 2))
            cv2.circle(heatmap_accumulator, (gx, gy), 18, HEATMAP_INTENSITY, -1)
        heatmap_accumulator *= HEATMAP_DECAY

        # --- Render heatmap ---
        heatmap_blurred = cv2.GaussianBlur(heatmap_accumulator, (31, 31), 0)
        max_val = heatmap_blurred.max()
        if max_val > 0.001:
            heatmap_norm = np.clip(heatmap_blurred / max_val * 255, 0, 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            # Use intensity as alpha to avoid blue-tint on empty areas
            alpha = (heatmap_norm.astype(np.float32) / 255.0)
            alpha3 = np.stack([alpha, alpha, alpha], axis=-1)
            screen_frame = (heatmap_colored.astype(np.float32) * alpha3 * 0.7).astype(np.uint8)
        else:
            screen_frame = np.zeros((sh, sw, 3), dtype=np.uint8)

        # Draw 3x3 grid overlay
        for i in range(1, 3):
            cv2.line(screen_frame, (sw * i // 3, 0), (sw * i // 3, sh), (50, 50, 50), 1)
            cv2.line(screen_frame, (0, sh * i // 3), (sw, sh * i // 3), (50, 50, 50), 1)

        # Draw center crosshair
        cx, cy = sw // 2, sh // 2
        cv2.line(screen_frame, (cx - 15, cy), (cx + 15, cy), (70, 70, 70), 1)
        cv2.line(screen_frame, (cx, cy - 15), (cx, cy + 15), (70, 70, 70), 1)

        if screen_x is not None and screen_y is not None:
            # Draw fading trail
            trail = axis_processor.trail_history
            for i, (tx, ty) in enumerate(trail):
                alpha_t = (i + 1) / len(trail)
                radius = max(2, int(4 * alpha_t))
                color = (0, int(120 * alpha_t), 0)
                cv2.circle(screen_frame, (tx // 2, ty // 2), radius, color, -1)

            # Draw main dot with glow
            cv2.circle(screen_frame, (screen_x // 2, screen_y // 2), 14, (0, 100, 0), -1)
            cv2.circle(screen_frame, (screen_x // 2, screen_y // 2), 10, (0, 255, 0), -1)
            cv2.circle(screen_frame, (screen_x // 2, screen_y // 2), 12, (0, 200, 0), 2)

        # Show tracking mode and yaw/pitch
        cv2.putText(screen_frame, tracking_mode, (sw - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
        if tracking_mode == "Head + Eye":
            # Show adaptive blend weights for debugging
            head_dev = math.sqrt(yaw**2 + pitch**2) if yaw is not None and pitch is not None else 0
            max_dev = math.sqrt(axis_processor.yawDegrees**2 + axis_processor.pitchDegrees**2)
            hf = min(head_dev / max_dev, 1.0)
            ew = 0.7 - 0.4 * hf
            cv2.putText(screen_frame, f"Head:{1-ew:.0%} Eye:{ew:.0%}", (sw - 175, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 150, 150), 1)
        if yaw is not None and pitch is not None:
            cv2.putText(screen_frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}", (10, sh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Calibration data and gaze direction overlay
        if eye_calibrator.calibration_stage == 5:
            cv2.putText(screen_frame, f"Calib Center: ({eye_calibrator.calibration_iris_boxheight_center:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(screen_frame, f"Calib Up: {eye_calibrator.calibration_iris_boxheight_up:.4f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(screen_frame, f"Calib Down: {eye_calibrator.calibration_iris_boxheight_down:.4f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(screen_frame, f"Calib Left: {eye_calibrator.calibration_left:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(screen_frame, f"Calib Right: {eye_calibrator.calibration_right:.4f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            directionsval = gaze_processor.update_direction(raw_eye_data, eye_calibrator.calibrated_thresholds)
            
            if directionsval is not None:
                if directionsval[0] != "Center":
                    current_gaze = directionsval[0]
                elif directionsval[1] != "Center":
                    current_gaze = directionsval[1]
                else:
                    current_gaze = "Center"
                
                cv2.putText(screen_frame, f"Gaze Direction: ({directionsval[0]}({directionsval[2]}), {directionsval[1]}({directionsval[3]}))", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Direction-to-grid-cell mapping
                grid_map = {
                    ("Up", "Left"): (0, 0), ("Up", "Center"): (1, 0), ("Up", "Right"): (2, 0),
                    ("Center", "Left"): (0, 1), ("Center", "Center"): (1, 1), ("Center", "Right"): (2, 1),
                    ("Down", "Left"): (0, 2), ("Down", "Center"): (1, 2), ("Down", "Right"): (2, 2),
                }
                cell = grid_map.get((directionsval[0], directionsval[1]))
                if cell is not None:
                    col, row = cell
                    x1, y1 = sw * col // 3, sh * row // 3
                    x2, y2 = sw * (col + 1) // 3, sh * (row + 1) // 3
                    cv2.rectangle(screen_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Estimated Screen Position", screen_frame)
    # ===================== END OF ESTIMATED SCREEN POSITION DEBUGGING =========

    if eye_calibrator.calibration_stage == 5:
        scoring_processor.update(frame, current_gaze)

scoring_processor.cleanup()
cap.release()
cv2.destroyAllWindows()
