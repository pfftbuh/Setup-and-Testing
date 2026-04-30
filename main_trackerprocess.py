import cv2
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
        yaw, pitch = axis_processor.process(avg_direction)
        screen_x, screen_y = axis_processor.get_estimated_screen_position()
        if yaw is not None and pitch is not None:
            cv2.putText(output_frame, f"Yaw: {yaw:.2f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Pitch: {pitch:.2f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Landmarks", output_frame)
    # ===================== END OF FACE PROCESSING ======================
    
    # ===================== ESTIMATED SCREEN POSITION DEBUGGING =========
    # Frame is for debugging purposes, create a frame with the estimated screen position.
    if avg_direction is not None:
        screen_frame = np.zeros((1080//2, 1920//2, 3), dtype=np.uint8)
        if screen_x is not None and screen_y is not None:
            cv2.circle(screen_frame, (int(screen_x//2), int(screen_y//2)), 10, (0, 255, 0), -1)
        # For debugging purposes, display calibration thresholds values on the screen frame.
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
                
                cv2.putText(screen_frame, f"Gaze Direction: ({directionsval[0]}({directionsval[2]}), {directionsval[1]}({directionsval[3]}))", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if directionsval[0] == "Up" and directionsval[1] == "Left":
                   # Draw a rectangle in the top-left corner of the screen frame to indicate up-left gaze direction.
                   cv2.rectangle(screen_frame, (0, 0), (screen_frame.shape[1]//3, screen_frame.shape[0]//3), (255, 0, 0), 2) 
                elif directionsval[0] == "Up" and directionsval[1] == "Right":
                    # Draw a rectangle in the top-right corner of the screen frame to indicate up-right gaze direction.
                    cv2.rectangle(screen_frame, (screen_frame.shape[1]*2//3, 0), (screen_frame.shape[1], screen_frame.shape[0]//3), (255, 0, 0), 2)
                elif directionsval[0] == "Down" and directionsval[1] == "Left":
                    # Draw a rectangle in the bottom-left corner of the screen frame to indicate down-left gaze direction.
                    cv2.rectangle(screen_frame, (0, screen_frame.shape[0]*2//3), (screen_frame.shape[1]//3, screen_frame.shape[0]), (255, 0, 0), 2)
                elif directionsval[0] == "Down" and directionsval[1] == "Right":
                    # Draw a rectangle in the bottom-right corner of the screen frame to indicate down-right gaze direction.
                    cv2.rectangle(screen_frame, (screen_frame.shape[1]*2//3, screen_frame.shape[0]*2//3), (screen_frame.shape[1], screen_frame.shape[0]), (255, 0, 0), 2)
                elif directionsval[0] == "Center" and directionsval[1] == "Left":
                    # Draw a rectangle in the left-center of the screen frame to indicate center-left gaze direction.
                    cv2.rectangle(screen_frame, (0, screen_frame.shape[0]//3), (screen_frame.shape[1]//3, screen_frame.shape[0]*2//3), (255, 0, 0), 2)
                elif directionsval[0] == "Center" and directionsval[1] == "Right":
                    # Draw a rectangle in the right-center of the screen frame to indicate center-right gaze direction.
                    cv2.rectangle(screen_frame, (screen_frame.shape[1]*2//3, screen_frame.shape[0]//3), (screen_frame.shape[1], screen_frame.shape[0]*2//3), (255, 0, 0), 2)
                elif directionsval[0] == "Up" and directionsval[1] == "Center":
                    # Draw a rectangle in the top-center of the screen frame to indicate up-center gaze direction.
                    cv2.rectangle(screen_frame, (screen_frame.shape[1]//3, 0), (screen_frame.shape[1]*2//3, screen_frame.shape[0]//3), (255, 0, 0), 2)
                elif directionsval[0] == "Down" and directionsval[1] == "Center":
                    # Draw a rectangle in the bottom-center of the screen frame to indicate down-center gaze direction.
                    cv2.rectangle(screen_frame, (screen_frame.shape[1]//3, screen_frame.shape[0]*2//3), (screen_frame.shape[1]*2//3, screen_frame.shape[0]), (255, 0, 0), 2)
                elif directionsval[0] == "Center" and directionsval[1] == "Center":
                    # Draw a rectangle in the center of the screen frame to indicate center gaze direction.
                    cv2.rectangle(screen_frame, (screen_frame.shape[1]//3, screen_frame.shape[0]//3), (screen_frame.shape[1]*2//3, screen_frame.shape[0]*2//3), (255, 0, 0), 2)                    

        cv2.imshow("Estimated Screen Position", screen_frame)

    # ===================== END OF ESTIMATED SCREEN POSITION DEBUGGING =========
    
    # ====================== EYE PROCESSING ======================
    # Draw eye landmarks and place on a separate frame for debugging purposes.
    eye_frame = frame.copy()
    eye_results = eye_processor.process_frame(eye_frame)
    eye_frame, raw_eye_data = eye_processor._draw_landmarks(eye_frame, eye_results)
    cv2.imshow("Eye Landmarks", eye_frame)
    # ===================== END OF EYE PROCESSING =====================

    scoring_processor.update(frame, current_gaze)

scoring_processor.cleanup()
cap.release()
cv2.destroyAllWindows()
