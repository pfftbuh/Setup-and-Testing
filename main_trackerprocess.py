import cv2
import numpy as np
import face_trackprocessor as ftp
import face_axisprocessor as fap
import eye_trackprocessor as etp


processor = ftp.FaceLandmarkerProcessor()
axis_processor = fap.FaceAxisProcessor()
eye_processor = etp.EyeLandmarkerProcessor()
cap = cv2.VideoCapture(0)

# DEBUGGING PURPOSES: This loop processes the video feed from the webcam, detects face and eye landmarks, 
# estimates head pose, and displays the results in real-time. 
# It also allows for calibration of the head pose estimation by pressing the 'c' key. 
# The estimated screen position is visualized on a separate frame for debugging purposes. 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)  # Mirror the frame for a more natural webcam experience

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

    # Draw eye landmarks and place on a separate frame for debugging purposes.
    eye_frame = frame.copy()
    eye_results = eye_processor.process_frame(eye_frame)
    eye_frame, _ = eye_processor._draw_landmarks(eye_frame, eye_results)
    cv2.imshow("Eye Landmarks", eye_frame)


    # Frame is for debugging purposes, create a frame with the estimated screen position.
    if avg_direction is not None:
        screen_frame = np.zeros((1080//2, 1920//2, 3), dtype=np.uint8)
        if screen_x is not None and screen_y is not None:
            cv2.circle(screen_frame, (int(screen_x//2), int(screen_y//2)), 10, (0, 255, 0), -1)
        cv2.imshow("Estimated Screen Position", screen_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and avg_direction is not None:
        axis_processor.calibrate(avg_direction)
        print("Calibrated! Current pose set as zero.")

cap.release()
cv2.destroyAllWindows()
