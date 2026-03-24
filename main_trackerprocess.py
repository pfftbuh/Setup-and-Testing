import cv2
import numpy as np
import face_trackprocessor as ftp
import face_axisprocessor as fap


processor = ftp.FaceLandmarkerProcessor()
axis_processor = fap.FaceAxisProcessor()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = processor.process_frame(frame)
    output_frame, avg_direction = processor._draw_landmarks(frame, results)
    if avg_direction is not None:
        yaw, pitch = axis_processor.process(avg_direction)
        screen_x, screen_y = axis_processor.get_estimated_screen_position()
        if yaw is not None and pitch is not None:
            cv2.putText(output_frame, f"Yaw: {yaw:.2f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Pitch: {pitch:.2f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Landmarks", output_frame)

    # Frame is for debugging purposes, create a frame with the estimated screen position.
    if avg_direction is not None:
        screen_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        if screen_x is not None and screen_y is not None:
            cv2.circle(screen_frame, (int(screen_x), int(screen_y)), 10, (0, 255, 0), -1)
        cv2.imshow("Estimated Screen Position", screen_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and avg_direction is not None:
        axis_processor.calibrate(avg_direction)
        print("Calibrated! Current pose set as zero.")

cap.release()
cv2.destroyAllWindows()
