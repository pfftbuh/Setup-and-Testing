import cv2
import numpy as np
import face_trackprocessor as ftp
import face_axisprocessor as fap
import eye_trackprocessor as etp
import file_creationprocessor as fcp
import pyautogui

processor = ftp.FaceLandmarkerProcessor()
axis_processor = fap.FaceAxisProcessor()
eye_processor = etp.EyeLandmarkerProcessor()
file_creator = fcp.FileCreationProcessor("eye_data.json")

cap = cv2.VideoCapture(0)

point_counter = 0
eye_data_samples = 1000
eye_data_list = []  # List to store eye data samples

# For creating and training model for predicting screen position based on head pose and eye gaze data.

while True:
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
        # Normalize screen_x and screen_y to be between 0 and 1 based on the screen resolution (1920x1080).
        if screen_x is not None and screen_y is not None:
            screen_x_normalized = screen_x / 1920
            screen_y_normalized = screen_y / 1080
            
    # Show processed face frame with landmarks and head pose estimation.
    cv2.imshow("Face Landmarks", output_frame)
    # ===================== END OF FACE PROCESSING ======================
    
    # ===================== ESTIMATED SCREEN POSITION DEBUGGING =========
    
    screen_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    if avg_direction is not None:
        if screen_x is not None and screen_y is not None:
            cv2.circle(screen_frame, (int(screen_x_normalized * 1280), int(screen_y_normalized * 720)), 10, (0, 255, 0), -1)
        cv2.imshow("Estimated Screen Position", screen_frame)
    
    # ===================== END OF ESTIMATED SCREEN POSITION DEBUGGING =========
    
    # ===================== EYE PROCESSING ============================
    eye_frame = frame.copy()
    eye_results = eye_processor.process_frame(eye_frame)
    eye_frame, raw_eye_data = eye_processor._draw_landmarks(eye_frame, eye_results)
    cv2.imshow("Eye Landmarks", eye_frame)
    # ===================== END OF EYE PROCESSING =====================
    
    # ===================== 9 POINT CALIBRATION =======================

    # use pyautogui to create window the size of the screen and display 9 points for calibration purposes.
    screen_width, screen_height = pyautogui.size()
    points_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Define the 9 calibration points (corners, edges, and center of the screen).
    for i in range(3):
        for j in range(3):
            point_x = int((j + 0.5) * screen_width / 3)
            point_y = int((i + 0.5) * screen_height / 3)
            cv2.circle(points_frame, (point_x, point_y), 20, (0, 255, 0), -1)
        
    cv2.imshow("Calibration Points", points_frame)
    # ===================== END OF 9 POINT CALIBRATION =================
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if avg_direction is not None:
            axis_processor.calibrate(avg_direction)
    if key == ord('t'):
        # Capture cursor position
        cursor_x, cursor_y = pyautogui.position()
        print(f"Cursor position: ({cursor_x}, {cursor_y})")
        # Normalize cursor position to be between 0 and 1 based on the screen resolution (1920x1080).
        cursor_x_normalized = cursor_x / 1920
        cursor_y_normalized = cursor_y / 1080
        if avg_direction is not None and screen_x is not None and screen_y is not None:
            # Collect data point for training: (yaw, pitch, eye_gaze_data) -> (screen_x_normalized, screen_y_normalized)
            # Limited to 4 decmial places for easier model training and to reduce noise.
            yaw = round(yaw, 4)
            pitch = round(pitch, 4)
            eye_gaze_data = [
                round(raw_eye_data['left']['iris_boxheight'], 4),
                round(raw_eye_data['right']['iris_boxheight'], 4),
                round(raw_eye_data['left']['pupil'][0], 4),
                round(raw_eye_data['right']['pupil'][0], 4),
                round(raw_eye_data['left']['pupil'][1], 4),
                round(raw_eye_data['right']['pupil'][1], 4)
                ]
            data_point = {
                'yaw': yaw,
                'pitch': pitch,
                'eye_gaze': eye_gaze_data,
                'screen_x': screen_x_normalized,
                'screen_y': screen_y_normalized,
                'cursor_x': cursor_x_normalized,
                'cursor_y': cursor_y_normalized
            }
            print(f"Collected data point: {data_point}")
        if len(eye_data_list) < eye_data_samples:
            eye_data_list.append(data_point)
            if len(eye_data_list) % 10 == 0:
                print(f"Collected {len(eye_data_list)}/{eye_data_samples} eye data samples.")
        if len(eye_data_list) >= eye_data_samples:
            print("Collected enough eye data samples for training!")
        
        # Create a JSON file with the collected eye data samples for training purposes.
        file_creator.create_file(eye_data_list)

    if key == ord('q'):
        break
    