import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time

class EyeLandmarkerProcessor:
    def __init__(self, model_path: str = "face_landmarker.task"):
        # Initialize MediaPipe Face Mesh or any other face landmark detection model here.
        self.model_path = model_path

        BaseOptions = python.BaseOptions
        VisionRunningMode = vision.RunningMode
        EyeLandmarker = vision.FaceLandmarker
        EyeLandmarkerOptions = vision.FaceLandmarkerOptions

        options = EyeLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )

        self.eye_landmarker = EyeLandmarker.create_from_options(options)

        # left and right eyelid landmark indices
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 130]
        self.right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 359]

        # iris tracking indices
        self.left_iris_indices = [468, 469, 470, 471]
        self.right_iris_indices = [473, 474, 475, 476]
        self.pupil_indices = [473, 468]  # right and left iris center points

        # iris box indices
        self.left_iris_box_indices = [160, 153]
        self.right_iris_box_indices = [387, 380]

    # Calculate bounding box around both eyes
    def get_eye_bbox(self, all_eye_points, w, h, padding=5):
        all_eye_points = np.array(all_eye_points)
        x_min = max(0, int(all_eye_points[:, 0].min()) - padding - 20)
        x_max = min(w, int(all_eye_points[:, 0].max()) + padding + 20)
        y_min = max(0, int(all_eye_points[:, 1].min()) - padding - 10)
        y_max = min(h, int(all_eye_points[:, 1].max()) + padding + 10)
        
        return x_min, x_max, y_min, y_max

    def process_frame(self, frame):
        # Process the frame to detect eye landmarks and return the results.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        try:
            results = self.eye_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            return None
        return results
    
    def _draw_landmarks(self, frame, results):

        all_eye_points = []

        if not results.face_landmarks:
            cv2.putText(frame, "No face detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, None
        
        elif results.face_landmarks:

            h, w = frame.shape[:2]
            for face_landmarks in results.face_landmarks:

                # Collect left eye landmarks for bounding box calculation
                for idx in self.left_eye_indices:
                    landmark = face_landmarks[idx]
                    all_eye_points.append((landmark.x * w, landmark.y * h))

                # Collect right eye landmarks for bounding box calculation
                for idx in self.right_eye_indices:
                    landmark = face_landmarks[idx]
                    all_eye_points.append((landmark.x * w, landmark.y * h))
                
                # Draw left iris landmarks
                for idx in self.left_iris_indices:
                    landmark = face_landmarks[idx]
                    cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)
                
                # Draw right iris landmarks
                for idx in self.right_iris_indices:
                    landmark = face_landmarks[idx]
                    cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 1, (0, 255, 0), -1)
                
                # Draw pupil landmarks
                for idx in self.pupil_indices:
                    landmark = face_landmarks[idx]
                    cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 3, (0, 0, 255), -1)
                
                # Draw circles on left iris box indices
                left_iris_box = []
                for idx in self.left_iris_box_indices:
                    landmark = face_landmarks[idx]
                    left_iris_box.append((landmark.x * w, landmark.y * h))
                    cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 1, (255, 0, 0), -1)
                
                # Draw bounding box on left iris box indices
                cv2.rectangle(frame, (int(left_iris_box[0][0]), int(left_iris_box[0][1])), 
                            (int(left_iris_box[1][0]), int(left_iris_box[1][1])), (255, 0, 0), 1)
                
                mid_left_iris_y = (left_iris_box[0][1] + left_iris_box[1][1]) / 2
                # Draw a line across the middle of the left iris box for debugging purposes
                cv2.line(frame, (int(left_iris_box[0][0]), int(mid_left_iris_y)), (int(left_iris_box[1][0]), int(mid_left_iris_y)), (255, 0, 0), 1)

                # Draw circles on right iris box indices
                right_iris_box = []
                for idx in self.right_iris_box_indices:
                    landmark = face_landmarks[idx]
                    right_iris_box.append((landmark.x * w, landmark.y * h))
                    cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 1, (255, 0, 0), -1)
                
                # Draw bounding box on right iris box indices
                cv2.rectangle(frame, (int(right_iris_box[0][0]), int(right_iris_box[0][1])), 
                            (int(right_iris_box[1][0]), int(right_iris_box[1][1])), (255, 0, 0), 1)    
                
                mid_right_iris_y = (right_iris_box[0][1] + right_iris_box[1][1]) / 2
                # Draw a line across the middle of the right iris box for debugging purposes
                cv2.line(frame, (int(right_iris_box[0][0]), int(mid_right_iris_y)), (int(right_iris_box[1][0]), int(mid_right_iris_y)), (255, 0, 0), 1)


            x_min, x_max, y_min, y_max = self.get_eye_bbox(all_eye_points, w, h)

            eye_frame = frame[y_min:y_max, x_min:x_max].copy()
            # Resize the eye frame to a fixed size for better visualization
            eye_frame = cv2.resize(eye_frame, (450, 200))

            return eye_frame, None