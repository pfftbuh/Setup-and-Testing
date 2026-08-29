import cv2 
import numpy as np # type: ignore
import mediapipe as mp # type: ignore
from collections import deque

class EyeCalibrationProcessor:
    def __init__(self):
        self.sample_count = 60

        self.calibration_up = 0.0
        self.calibration_down = 0.0
        self.calibration_h_center = 0.0

        self.calibration_left = 0.0
        self.calibration_right = 0.0
        self.calibration_v_center = 0.0

        self.calibration_iris_boxheight_center = 0.0
        self.calibration_iris_boxheight_up = 0.0
        self.calibration_iris_boxheight_down = 0.0

        self.calibration_stage = -1
        self.calibrated = False

        self.calibration_h_center_values = deque(maxlen=self.sample_count)
        self.calibration_v_center_values = deque(maxlen=self.sample_count)
        self.calibration_up_values = deque(maxlen=self.sample_count)
        self.calibration_down_values = deque(maxlen=self.sample_count)
        self.calibration_left_values = deque(maxlen=self.sample_count)
        self.calibration_right_values = deque(maxlen=self.sample_count)

        self.calibration_iris_boxheight_center_values = deque(maxlen=self.sample_count)
        self.calibration_iris_boxheight_up_values = deque(maxlen=self.sample_count)
        self.calibration_iris_boxheight_down_values = deque(maxlen=self.sample_count)

        self.calibrated_thresholds = {
            'up': self.calibration_up,
            'down': self.calibration_down,
            'center': self.calibration_h_center,
            'left': self.calibration_left,
            'right': self.calibration_right,
            'v_center': self.calibration_v_center,
            'iris_boxheight_center': self.calibration_iris_boxheight_center,
            'iris_boxheight_up': self.calibration_iris_boxheight_up,
            'iris_boxheight_down': self.calibration_iris_boxheight_down
        }
  
    
    def calibrate(self, raw_eye_data_list):
        # raw_eye_data contains normalized pupil positions and iris box heights for both eyes
        # We can use this data to set the calibration offsets for gaze estimation.
        for raw_eye_data in raw_eye_data_list:

            if self.calibration_stage == 0:
                # Calibrate center position — one real sample per frame
                avg_pupil_x = (raw_eye_data['left']['pupil'][0] + raw_eye_data['right']['pupil'][0]) / 2.0
                avg_pupil_y = (raw_eye_data['left']['pupil'][1] + raw_eye_data['right']['pupil'][1]) / 2.0
                self.calibration_h_center_values.append(avg_pupil_x)
                self.calibration_v_center_values.append(avg_pupil_y)
                self.calibration_iris_boxheight_center_values.append((raw_eye_data['left']['iris_boxheight'] + raw_eye_data['right']['iris_boxheight']) / 2.0)

            elif self.calibration_stage == 1:
                # Calibrate up position
                avg_pupil_y = (raw_eye_data['left']['pupil'][1] + raw_eye_data['right']['pupil'][1]) / 2.0
                self.calibration_up_values.append(avg_pupil_y)
                self.calibration_iris_boxheight_up_values.append((raw_eye_data['left']['iris_boxheight'] + raw_eye_data['right']['iris_boxheight']) / 2.0)

            elif self.calibration_stage == 2:
                # Calibrate down position
                avg_pupil_y = (raw_eye_data['left']['pupil'][1] + raw_eye_data['right']['pupil'][1]) / 2.0
                self.calibration_down_values.append(avg_pupil_y)
                self.calibration_iris_boxheight_down_values.append((raw_eye_data['left']['iris_boxheight'] + raw_eye_data['right']['iris_boxheight']) / 2.0)

            elif self.calibration_stage == 3:
                # Calibrate left position
                avg_pupil_x = (raw_eye_data['left']['pupil'][0] + raw_eye_data['right']['pupil'][0]) / 2.0
                self.calibration_left_values.append(avg_pupil_x)

            elif self.calibration_stage == 4:
                # Calibrate right position
                avg_pupil_x = (raw_eye_data['left']['pupil'][0] + raw_eye_data['right']['pupil'][0]) / 2.0
                self.calibration_right_values.append(avg_pupil_x)

    
    def next_stage(self):
        if self.calibration_stage < 5:
            self.calibration_stage += 1
        else:
            print("Calibration already complete. No more stages.")

        if self.calibration_stage == 0:
            print("Starting eye calibration. Please look at the center and press '{' to capture.")
        elif self.calibration_stage == 1:
            print("Please look up and press '{' to capture.")
        elif self.calibration_stage == 2:
            print("Please look down and press '{' to capture.")
        elif self.calibration_stage == 3:
            print("Please look left and press '{' to capture.")
        elif self.calibration_stage == 4:
            print("Please look right and press '{' to capture.")
        elif self.calibration_stage == 5:
            self.calibration_h_center = np.mean(self.calibration_h_center_values) if self.calibration_h_center_values else 0.0
            self.calibration_v_center = np.mean(self.calibration_v_center_values) if self.calibration_v_center_values else 0.0
            self.calibration_up = np.mean(self.calibration_up_values) if self.calibration_up_values else 0.0
            self.calibration_down = np.mean(self.calibration_down_values) if self.calibration_down_values else 0.0
            self.calibration_left = np.mean(self.calibration_left_values) if self.calibration_left_values else 0.0
            self.calibration_right = np.mean(self.calibration_right_values) if self.calibration_right_values else 0.0
            self.calibration_iris_boxheight_center = np.mean(self.calibration_iris_boxheight_center_values) if self.calibration_iris_boxheight_center_values else 0.0
            self.calibration_iris_boxheight_up = np.mean(self.calibration_iris_boxheight_up_values) if self.calibration_iris_boxheight_up_values else 0.0
            self.calibration_iris_boxheight_down = np.mean(self.calibration_iris_boxheight_down_values) if self.calibration_iris_boxheight_down_values else 0.0

            self.calibrated_thresholds = {
                'up': self.calibration_up,
                'down': self.calibration_down,
                'center': self.calibration_h_center,
                'left': self.calibration_left,
                'right': self.calibration_right,
                'v_center': self.calibration_v_center,
                'iris_boxheight_center': self.calibration_iris_boxheight_center,
                'iris_boxheight_up': self.calibration_iris_boxheight_up,
                'iris_boxheight_down': self.calibration_iris_boxheight_down
            }
            
            print("Calibration complete!")
            print(f"Center: ({self.calibration_h_center:.4f}, {self.calibration_v_center:.4f})")
            print(f"Up: ({self.calibration_up:.4f}), Down: ({self.calibration_down:.4f})")
            print(f"Left: ({self.calibration_left:.4f}), Right: ({self.calibration_right:.4f})")
            print(f"Iris Boxheight Center: ({self.calibration_iris_boxheight_center:.4f}), Up: ({self.calibration_iris_boxheight_up:.4f}), Down: ({self.calibration_iris_boxheight_down:.4f})")
            self.calibrated = True
        elif self.calibration_stage > 5:
            print("Calibration already complete. No more stages. Press '}' to end session.")


