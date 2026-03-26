import cv2 
import numpy as np # type: ignore
import mediapipe as mp # type: ignore
from collections import deque

class EyeCalibrationProcessor:
    def __init__(self):
        self.sample_count = 30

        self.calibration_x_up = 0.0
        self.calibration_x_down = 0.0
        self.calibration_x_center = 0.0

        self.calibration_y_left = 0.0
        self.calibration_y_right = 0.0
        self.calibration_y_center = 0.0

        self.calibration_stage = -1
        self.calibrated = False

        self.calibration_x_center_values = deque(maxlen=self.sample_count)
        self.calibration_y_center_values = deque(maxlen=self.sample_count)
        self.calibration_x_up_values = deque(maxlen=self.sample_count)
        self.calibration_x_down_values = deque(maxlen=self.sample_count)
        self.calibration_y_left_values = deque(maxlen=self.sample_count)
        self.calibration_y_right_values = deque(maxlen=self.sample_count)
  
    
    def calibrate(self, raw_eye_data_list):
        # raw_eye_data contains normalized pupil positions and iris box heights for both eyes
        # We can use this data to set the calibration offsets for gaze estimation.
        for raw_eye_data in raw_eye_data_list:

            if self.calibration_stage == 0:
                # Calibrate center position — one real sample per frame
                self.calibration_x_center_values.append(raw_eye_data['left']['pupil'][0])
                self.calibration_y_center_values.append(raw_eye_data['left']['pupil'][1])

            elif self.calibration_stage == 1:
                # Calibrate up position
                self.calibration_x_up_values.append(raw_eye_data['left']['pupil'][0])

            elif self.calibration_stage == 2:
                # Calibrate down position
                self.calibration_x_down_values.append(raw_eye_data['left']['pupil'][0])

            elif self.calibration_stage == 3:
                # Calibrate left position
                self.calibration_y_left_values.append(raw_eye_data['left']['pupil'][1])

            elif self.calibration_stage == 4:
                # Calibrate right position
                self.calibration_y_right_values.append(raw_eye_data['left']['pupil'][1])


        
    
    def next_stage(self):
        
        self.calibration_stage += 1

        if self.calibration_stage == 0:
            print("Starting eye calibration. Please look at the center and press 'e' to capture.")
        elif self.calibration_stage == 1:
            print("Please look up and press 'e' to capture.")
        elif self.calibration_stage == 2:
            print("Please look down and press 'e' to capture.")
        elif self.calibration_stage == 3:
            print("Please look left and press 'e' to capture.")
        elif self.calibration_stage == 4:
            print("Please look right and press 'e' to capture.")
        elif self.calibration_stage == 5:
            self.calibration_x_center = np.mean(self.calibration_x_center_values) if self.calibration_x_center_values else 0.0
            self.calibration_y_center = np.mean(self.calibration_y_center_values) if self.calibration_y_center_values else 0.0
            self.calibration_x_up = np.mean(self.calibration_x_up_values) if self.calibration_x_up_values else 0.0
            self.calibration_x_down = np.mean(self.calibration_x_down_values) if self.calibration_x_down_values else 0.0
            self.calibration_y_left = np.mean(self.calibration_y_left_values) if self.calibration_y_left_values else 0.0
            self.calibration_y_right = np.mean(self.calibration_y_right_values) if self.calibration_y_right_values else 0.0
            print("Calibration complete!")
            print(f"Center: ({self.calibration_x_center:.4f}, {self.calibration_y_center:.4f})")
            print(f"Up: ({self.calibration_x_up:.4f}), Down: ({self.calibration_x_down:.4f})")
            print(f"Left: ({self.calibration_y_left:.4f}), Right: ({self.calibration_y_right:.4f})")
            self.calibrated = True
        else:
            print("Calibration already complete. No more stages.")


