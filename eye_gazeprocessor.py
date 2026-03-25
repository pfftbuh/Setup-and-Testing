# Where calculations of eye gaze direction and screen position estimation are performed.
# This file processes the eye landmarks detected by MediaPipe 
# and calculates the estimated gaze direction and screen position based on the eye landmarks. 
# It will use the eye_trackprocessor to get the eye landmarks, 
# and then apply calculations to estimate where on the screen the user is looking.

# Calibration is performed with 5 stages: left edge, right edge, top edge, bottom edge, and center.

import cv2
import numpy as np
from collections import deque
import mediapipe as mp

class EyeGazeProcessor:
    def __init__(self):
        
        self.raw_eye_data = {
            'left': {
                'iris_boxgheight': None,
                'pupil': None
            },
            'right': {
                'iris_boxheight': None,
                'pupil': None
            }
        }
        
        # iris tracking indices
        self.left_iris_indices = [468, 469, 470, 471]
        self.right_iris_indices = [473, 474, 475, 476]
        self.pupil_indices = [473, 468]  # right and left iris center points

        # iris box indices
        self.left_iris_box_indices = [160, 153]
        self.right_iris_box_indices = [387, 380]
        
        self.pupil_points = {
                'left': None,
                'right': None
        }
    
    def _calculate_eyelid_height(self, iris_box):
        # Calculate the height of the eyelid based on the iris box landmarks.
        y1 = iris_box[0][1]  # top point
        y2 = iris_box[1][1]  # bottom point
        
        return abs(y2 - y1)
    
    # This calculates the position of the iris center relative to the eye bounding box,
    # which is used to estimate the gaze direction.
    def _calculate_iris_position(self, eye_data):
        
        left_iris_box = eye_data['left']
        right_iris_box = eye_data['right']
        pupil_left = eye_data['pupil_left']
        pupil_right = eye_data['pupil_right']
        
        left_eyelid_height = self._calculate_eyelid_height(left_iris_box)
        right_eyelid_height = self._calculate_eyelid_height(right_iris_box)
        
        # Calculate the position of the pupil relative to the eye bounding box for both eyes.
        if pupil_left is not None and left_iris_box is not None:
            left_iris_center_x = (left_iris_box[0][0] + left_iris_box[1][0]) / 2
            left_iris_center_y = (left_iris_box[0][1] + left_iris_box[1][1]) / 2
            left_pupil_offset_x = pupil_left[0] - left_iris_center_x
            left_pupil_offset_y = pupil_left[1] - left_iris_center_y
        else:
            left_pupil_offset_x, left_pupil_offset_y = None, None   
            
        if pupil_right is not None and right_iris_box is not None:
            right_iris_center_x = (right_iris_box[0][0] + right_iris_box[1][0]) / 2
            right_iris_center_y = (right_iris_box[0][1] + right_iris_box[1][1]) / 2
            right_pupil_offset_x = pupil_right[0] - right_iris_center_x
            right_pupil_offset_y = pupil_right[1] - right_iris_center_y
        else:
            right_pupil_offset_x, right_pupil_offset_y = None, None
        
        self.raw_eye_data['left']['iris_boxheight'] = left_eyelid_height
        self.raw_eye_data['right']['iris_boxheight'] = right_eyelid_height
        self.raw_eye_data['left']['pupil'] = (left_pupil_offset_x, left_pupil_offset_y)
        self.raw_eye_data['right']['pupil'] = (right_pupil_offset_x, right_pupil_offset_y)
        
        return self.raw_eye_data
            
                    
        

        


