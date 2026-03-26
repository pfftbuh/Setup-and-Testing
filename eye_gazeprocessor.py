# eye_gazeprocessor.py
# This file contains the EyeGazeProcessor class, which processes eye landmarks to estimate gaze direction
# based on the position of the pupil relative to the eye bounding box. 
# It also calculates the height of the eyelids based on the iris box landmarks, 
# which can be used for further gaze estimation or blink detection. 
# The processor is designed to work with the EyeLandmarkerProcessor and can be calibrated for individual users to improve accuracy.
import cv2
import numpy as np
from collections import deque
import mediapipe as mp

class EyeGazeProcessor:
    def __init__(self):
    
        self.raw_eye_data = {
            'left': {
                'iris_boxheight': None,
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
        
        # Inter-pupil distance scales with camera distance just like iris_boxheight
        inter_pupil_dist = abs(pupil_right[0] - pupil_left[0])

        left_eyelid_height = self._calculate_eyelid_height(left_iris_box)
        right_eyelid_height = self._calculate_eyelid_height(right_iris_box)

        # Normalize: divide by inter-pupil distance to make scale-invariant
        if inter_pupil_dist > 0:
            left_eyelid_height = left_eyelid_height / inter_pupil_dist
            right_eyelid_height = right_eyelid_height / inter_pupil_dist
        
        # Calculate the normalized position of the pupil within the eye bounding box for gaze estimation.
        left_x_span = left_iris_box[1][0] - left_iris_box[0][0]
        left_y_span = left_iris_box[1][1] - left_iris_box[0][1]
        right_x_span = right_iris_box[1][0] - right_iris_box[0][0]
        right_y_span = right_iris_box[1][1] - right_iris_box[0][1]

        if 0 in (left_x_span, left_y_span, right_x_span, right_y_span):
            return self.raw_eye_data  # return last valid data

        left_pupil_offset_x = (pupil_left[0] - left_iris_box[0][0]) / (left_x_span)
        left_pupil_offset_y = (pupil_left[1] - left_iris_box[0][1]) / (left_y_span)
        right_pupil_offset_x = (pupil_right[0] - right_iris_box[0][0]) / (right_x_span)
        right_pupil_offset_y = (pupil_right[1] - right_iris_box[0][1]) / (right_y_span)

        self.raw_eye_data['left']['iris_boxheight'] = left_eyelid_height
        self.raw_eye_data['right']['iris_boxheight'] = right_eyelid_height
        self.raw_eye_data['left']['pupil'] = (left_pupil_offset_x, left_pupil_offset_y)
        self.raw_eye_data['right']['pupil'] = (right_pupil_offset_x, right_pupil_offset_y)
        
        return self.raw_eye_data







