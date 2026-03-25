# This is where the raw eye gaze data is processed. 
# It calculates the position of the pupil relative to the eye bounding box, 
# which is used to estimate the gaze direction. 
# This module returns a list of iris height and pupil position data for both eyes, 
# which is then used by the eye_gazetoscreenposprocessor.

import cv2
import numpy as np
from collections import deque
import mediapipe as mp

class EyeGazeProcessor:
    def __init__(self):
        self.iris_data = {
            'left_eye_boxheight': 0.0,
            'right_eye_boxheight': 0.0,
            'left_iris_position': (0.0, 0.0),
            'right_iris_position': (0.0, 0.0)
        }
        
        # iris tracking indices
        self.left_iris_indices = [468, 469, 470, 471]
        self.right_iris_indices = [473, 474, 475, 476]
        self.pupil_indices = [473, 468]  # right and left iris center points

        # iris box indices
        self.left_iris_box_indices = [160, 153]
        self.right_iris_box_indices = [387, 380]
        
        self.pupil_points = {
                'left': (0,0),
                'right': (0,0)
        }
    
    def _calculate_eyelid_height(self, iris_box):
        # Calculate the height of the eyelid based on the iris box landmarks.
        y1 = iris_box[0][1]  # top point
        y2 = iris_box[1][1]  # bottom point
        
        return abs(y2 - y1)
    
    # This calculates the position of the iris center relative to the eye bounding box,
    # which is used to estimate the gaze direction.
    def _calculate_iris_position(self, main_face_landmarks):
        
        left_iris_box = []
        right_iris_box = []
        for face_landmarks in main_face_landmarks:
            
            # Calculate pupil position.
            for idx in self.pupil_indices:
                landmark = face_landmarks[idx]
                if idx == self.pupil_indices[0]:  # right eye
                    self.pupil_points['right'] = (landmark.x, landmark.y)
                else:  # left eye
                    self.pupil_points['left'] = (landmark.x, landmark.y)
            
            # Calculate iris box height for left eye.
            for idx in self.left_iris_box_indices:
                landmark = face_landmarks[idx]
                left_iris_box.append((landmark.x, landmark.y))
            
            # Calculate iris box height for right eye.
            for idx in self.right_iris_box_indices:
                landmark = face_landmarks[idx]
                right_iris_box.append((landmark.x, landmark.y))
                
            left_eyelid_height = self._calculate_eyelid_height(left_iris_box)
            left_iris_box_left = left_iris_box[0][0]
            left_iris_box_right = left_iris_box[1][0]
            
            right_eyelid_height = self._calculate_eyelid_height(right_iris_box)
            right_iris_box_left = right_iris_box[0][0]
            right_iris_box_right = right_iris_box[1][0]
            
        # Calculate the position of the left pupil relative to the left eye bounding box.
        if self.pupil_points['left'] is not None:
            left_pupil_x = self.pupil_points['left'][0]
            left_pupil_y = self.pupil_points['left'][1]
            left_iris_position_x = (left_pupil_x - left_iris_box_left) / (left_iris_box_right - left_iris_box_left)
            left_iris_position_y = (left_pupil_y - left_iris_box[0][1]) / left_eyelid_height
        else:
            left_iris_position_x = None
            left_iris_position_y = None

        # Calculate the position of the right pupil relative to the right eye bounding box.
        if self.pupil_points['right'] is not None:
            right_pupil_x = self.pupil_points['right'][0]
            right_pupil_y = self.pupil_points['right'][1]
            right_iris_position_x = (right_pupil_x - right_iris_box_left) / (right_iris_box_right - right_iris_box_left)
            right_iris_position_y = (right_pupil_y - right_iris_box[0][1]) / right_eyelid_height
        else:
            right_iris_position_x = None    
            right_iris_position_y = None
            
        self.iris_data['left_eye_boxheight'] = left_eyelid_height
        self.iris_data['right_eye_boxheight'] = right_eyelid_height
        self.iris_data['left_iris_position'] = (left_iris_position_x, left_iris_position_y)
        self.iris_data['right_iris_position'] = (right_iris_position_x, right_iris_position_y)
        
        return self.iris_data
            
        

            
                    
        

        


