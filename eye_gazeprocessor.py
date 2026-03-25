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
        self.calibration_stage = 0
        self.calibration_data = {
            'left': deque(maxlen=60),
            'right': deque(maxlen=60),
            'top': deque(maxlen=60),
            'bottom': deque(maxlen=60),
            'center': deque(maxlen=60)
        }

        


