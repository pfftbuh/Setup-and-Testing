import numpy as np
import time

class EyeScreenPosProcessor:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Smoothing state
        self.smoothed_pos = None
        self.last_t = None

        # Time constant (seconds): bigger = smoother/slower
        self.smoothing_tau = 0.18

        self.segments = {
            "UpLeft": ((0, 0), (screen_width//3, screen_height//3)),
            "UpCenter": ((screen_width//3, 0), (screen_width*2//3, screen_height//3)),
            "UpRight": ((screen_width*2//3, 0), (screen_width, screen_height//3)),
            "CenterLeft": ((0, screen_height//3), (screen_width//3, screen_height*2//3)),
            "CenterCenter": ((screen_width//3, screen_height//3), (screen_width*2//3, screen_height*2//3)),
            "CenterRight": ((screen_width*2//3, screen_height//3), (screen_width, screen_height*2//3)),
            "DownLeft": ((0, screen_height*2//3), (screen_width//3, screen_height)),
            "DownCenter": ((screen_width//3, screen_height*2//3), (screen_width*2//3, screen_height)),
            "DownRight": ((screen_width*2//3, screen_height*2//3), (screen_width, screen_height)),
        }

    def _smooth(self, target_xy):
        now = time.time()
        if self.smoothed_pos is None or self.last_t is None:
            self.smoothed_pos = np.array(target_xy, dtype=float)
            self.last_t = now
            return tuple(self.smoothed_pos)

        dt = max(1e-3, now - self.last_t)
        self.last_t = now

        # Frame-rate independent EMA: alpha in (0,1)
        alpha = 1.0 - np.exp(-dt / self.smoothing_tau)

        target = np.array(target_xy, dtype=float)
        self.smoothed_pos += alpha * (target - self.smoothed_pos)

        self.smoothed_pos[0] = np.clip(self.smoothed_pos[0], 0, self.screen_width - 1)
        self.smoothed_pos[1] = np.clip(self.smoothed_pos[1], 0, self.screen_height - 1)

        return tuple(self.smoothed_pos)

    def process(self, raw_eye_data, gaze_direction):
        # Convert normalized eye position to screen coordinates
        if raw_eye_data is None or gaze_direction is None:
            return None # or return last known position
        pupil_left = raw_eye_data['left']['pupil']
        pupil_right = raw_eye_data['right']['pupil']
        if pupil_left is None or pupil_right is None:
            return None # or return last known position
        
        avg_pupil_x = (pupil_left[0] + pupil_right[0]) / 2.0
        avg_pupil_y = (pupil_left[1] + pupil_right[1]) / 2.0

        u = np.clip(avg_pupil_x, 0.0, 1.0)
        v = np.clip(avg_pupil_y, 0.0, 1.0)

        
        if gaze_direction[0] == "Up" and gaze_direction[1] == "Left":
            # Map to up-left segment
            x1, y1 = self.segments["UpLeft"][0]
            x2, y2 = self.segments["UpLeft"][1]
            screen_x = u * (x2 - x1) + x1  
            screen_y = v * (y2 - y1) + y1
        elif gaze_direction[0] == "Up" and gaze_direction[1] == "Center":
            # Map to up-center segment
            x1, y1 = self.segments["UpCenter"][0]
            x2, y2 = self.segments["UpCenter"][1]
            screen_x = u * (x2 - x1) + x1
            screen_y = v * (y2 - y1) + y1
        elif gaze_direction[0] == "Up" and gaze_direction[1] == "Right":
            # Map to up-right segment
            x1, y1 = self.segments["UpRight"][0]
            x2, y2 = self.segments["UpRight"][1]
            screen_x = u * (x2 - x1) + x1
            screen_y = v * (y2 - y1) + y1
        elif gaze_direction[0] == "Center" and gaze_direction[1] == "Left":
            # Map to center-left segment
            x1, y1 = self.segments["CenterLeft"][0]
            x2, y2 = self.segments["CenterLeft"][1]
            screen_x = u * (x2 - x1) + x1
            screen_y = v * (y2 - y1) + y1
        elif gaze_direction[0] == "Center" and gaze_direction[1] == "Center":
            # Map to center-center segment
            x1, y1 = self.segments["CenterCenter"][0]
            x2, y2 = self.segments["CenterCenter"][1]
            screen_x = u * (x2 - x1) + x1
            screen_y = v * (y2 - y1) + y1
        elif gaze_direction[0] == "Center" and gaze_direction[1] == "Right":
            # Map to center-right segment
            x1, y1 = self.segments["CenterRight"][0]
            x2, y2 = self.segments["CenterRight"][1]
            screen_x = u * (x2 - x1) + x1
            screen_y = v * (y2 - y1) + y1
        elif gaze_direction[0] == "Down" and gaze_direction[1] == "Left":
            # Map to down-left segment
            x1, y1 = self.segments["DownLeft"][0]
            x2, y2 = self.segments["DownLeft"][1]
            screen_x = u * (x2 - x1) + x1
            screen_y = v * (y2 - y1) + y1
        elif gaze_direction[0] == "Down" and gaze_direction[1] == "Center":
            # Map to down-center segment
            x1, y1 = self.segments["DownCenter"][0]
            x2, y2 = self.segments["DownCenter"][1]
            screen_x = u * (x2 - x1) + x1
            screen_y = v * (y2 - y1) + y1
        elif gaze_direction[0] == "Down" and gaze_direction[1] == "Right":
            # Map to down-right segment
            x1, y1 = self.segments["DownRight"][0]
            x2, y2 = self.segments["DownRight"][1]
            screen_x = u * (x2 - x1) + x1
            screen_y = v * (y2 - y1) + y1
        else:
            return None 
        
        target = (screen_x, screen_y)
        smooth_x, smooth_y = self._smooth(target)
        return (smooth_x, smooth_y)