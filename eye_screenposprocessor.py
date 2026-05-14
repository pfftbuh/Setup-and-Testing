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
            "Up Left": ((0, 0), (screen_width//3, screen_height//3)),
            "Up Center": ((screen_width//3, 0), (screen_width*2//3, screen_height//3)),
            "Up Right": ((screen_width*2//3, 0), (screen_width, screen_height//3)),
            "Center Left": ((0, screen_height//3), (screen_width//3, screen_height*2//3)),
            "Center Center": ((screen_width//3, screen_height//3), (screen_width*2//3, screen_height*2//3)),
            "Center Right": ((screen_width*2//3, screen_height//3), (screen_width, screen_height*2//3)),
            "Down Left": ((0, screen_height*2//3), (screen_width//3, screen_height)),
            "Down Center": ((screen_width//3, screen_height*2//3), (screen_width*2//3, screen_height)),
            "Down Right": ((screen_width*2//3, screen_height*2//3), (screen_width, screen_height)),
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

    def process(self, calibrated_thresholds, raw_eye_data):
        # Convert normalized eye position to screen coordinates
        if raw_eye_data is None:
            return None # or return last known position
        pupil_left = raw_eye_data['left']['pupil']
        pupil_right = raw_eye_data['right']['pupil']
        if pupil_left is None or pupil_right is None:
            return None # or return last known position
        
        avg_pupil_x = (pupil_left[0] + pupil_right[0]) / 2.0
        avg_pupil_y = (pupil_left[1] + pupil_right[1]) / 2.0
        avg_iris_boxheight = (raw_eye_data['left']['iris_boxheight'] + raw_eye_data['right']['iris_boxheight']) / 2.0

        # Map estimated screen position using eyelid height to determine vertical segment, and pupil position for horizontal segment.
        h_up = calibrated_thresholds['iris_boxheight_up']
        h_down = calibrated_thresholds['iris_boxheight_down']
        h_center = calibrated_thresholds['iris_boxheight_center']

        h_range = h_up - h_down

        if h_range > 0:
            # Normalize vertical position based on eyelid height relative to calibrated up/down thresholds.
            v_norm = (avg_iris_boxheight - h_down) / h_range
            v_norm = np.clip(v_norm, 0.0, 1.0)
        else:
            v_norm = 0.5  # default to center if no valid range
        
        w_left = calibrated_thresholds['left']
        w_right = calibrated_thresholds['right']
        w_range = w_left - w_right

        if w_range > 0:
            # Normalize horizontal position based on pupil offset relative to calibrated left/right thresholds.
            h_norm = (avg_pupil_x - w_right) / w_range
            h_norm = np.clip(h_norm, 0.0, 1.0)
        else:
            h_norm = 0.5  # default to center if no valid range
        
        # For simplicity, we can directly map the normalized (h_norm, v_norm) to screen coordinates.
        screen_x = int((1.0 - h_norm) * (self.screen_width - 1))
        screen_y = int((1.0 - v_norm) * (self.screen_height - 1))

        target = (screen_x, screen_y)
        smooth_x, smooth_y = self._smooth(target)
        # If None is returned, it means we don't have valid eye data to estimate position, so we can choose to return the last known smoothed position.
        if smooth_x is None or smooth_y is None:
            return self.smoothed_pos if self.smoothed_pos is not None else (self.screen_width // 2, self.screen_height // 2)
        
        # Determine which segment the smoothed position falls into
        direction = None
        for segment_name, (top_left, bottom_right) in self.segments.items():
            if top_left[0] <= smooth_x < bottom_right[0] and top_left[1] <= smooth_y < bottom_right[1]:
                direction = segment_name.split(" ")
                break
            
        if direction is None:
            direction = ("Center", "Center")

        return ((smooth_x, smooth_y), direction)