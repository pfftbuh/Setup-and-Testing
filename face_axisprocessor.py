import numpy as np
import math
from collections import deque

class FaceAxisProcessor:
    def __init__(self):
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0
        self._yaw_deg = 0.0
        self._pitch_deg = 0.0
        
        #specify degrees at which screen border will be reached
        self.yawDegrees = 25 # x degrees left or right
        self.pitchDegrees = 18 # x degrees up or down (wider range for comfort)

        # Smoothing: exponential moving average (separate for head-only vs combined)
        self._smooth_x = None
        self._smooth_y = None
        self._smoothing_alpha = 0.20  # tuned for responsiveness without jitter

        # Trail history for visualization
        self.trail_history = deque(maxlen=15)

        # Screen dimensions
        self.screen_w = 1920
        self.screen_h = 1080

        # Eye data temporal smoothing (reduces noise from frame-to-frame iris jitter)
        self._eye_h_buffer = deque(maxlen=5)
        self._eye_v_buffer = deque(maxlen=5)
    
    def process(self, avg_direction):
        if avg_direction is None:
            return None, None
        yaw_deg, pitch_deg = self._compute_raw_angles(avg_direction)
        yaw_deg += self.calibration_offset_yaw
        pitch_deg += self.calibration_offset_pitch
        self._yaw_deg = yaw_deg      # store calibrated here
        self._pitch_deg = pitch_deg
        return yaw_deg, pitch_deg
    
    def _compute_raw_angles(self, avg_direction):
        reference_axis = np.array([0, 0, 1])  # Camera facing forward

        # Project onto XZ plane for yaw
        yaw_proj = np.array([avg_direction[0], 0, avg_direction[2]])
        yaw_norm = np.linalg.norm(yaw_proj)
        if yaw_norm > 0:
            yaw_proj = yaw_proj / yaw_norm
            # Clip dot product to avoid NaN from float precision issues
            dot_val = max(min(np.dot(reference_axis, yaw_proj), 1.0), -1.0)
            yaw_deg = np.degrees(np.arccos(dot_val))
            # Assign sign based on horizontal direction
            if avg_direction[0] < 0:
                yaw_deg = -yaw_deg
        else:
            yaw_deg = 0.0

        # Project onto YZ plane for pitch
        pitch_proj = np.array([0, avg_direction[1], avg_direction[2]])
        pitch_norm = np.linalg.norm(pitch_proj)
        if pitch_norm > 0:
            pitch_proj = pitch_proj / pitch_norm
            dot_val = max(min(np.dot(reference_axis, pitch_proj), 1.0), -1.0)
            pitch_deg = np.degrees(np.arccos(dot_val))
            # Assign sign based on vertical direction
            if avg_direction[1] > 0:
                pitch_deg = -pitch_deg
        else:
            pitch_deg = 0.0

        return yaw_deg, pitch_deg

    def calibrate(self, avg_direction):
        """Set the current head pose as the zero reference."""
        raw_yaw, raw_pitch = self._compute_raw_angles(avg_direction)
        self.calibration_offset_yaw = -raw_yaw
        self.calibration_offset_pitch = -raw_pitch
        # Reset smoothing on recalibration
        self._smooth_x = None
        self._smooth_y = None
        self._eye_h_buffer.clear()
        self._eye_v_buffer.clear()
        self.trail_history.clear()

    def _get_head_screen_position(self):
        """Compute raw head-only screen position from yaw/pitch (no smoothing)."""
        head_x = ((self._yaw_deg + self.yawDegrees) / (2 * self.yawDegrees)) * self.screen_w
        head_y = ((self.pitchDegrees - self._pitch_deg) / (2 * self.pitchDegrees)) * self.screen_h
        return head_x, head_y

    def get_estimated_screen_position(self, raw_eye_data=None, calibrated_thresholds=None):
        """
        Compute smoothed screen position.
        If eye data and thresholds are provided, blends head pose with eye gaze.
        Otherwise uses head pose only.
        """
        head_x, head_y = self._get_head_screen_position()

        # If eye calibration data available, compute combined position
        if raw_eye_data is not None and calibrated_thresholds is not None:
            eye_h_norm, eye_v_norm = self._compute_eye_normalized(raw_eye_data, calibrated_thresholds)
            if eye_h_norm is not None:
                # Adaptive blending: when head is near center, trust eyes more
                # When head is turned far, trust head more (eyes have less range)
                head_deviation = math.sqrt(self._yaw_deg**2 + self._pitch_deg**2)
                max_deviation = math.sqrt(self.yawDegrees**2 + self.pitchDegrees**2)
                head_factor = min(head_deviation / max_deviation, 1.0)
                
                # Eye weight: 0.7 when centered, 0.3 when head is fully turned
                eye_weight = 0.7 - 0.4 * head_factor
                head_weight = 1.0 - eye_weight
                
                # Eye screen position: map normalized [-1,1] to screen coords
                eye_x = ((1 - eye_h_norm) / 2) * self.screen_w
                eye_y = ((1 - eye_v_norm) / 2) * self.screen_h
                
                raw_x = head_weight * head_x + eye_weight * eye_x
                raw_y = head_weight * head_y + eye_weight * eye_y
            else:
                raw_x, raw_y = head_x, head_y
        else:
            raw_x, raw_y = head_x, head_y

        # Apply exponential moving average smoothing
        if self._smooth_x is None:
            self._smooth_x = raw_x
            self._smooth_y = raw_y
        else:
            self._smooth_x += self._smoothing_alpha * (raw_x - self._smooth_x)
            self._smooth_y += self._smoothing_alpha * (raw_y - self._smooth_y)

        screen_x = max(0, min(self.screen_w - 1, int(self._smooth_x)))
        screen_y = max(0, min(self.screen_h - 1, int(self._smooth_y)))

        # Store for trail visualization
        self.trail_history.append((screen_x, screen_y))

        return screen_x, screen_y

    def _compute_eye_normalized(self, raw_eye_data, thresholds):
        """
        Compute temporally-smoothed, both-eye-averaged normalized gaze offset.
        Returns (h_norm, v_norm) in [-1, 1] range, or (None, None) on error.
        """
        try:
            # Average BOTH eyes for more stable tracking
            pupil_l_x = raw_eye_data['left']['pupil'][0]
            pupil_r_x = raw_eye_data['right']['pupil'][0]
            iris_bh_l = raw_eye_data['left']['iris_boxheight']
            iris_bh_r = raw_eye_data['right']['iris_boxheight']
        except (KeyError, TypeError):
            return None, None

        # Average both eyes
        avg_pupil_x = (pupil_l_x + pupil_r_x) / 2.0
        avg_iris_bh = (iris_bh_l + iris_bh_r) / 2.0

        x_center = thresholds['x_center']
        y_left = thresholds['y_left']
        y_right = thresholds['y_right']
        bh_center = thresholds['iris_boxheight_center']
        bh_up = thresholds['iris_boxheight_up']
        bh_down = thresholds['iris_boxheight_down']

        # Asymmetric normalization: use left range for left offsets, right range for right
        h_offset = avg_pupil_x - x_center
        if h_offset >= 0:
            h_range = abs(y_left - x_center) if abs(y_left - x_center) > 0.001 else 0.001
        else:
            h_range = abs(y_right - x_center) if abs(y_right - x_center) > 0.001 else 0.001
        h_norm = np.clip(h_offset / h_range, -1.0, 1.0)

        v_offset = avg_iris_bh - bh_center
        if v_offset >= 0:
            v_range = abs(bh_up - bh_center) if abs(bh_up - bh_center) > 0.001 else 0.001
        else:
            v_range = abs(bh_down - bh_center) if abs(bh_down - bh_center) > 0.001 else 0.001
        v_norm = np.clip(v_offset / v_range, -1.0, 1.0)

        # Temporal smoothing: buffer recent values and average
        self._eye_h_buffer.append(h_norm)
        self._eye_v_buffer.append(v_norm)

        smoothed_h = float(np.mean(self._eye_h_buffer))
        smoothed_v = float(np.mean(self._eye_v_buffer))

        return smoothed_h, smoothed_v
