import numpy as np
import math

class FaceAxisProcessor:
    def __init__(self):
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0
        self._yaw_deg = 0.0
        self._pitch_deg = 0.0

        # anchor offset in pixels applied to the final screen position estimate
        self._anchor_offset_x = 0
        self._anchor_offset_y = 0
        
        #specify degrees at which screen border will be reached
        self.yawDegrees = 25 # x degrees left or right
        self.pitchDegrees = 12 # x degrees up or down
    
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
        reference_forward = np.array([0, 0, -1])

        # Horizontal (yaw) angle from reference (project onto XZ plane)
        xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
        xz_proj /= np.linalg.norm(xz_proj)
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad  # left is negative

        # Vertical (pitch) angle from reference (project onto YZ plane)
        yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
        yz_proj /= np.linalg.norm(yz_proj)
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad  # up is positive

        # Convert to degrees and re-center around 0
        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)

        #this results in the center being 180, +10 left = -170, +10 right = +170

        #convert left rotations to 0-180
        if yaw_deg < 0:
            yaw_deg = abs(yaw_deg)
        elif yaw_deg < 180:
            yaw_deg = 360 - yaw_deg

        if pitch_deg < 0:
            pitch_deg = 360 + pitch_deg

        self._yaw_deg = yaw_deg
        self._pitch_deg = pitch_deg

        return yaw_deg, pitch_deg

    def calibrate(self, avg_direction):
        """Set the current head pose as the zero reference."""
        raw_yaw, raw_pitch = self._compute_raw_angles(avg_direction)
        self.calibration_offset_yaw = -raw_yaw
        self.calibration_offset_pitch = -raw_pitch

    def update_anchor_from_face_position(self, face_center_x, face_center_y, frame_width, frame_height):
        """
        Shift the screen position anchor based on where the face sits in the camera frame.
        A face centered at (0.5, 0.5) produces no offset. Faces off-center push the
        estimated gaze position in the same direction proportionally to the screen size.
        
        Args:
            face_center_x: pixel x of the face center in the camera frame
            face_center_y: pixel y of the face center in the camera frame
            frame_width:   width of the camera frame in pixels
            frame_height:  height of the camera frame in pixels
        """
        screen_w = 1920
        screen_h = 1080

        # Normalise face position to [-0.5, 0.5] relative to frame centre
        norm_x = (face_center_x / frame_width) - 0.5
        norm_y = (face_center_y / frame_height) - 0.5

        # Map deviation directly onto screen space
        self._anchor_offset_x = int(norm_x * screen_w)
        self._anchor_offset_y = int(norm_y * screen_h)

    def get_estimated_screen_position(self):
        screen_w = 1920
        screen_h = 1080

        # yaw in [-yawDegrees, +yawDegrees] maps to [0, screen_w]
        screen_x = int(((self._yaw_deg + self.yawDegrees) / (2 * self.yawDegrees)) * screen_w)

        # pitch in [-pitchDegrees, +pitchDegrees] maps to [screen_h, 0]  (up=positive=top of screen)
        screen_y = int(((self.pitchDegrees - self._pitch_deg) / (2 * self.pitchDegrees)) * screen_h)

        # Apply anchor offset from face position in camera frame
        screen_x += self._anchor_offset_x
        screen_y += self._anchor_offset_y

        screen_x = max(0, min(screen_w - 1, screen_x))
        screen_y = max(0, min(screen_h - 1, screen_y))
        return screen_x, screen_y



