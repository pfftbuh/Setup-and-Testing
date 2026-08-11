import numpy as np

class GazeDirectionProcessor:
    def __init__(self, screen_width=1920, screen_height=1080):
        self.avg_direction = None
        self.screen_width = screen_width
        self.screen_height = screen_height

    def weighted_screen_position(self, screen_pos_face, screen_pos_eye, weight_face=0.30, weight_eye=0.70):
        if screen_pos_face is None and screen_pos_eye is None:
            return None, None
        elif screen_pos_face is None:
            blended = screen_pos_eye
        elif screen_pos_eye is None:
            blended = screen_pos_face
        else:
            weighted_x = int(screen_pos_face[0] * weight_face + screen_pos_eye[0] * weight_eye)
            weighted_y = int(screen_pos_face[1] * weight_face + screen_pos_eye[1] * weight_eye)
            blended = (weighted_x, weighted_y)

        label = self._label_position(blended[0], blended[1])
        return blended, label

    def _label_position(self, x, y):
        # Horizontal segmentation
        if x < self.screen_width / 3:
            col = "Left"
        elif x < 2 * self.screen_width / 3:
            col = "Center"
        else:
            col = "Right"

        # Vertical segmentation
        if y < self.screen_height / 3:
            row = "Up"
        elif y < 2 * self.screen_height / 3:
            row = "Center"
        else:
            row = "Down"

        return row, col

    def update_direction(self, raw_eye_data, calibrated_thresholds):
        if raw_eye_data is None or calibrated_thresholds is None:
            return None
        
        pupil_left = raw_eye_data['left']['pupil']
        pupil_right = raw_eye_data['right']['pupil']
        iris_box_left = raw_eye_data['left']['iris_boxheight']
        iris_box_right = raw_eye_data['right']['iris_boxheight']

        if pupil_left is not None and pupil_right is not None:
            avg_pupil_x = (pupil_left[0] + pupil_right[0]) / 2.0
            avg_pupil_y = (pupil_left[1] + pupil_right[1]) / 2.0
            avg_iris_boxheight = (iris_box_left + iris_box_right) / 2.0

            direction_v_offset = avg_iris_boxheight - calibrated_thresholds['iris_boxheight_center']
            direction_h_offset = avg_pupil_x - calibrated_thresholds['center']

            vertical_deadzone = 0.01
            horizontal_deadzone = 0.05

            if abs(direction_v_offset) <= vertical_deadzone:
                direction_vertical = "Center"
            elif direction_v_offset > 0:
                direction_vertical = "Up"
            else:
                direction_vertical = "Down"

            if abs(direction_h_offset) <= horizontal_deadzone:
                direction_horizontal = "Center"
            elif direction_h_offset > 0:
                direction_horizontal = "Left"   # mirrored camera
            else:
                direction_horizontal = "Right"

            self.avg_direction = (
                direction_vertical,
                direction_horizontal,
                abs(direction_v_offset),
                abs(direction_h_offset)
            )

            return self.avg_direction
