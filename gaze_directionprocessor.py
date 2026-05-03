import numpy as np

class GazeDirectionProcessor:
    def __init__(self):
        self.avg_direction = None

    def weighted_screen_position(self, screen_pos_face, screen_pos_eye, weight_face=0.5, weight_eye=0.5):
        if screen_pos_face is None and screen_pos_eye is None:
            return None
        elif screen_pos_face is None:
            return screen_pos_eye
        elif screen_pos_eye is None:
            return screen_pos_face
        
        weighted_x = int(screen_pos_face[0] * weight_face + screen_pos_eye[0] * weight_eye)
        weighted_y = int(screen_pos_face[1] * weight_face + screen_pos_eye[1] * weight_eye)
        
        return (weighted_x, weighted_y)

    def update_direction(self, raw_eye_data, calibrated_thresholds):
        if raw_eye_data is None or calibrated_thresholds is None:
            return None
        
        # Calculate the average pupil position across both eyes
        pupil_left = raw_eye_data['left']['pupil']
        pupil_right = raw_eye_data['right']['pupil']
        iris_box_left = raw_eye_data['left']['iris_boxheight']
        iris_box_right = raw_eye_data['right']['iris_boxheight']
        calibrated_up = calibrated_thresholds['iris_boxheight_up']
        calibrated_down = calibrated_thresholds['iris_boxheight_down']

        if pupil_left is not None and pupil_right is not None:
            avg_pupil_x = (pupil_left[0] + pupil_right[0]) / 2.0
            avg_pupil_y = (pupil_left[1] + pupil_right[1]) / 2.0
            avg_iris_boxheight = (iris_box_left + iris_box_right) / 2.0
            
            if avg_pupil_x is not None and avg_pupil_y is not None:

                direction_v_offset = avg_iris_boxheight - calibrated_thresholds['iris_boxheight_center']
                direction_h_offset = avg_pupil_x - calibrated_thresholds['center']

                vertical_deadzone = 0.01
                horizontal_deadzone = 0.05

                if abs(direction_v_offset) <= vertical_deadzone:
                    direction_vertical = "Center"
                elif direction_v_offset > 0:
                    direction_vertical = "Up"
                elif direction_v_offset < 0:
                    direction_vertical = "Down"

                if abs(direction_h_offset) <= horizontal_deadzone:
                    direction_horizontal = "Center"
                elif direction_h_offset > 0:
                    direction_horizontal = "Left"   # mirrored camera
                else:
                    direction_horizontal = "Right"

                self.avg_direction = f"{direction_vertical}", f"{direction_horizontal}", f"{abs(direction_v_offset)}", f"{abs(direction_h_offset)}"

            
            return self.avg_direction