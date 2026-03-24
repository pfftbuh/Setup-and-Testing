import numpy as np # type: ignore
import math

class FaceDistanceProcessor:
    def __init__(self):
        self.y_dist = [240, 132, 350, 560, 200]
        self.cm_dist = [50, 90, 30, 20, 70]
        self.coeff = np.polyfit(self.y_dist, self.cm_dist, 2)
    
    def landmark_to_np(self, landmark, frame):
        h, w = frame.shape[:2]
        return np.array([landmark.x * w, landmark.y * h], dtype=np.int32)
    
    def estimate_distance(self, xy1, xy2):
        # Estimate the distance based on the distance between the left and right eye of the face.
        width_pts = math.sqrt((xy2[0] - xy1[0]) ** 2 + (xy2[1] - xy1[1]) ** 2)
        A, B, C = self.coeff
        distance = A * width_pts ** 2 + B * width_pts + C
        return distance