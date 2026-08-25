import cv2
import numpy as np
import os
from datetime import datetime

class HeatmapProcessor:
    def __init__(self, screen_width=1920, screen_height=1080, output_dir=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        self.gaze_points = []
        
    def add_point(self, screen_pos):
        """Add a screen position (x, y) tuple to the collection."""
        if screen_pos is not None and isinstance(screen_pos, (tuple, list)) and len(screen_pos) >= 2:
            x, y = int(screen_pos[0]), int(screen_pos[1])
            # Clamp to screen bounds
            x = max(0, min(self.screen_width - 1, x))
            y = max(0, min(self.screen_height - 1, y))
            self.gaze_points.append((x, y))
            
    def generate_heatmap(self, output_filename=None):
        """Generates and saves the heatmap image from the collected points."""
        if not self.gaze_points:
            print("[HeatmapProcessor] No gaze points collected. Skipping heatmap generation.")
            return None
            
        print(f"[HeatmapProcessor] Generating heatmap from {len(self.gaze_points)} points...")
        
        # 1. Create a blank single-channel float32 image
        accumulator = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
        
        # 2. Accumulate points — draw a filled circle per point for wider base coverage
        point_radius = 5  # pixels; increase this for a larger initial coverage area
        for x, y in self.gaze_points:
            cv2.circle(accumulator, (x, y), point_radius, 1.0, thickness=-1)
            
        # 3. Apply a large Gaussian blur to smooth the points into a heatmap cloud
        # Increase kernel size (must be odd) for a wider, softer spread
        blur_kernel = 271  # increase this value for more spread // Originally 301 as of 2024-06-05
        if blur_kernel % 2 == 0:
            blur_kernel += 1  # Ensure kernel size is odd
        blurred = cv2.GaussianBlur(accumulator, (blur_kernel, blur_kernel), 0)
        
        # 4. Normalize to 0-255 range for color mapping
        max_val = np.max(blurred)
        if max_val > 0:
            normalized = (blurred / max_val) * 255
        else:
            normalized = blurred
            
        normalized = np.uint8(normalized)
        
        # 5. Apply Colormap (JET is standard for thermal/heat visualizations)
        heatmap_img = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Optional: Make zero-value areas black instead of the dark blue of JET colormap
        # In JET, 0 maps to (128, 0, 0) which is dark blue in BGR. We can mask it out.
        mask = (normalized == 0)
        heatmap_img[mask] = [0, 0, 0] # Set true 0 areas to pure black
        
        # 6. Save the image
        if output_filename is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(self.output_dir, f"session_heatmap_{timestamp_str}.png")
            
        cv2.imwrite(output_filename, heatmap_img)
        abs_path = os.path.abspath(output_filename)
        print(f"[HeatmapProcessor] Heatmap saved successfully to: {abs_path}")
        
        return abs_path
