import cv2
import time
import threading
from collections import deque
from datetime import datetime

class SuspicionScoringProcessor:
    def __init__(self):
        # Feature 2: Optimized Video Evidence Capture
        self.frame_buffer = deque(maxlen=90)  # 3 seconds at 30 fps
        self.is_recording = False
        self.post_roll = []
        self.pre_roll_copy = []
        self.trigger_reason = ""
        
        # Feature 1: The State Machine (Academic Thresholds)
        self.current_direction = "Center"
        self.direction_start_time = time.time()
        
        # Rolling window for frequency tracking (60 seconds)
        self.shift_timestamps = deque()
        
    def update(self, frame, gaze_direction: str):
        # Pre-roll Buffer: append the incoming frame immediately
        self.frame_buffer.append(frame)
        
        # If currently recording post-roll, bypass the state machine
        if self.is_recording:
            # We copy the frame to avoid any external modification
            self.post_roll.append(frame.copy())
            
            # Post-roll Capture complete (90 frames = 3 seconds)
            if len(self.post_roll) >= 90:
                # Trigger asynchronous saving
                threading.Thread(
                    target=self._save_video_async,
                    args=(self.pre_roll_copy, self.post_roll, self.trigger_reason),
                    daemon=True
                ).start()
                
                # Reset recording state
                self.is_recording = False
                self.post_roll = []
                self.pre_roll_copy = []
                self.trigger_reason = ""
                
                # Reset state machine timer to prevent instant re-triggering
                self.direction_start_time = time.time()
            return

        current_time = time.time()

        # Track shifts from Center to any off-screen direction
        if gaze_direction != self.current_direction:
            if self.current_direction == "Center" and gaze_direction != "Center":
                self.shift_timestamps.append(current_time)
            
            self.current_direction = gaze_direction
            self.direction_start_time = current_time

        # Maintain 60-second rolling window for shifts
        while self.shift_timestamps and current_time - self.shift_timestamps[0] > 60.0:
            self.shift_timestamps.popleft()

        violation_reason = None

        # Duration (Stopwatch) Check
        duration = current_time - self.direction_start_time
        
        if gaze_direction in ["Left", "Right", "Up"] and duration > 3.0:
            violation_reason = f"{gaze_direction}_duration"
        elif gaze_direction == "Down" and duration > 5.0:
            violation_reason = f"{gaze_direction}_duration"
            
        # Frequency (Rolling Window) Check
        if not violation_reason and len(self.shift_timestamps) > 6:
            violation_reason = "high_frequency_shifts"

        # Trigger Execution
        if violation_reason:
            print(f"[SuspicionScoring] VIOLATION TRIGGERED: {violation_reason}")
            self.is_recording = True
            self.trigger_reason = violation_reason
            # Make a deep copy of the 90 frames in the deque
            self.pre_roll_copy = [f.copy() for f in self.frame_buffer]
            # Clear shift history so we don't spam triggers immediately after
            self.shift_timestamps.clear()

    def _save_video_async(self, pre_frames, post_frames, reason):
        if not pre_frames and not post_frames:
            return
            
        print(f"[SuspicionScoring] Async video saving started for reason: {reason}")
        all_frames = pre_frames + post_frames
        
        # Dynamically extract resolution from the first frame
        first_frame = all_frames[0]
        height, width = first_frame.shape[:2]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_violation_{reason}.mp4"
        
        # Use cv2.VideoWriter with mp4v codec at 30 FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        
        for f in all_frames:
            out.write(f)
            
        out.release()
        print(f"[SuspicionScoring] Video evidence saved successfully: {filename}")
