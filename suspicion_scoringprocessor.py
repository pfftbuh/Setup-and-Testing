import cv2
import time
import threading
import csv
import collections
from datetime import datetime

class SuspicionScoringProcessor:
    def __init__(self):
        # Feature 1: Event-Based CSV Logging Initialization
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"session_log_{self.session_timestamp}.csv"
        self._init_csv()
        
        # State Tracking for CSV
        self.current_state_direction = "Center"
        self.state_start_time = time.time()
        self.current_violation_label = "normal"
        self.current_score = 0
        self.current_video_file = ""
        
        # Feature 2: State Machine
        self.shift_timestamps = collections.deque()
        
        # Feature 3: Optimized Video Capture
        self.frame_buffer = collections.deque(maxlen=90)  # 3 seconds at 30 fps
        self.is_recording = False
        self.post_roll = []
        self.pre_roll_copy = []
        self.recording_reason = ""
        self.recording_filename = ""
        
        # Feature 4: Threading
        self.active_threads = []
        self._threads_lock = threading.Lock()

    def _init_csv(self):
        headers = [
            "Gaze direction", 
            "Timestamp start", 
            "Timestamp finish", 
            "Violation label", 
            "Numerical behavioural score", 
            "Video Evidence File"
        ]
        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _log_state_to_csv(self, direction, start_time, finish_time, label, score, video_file):
        start_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        finish_str = datetime.fromtimestamp(finish_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        with open(self.csv_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([direction, start_str, finish_str, label, score, video_file])

    def update(self, frame, gaze_direction: str):
        current_time = time.time()
        
        # Feature 3: Pre-roll buffer
        self.frame_buffer.append(frame)
        
        # Handle state transitions for CSV Logging
        if gaze_direction != self.current_state_direction:
            # Gaze state has finished, log it
            self._log_state_to_csv(
                self.current_state_direction,
                self.state_start_time,
                current_time,
                self.current_violation_label,
                self.current_score,
                self.current_video_file
            )
            
            # Feature 2: Record shifts from Center to anything else
            if self.current_state_direction == "Center" and gaze_direction != "Center":
                self.shift_timestamps.append(current_time)
                
            # Reset state for new direction
            self.current_state_direction = gaze_direction
            self.state_start_time = current_time
            self.current_violation_label = "normal"
            self.current_score = 0
            self.current_video_file = ""

        # Maintain 60-second rolling window for shifts
        while self.shift_timestamps and current_time - self.shift_timestamps[0] > 60.0:
            self.shift_timestamps.popleft()

        violation_reason = None

        # Check thresholds only if not currently recording post-roll
        if not self.is_recording:
            # Duration Check
            duration = current_time - self.state_start_time
            if gaze_direction in ["Left", "Right", "Up"] and duration > 3.0:
                violation_reason = f"{gaze_direction}_duration"
            elif gaze_direction == "Down" and duration > 5.0:
                violation_reason = f"{gaze_direction}_duration"
                
            # Frequency Check
            if not violation_reason and len(self.shift_timestamps) > 6:
                violation_reason = "frantic eye movement"
                
            # Trigger Execution
            if violation_reason:
                self.is_recording = True
                self.recording_reason = violation_reason
                
                # Update CSV logic state
                self.current_violation_label = violation_reason
                self.current_score = 100
                
                # Define video filename early for linkage
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp_str}_violation_{violation_reason}.mp4"
                self.current_video_file = filename
                self.recording_filename = filename
                
                # Copy 90 deque frames for pre-roll
                self.pre_roll_copy = [f.copy() for f in self.frame_buffer]
                
                # Clear shift history to avoid back-to-back frantic triggers
                self.shift_timestamps.clear()
        else:
            # Post-roll Capture
            self.post_roll.append(frame.copy())
            
            if len(self.post_roll) >= 90:
                # Trigger Feature 4: Threading
                t = threading.Thread(
                    target=self._save_video_async,
                    args=(self.pre_roll_copy, self.post_roll, self.recording_reason, self.recording_filename),
                    daemon=True
                )
                with self._threads_lock:
                    self.active_threads.append(t)
                t.start()
                
                # Reset recording state
                self.is_recording = False
                self.post_roll = []
                self.pre_roll_copy = []
                self.recording_reason = ""
                self.recording_filename = ""
                
        # Feature 3: Visual Feedback Payload
        return {
            "is_recording": self.is_recording,
            "current_violation": self.recording_reason if self.is_recording else ""
        }

    def _save_video_async(self, pre_frames, post_frames, reason, filename):
        if not pre_frames and not post_frames:
            return
            
        all_frames = pre_frames + post_frames
        
        # Dynamically extract resolution from the first frame
        first_frame = all_frames[0]
        height, width = first_frame.shape[:2]
        
        # Change filename extension from .mp4 to .avi for better Windows compatibility
        filename = filename.replace('.mp4', '.avi')
        
        # Use cv2.VideoWriter with XVID codec which is highly reliable on Windows
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        
        for f in all_frames:
            out.write(f)
            
        out.release()
        import os
        abs_path = os.path.abspath(filename)
        print(f"[SuspicionScoring] Video evidence saved successfully to: {abs_path}")
        
        # Thread cleanup
        with self._threads_lock:
            current_thread = threading.current_thread()
            if current_thread in self.active_threads:
                self.active_threads.remove(current_thread)

    def cleanup(self):
        # Log the final state
        self._log_state_to_csv(
            self.current_state_direction,
            self.state_start_time,
            time.time(),
            self.current_violation_label,
            self.current_score,
            self.current_video_file
        )
        
        # Feature 4: Graceful Shutdown
        with self._threads_lock:
            threads_copy = list(self.active_threads)
        for t in threads_copy:
            if t.is_alive():
                print("[SuspicionScoring] Waiting for video save to finish...")
                t.join()
