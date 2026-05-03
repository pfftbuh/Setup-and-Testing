import csv
import collections
import threading
import time
from datetime import datetime
import cv2
import os

class SuspicionScoringProcessor:
    def __init__(self, side_threshold=3.0, down_threshold=5.0, freq_threshold=6, screen_width=1920, screen_height=1080):
        self.side_threshold = side_threshold
        self.down_threshold = down_threshold
        self.freq_threshold = freq_threshold
        self.screen_width = screen_width
        self.screen_height = screen_height

        # CSV Logging Initialization
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"session_log_{self.session_timestamp}.csv"
        self._init_csv()

        # State Machine Tracking
        self.current_state_direction = "Center"
        self.state_start_time = time.time()
        self.current_violation_label = "normal"
        self.current_score = 0
        self.current_video_file = ""

        self.shift_timestamps = collections.deque()

        # Optimized Video Capture
        self.frame_buffer = collections.deque(maxlen=90)  # 3 seconds at 30 fps
        self.is_recording = False
        self.post_roll = []
        self.pre_roll_copy = []
        self.recording_reason = ""
        self.recording_filename = ""

        # Threading
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

    def update(self, frame, raw_gaze_direction, eye_screen_pos: tuple, face_screen_pos: tuple, keystrokes: list):
        current_time = time.time()
        
        # Continuous Pre-roll
        if frame is not None:
            self.frame_buffer.append(frame)

        # Sanitization
        if isinstance(raw_gaze_direction, (list, tuple)):
            if len(raw_gaze_direction) > 0:
                gaze_direction = str(raw_gaze_direction[0])
            else:
                gaze_direction = "Center"
        else:
            gaze_direction = str(raw_gaze_direction) if raw_gaze_direction else "Center"

        # Handle state transitions for CSV Logging
        if gaze_direction != self.current_state_direction:
            self._log_state_to_csv(
                self.current_state_direction,
                self.state_start_time,
                current_time,
                self.current_violation_label,
                self.current_score,
                self.current_video_file
            )
            
            if self.current_state_direction == "Center" and gaze_direction != "Center":
                self.shift_timestamps.append(current_time)
                
            self.current_state_direction = gaze_direction
            self.state_start_time = current_time
            self.current_violation_label = "normal"
            self.current_score = 0
            self.current_video_file = ""

        while self.shift_timestamps and current_time - self.shift_timestamps[0] > 60.0:
            self.shift_timestamps.popleft()

        violation_reason = None

        if not self.is_recording:
            # 1. Keystrokes
            if keystrokes:
                violation_reason = f"forbidden_key_{keystrokes[0]}"
            
            # 2. Face Boundaries
            elif not violation_reason and face_screen_pos is not None:
                fx, fy = face_screen_pos
                if not (0 <= fx <= self.screen_width and 0 <= fy <= self.screen_height):
                    violation_reason = "face_off_screen"
            
            # 3. Eye Boundaries
            elif not violation_reason and eye_screen_pos is not None:
                ex, ey = eye_screen_pos
                if not (0 <= ex <= self.screen_width and 0 <= ey <= self.screen_height):
                    violation_reason = "eyes_off_screen"
            
            # 4. Frequency
            elif not violation_reason and len(self.shift_timestamps) > self.freq_threshold:
                violation_reason = "frantic_eye_movement"
                
            # 5. Duration
            elif not violation_reason:
                duration = current_time - self.state_start_time
                if gaze_direction in ["Left", "Right", "Up"] and duration > self.side_threshold:
                    violation_reason = f"{gaze_direction}_duration"
                elif gaze_direction == "Down" and duration > self.down_threshold:
                    violation_reason = f"{gaze_direction}_duration"
                
            # Trigger Execution
            if violation_reason:
                self.is_recording = True
                self.recording_reason = violation_reason
                self.current_violation_label = violation_reason
                self.current_score = 100
                
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Strip invalid characters from reason if any (e.g., from keystrokes)
                safe_reason = "".join(x for x in violation_reason if x.isalnum() or x in "_-")
                filename = f"{timestamp_str}_violation_{safe_reason}.mp4"
                self.current_video_file = filename
                self.recording_filename = filename
                
                self.pre_roll_copy = [f.copy() for f in self.frame_buffer]
                self.shift_timestamps.clear()
        else:
            # Post-roll Capture
            if frame is not None:
                self.post_roll.append(frame.copy())
            
            if len(self.post_roll) >= 90:
                t = threading.Thread(
                    target=self._save_video_async,
                    args=(self.pre_roll_copy, self.post_roll, self.recording_reason, self.recording_filename),
                    daemon=True
                )
                with self._threads_lock:
                    self.active_threads.append(t)
                t.start()
                
                self.is_recording = False
                self.post_roll = []
                self.pre_roll_copy = []
                self.recording_reason = ""
                self.recording_filename = ""
                
        return {
            "is_recording": self.is_recording,
            "current_violation": self.recording_reason if self.is_recording else ""
        }

    def _save_video_async(self, pre_frames, post_frames, reason, filename):
        if not pre_frames and not post_frames:
            return
            
        all_frames = pre_frames + post_frames
        first_frame = all_frames[0]
        height, width = first_frame.shape[:2]
        
        # Use mp4v codec for mp4 format as requested
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        
        for f in all_frames:
            out.write(f)
            
        out.release()
        abs_path = os.path.abspath(filename)
        print(f"[SuspicionScoring] Video evidence saved successfully to: {abs_path}")
        
        with self._threads_lock:
            current_thread = threading.current_thread()
            if current_thread in self.active_threads:
                self.active_threads.remove(current_thread)

    def cleanup(self):
        self._log_state_to_csv(
            self.current_state_direction,
            self.state_start_time,
            time.time(),
            self.current_violation_label,
            self.current_score,
            self.current_video_file
        )
        
        with self._threads_lock:
            threads_copy = list(self.active_threads)
        for t in threads_copy:
            if t.is_alive():
                print("[SuspicionScoring] Waiting for video save to finish...")
                t.join()
