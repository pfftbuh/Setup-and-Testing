import keyboard
import threading

class KeypressTrackProcessor:
    def __init__(self):
        self.suspicious_keys_buffer = []
        self._lock = threading.Lock()
        self._register_hotkeys()

    def _register_hotkeys(self):
        # We catch these chords
        chords = ['alt+tab', 'ctrl+c', 'ctrl+v', 'ctrl+x']
        for chord in chords:
            keyboard.add_hotkey(chord, self._on_suspicious_key, args=(chord,), suppress=False)

    def _on_suspicious_key(self, keyname):
        with self._lock:
            self.suspicious_keys_buffer.append(keyname)

    def get_suspicious_keys(self):
        with self._lock:
            keys = list(self.suspicious_keys_buffer)
            self.suspicious_keys_buffer.clear()
            return keys
