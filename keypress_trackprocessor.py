import keyboard
import threading

class KeypressTrackProcessor:
    def __init__(self):
        self.suspicious_keys_pressed = []
        self._lock = threading.Lock()
        
        # Register forbidden hotkeys
        # We use a non-blocking hook via the keyboard library
        keyboard.add_hotkey('alt+tab', self._on_hotkey, args=('alt+tab',))
        keyboard.add_hotkey('ctrl+c', self._on_hotkey, args=('ctrl+c',))
        keyboard.add_hotkey('ctrl+v', self._on_hotkey, args=('ctrl+v',))
        keyboard.add_hotkey('ctrl+x', self._on_hotkey, args=('ctrl+x',))

    def _on_hotkey(self, key_combo):
        """Callback triggered when a forbidden chord is pressed."""
        with self._lock:
            self.suspicious_keys_pressed.append(key_combo)

    def get_suspicious_keys(self):
        """
        Returns a list of forbidden chords pressed since the last call,
        then clears the buffer.
        """
        with self._lock:
            keys = list(self.suspicious_keys_pressed)
            self.suspicious_keys_pressed.clear()
            return keys

    def cleanup(self):
        """Unregister all hotkeys to prevent lingering hooks on exit."""
        keyboard.unhook_all()
