import keyboard
import threading

class KeypressTrackProcessor:
    def __init__(self):
        self.suspicious_keys_pressed = []
        self._lock = threading.Lock()
        
        # Register forbidden hotkeys
        # We use a non-blocking hook via the keyboard library
        
        # Standard copy/paste/switching
        keyboard.add_hotkey('alt+tab', self._on_hotkey, args=('alt+tab',))
        keyboard.add_hotkey('ctrl+c', self._on_hotkey, args=('ctrl+c',))
        keyboard.add_hotkey('ctrl+v', self._on_hotkey, args=('ctrl+v',))
        keyboard.add_hotkey('ctrl+x', self._on_hotkey, args=('ctrl+x',))

        # Navigation & App Switching
        keyboard.add_hotkey('windows', self._on_hotkey, args=('windows',))
        keyboard.add_hotkey('ctrl+esc', self._on_hotkey, args=('ctrl+esc',))
        keyboard.add_hotkey('ctrl+shift+esc', self._on_hotkey, args=('ctrl+shift+esc',))
        
        # Browser & Search Exploits
        keyboard.add_hotkey('ctrl+t', self._on_hotkey, args=('ctrl+t',))
        keyboard.add_hotkey('ctrl+n', self._on_hotkey, args=('ctrl+n',))
        keyboard.add_hotkey('ctrl+w', self._on_hotkey, args=('ctrl+w',))
        keyboard.add_hotkey('ctrl+f4', self._on_hotkey, args=('ctrl+f4',))
        keyboard.add_hotkey('f11', self._on_hotkey, args=('f11',))
        keyboard.add_hotkey('esc', self._on_hotkey, args=('esc',))
        
        # Exam Content Theft
        keyboard.add_hotkey('print screen', self._on_hotkey, args=('print screen',))
        keyboard.add_hotkey('windows+shift+s', self._on_hotkey, args=('windows+shift+s',))
        
        # Application Evasion
        keyboard.add_hotkey('alt+f4', self._on_hotkey, args=('alt+f4',))

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
