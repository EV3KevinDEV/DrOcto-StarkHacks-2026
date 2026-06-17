from __future__ import annotations

import threading


class VoiceModeController:
    def __init__(self, initial_enabled: bool = False):
        self._enabled = initial_enabled
        self._lock = threading.Lock()

    def get_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> bool:
        with self._lock:
            self._enabled = bool(enabled)
            return self._enabled

    def toggle(self) -> bool:
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled
