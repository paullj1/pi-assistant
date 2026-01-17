from collections import deque
import time

import numpy as np


class RMSGate:
    def __init__(self, window_seconds: float, floor: float, frame_seconds: float = 0.02):
        self._window_frames = max(1, int(window_seconds / frame_seconds))
        self._values = deque(maxlen=self._window_frames)
        self._floor = floor

    def update(self, rms: float) -> float:
        self._values.append(rms)
        return self.threshold()

    def threshold(self) -> float:
        if not self._values:
            return self._floor
        avg = sum(self._values) / len(self._values)
        return max(self._floor, avg)

    def should_send(self, rms: float) -> bool:
        threshold = self.update(rms)
        return rms >= threshold


class KeepAliveGate:
    def __init__(self, interval_seconds: float):
        self._interval = interval_seconds
        self._last_sent = 0.0

    def should_send(self) -> bool:
        now = time.time()
        if now - self._last_sent >= self._interval:
            self._last_sent = now
            return True
        return False


def pcm_rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio**2)))
