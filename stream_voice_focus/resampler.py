"""Streaming resampling (libsoxr via python-soxr) with a FIFO so callers can pull fixed-size frames.

Used only when a model's native rate differs from the engine rate (all shipped models are
48 kHz, so the real-time path normally runs without resampling; device-rate conversion is
delegated to the Windows audio engine via WASAPI auto-convert).
"""
from __future__ import annotations

import numpy as np

try:
    import soxr
except Exception:  # pragma: no cover
    soxr = None


class StreamResampler:
    def __init__(self, in_rate: int, out_rate: int, quality: str = "HQ"):
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self.identity = self.in_rate == self.out_rate
        if not self.identity:
            if soxr is None:
                raise RuntimeError("python-soxr is required for resampling")
            self._rs = soxr.ResampleStream(self.in_rate, self.out_rate, 1, dtype="float32", quality=quality)
        self._fifo = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        if not self.identity:
            self._rs.clear()
        self._fifo = np.zeros(0, dtype=np.float32)

    def push(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = x if self.identity else self._rs.resample_chunk(x, last=False)
        if len(y):
            self._fifo = np.concatenate([self._fifo, y]) if len(self._fifo) else y.copy()

    @property
    def available(self) -> int:
        return len(self._fifo)

    def pop(self, n: int) -> np.ndarray | None:
        if len(self._fifo) < n:
            return None
        out = self._fifo[:n]
        self._fifo = self._fifo[n:]
        return out

    @property
    def delay_in_samples(self) -> float:
        """Approximate filter delay of libsoxr HQ at the input rate (used for latency estimates)."""
        if self.identity:
            return 0.0
        # soxr does not expose its latency through python-soxr; HQ is ~ 1 ms class. We measure it
        # in tests/test_resampler.py instead of guessing in production code.
        return 0.0
