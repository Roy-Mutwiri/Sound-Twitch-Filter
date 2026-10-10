"""Single-producer / single-consumer ring buffer for float32 audio.

Lock-free under CPython: the producer only writes `_w`, the consumer only writes `_r`,
and both are monotonically increasing Python ints (assignment is atomic under the GIL).
Capacity is fixed, so a slow consumer can never make the buffer grow: `push` refuses data
that does not fit and reports how many samples were dropped.
"""
from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int, channels: int = 1):
        self.capacity = int(capacity)
        self.channels = int(channels)
        self._buf = np.zeros((self.capacity, self.channels), dtype=np.float32)
        self._w = 0  # total samples written (producer-owned)
        self._r = 0  # total samples read (consumer-owned)

    # ---- queries (safe from either side) -------------------------------------------------
    @property
    def available(self) -> int:
        return self._w - self._r

    @property
    def free(self) -> int:
        return self.capacity - (self._w - self._r)

    @property
    def total_written(self) -> int:
        return self._w

    @property
    def total_read(self) -> int:
        return self._r

    # ---- producer side --------------------------------------------------------------------
    def push(self, data: np.ndarray) -> int:
        """Append data (shape [n] or [n, channels]). Returns number of samples dropped (not written)."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n = data.shape[0]
        free = self.free
        dropped = 0
        if n > free:
            dropped = n - free
            data = data[:free]
            n = free
        if n == 0:
            return dropped
        w = self._w % self.capacity
        first = min(n, self.capacity - w)
        self._buf[w:w + first] = data[:first]
        if first < n:
            self._buf[:n - first] = data[first:]
        self._w += n
        return dropped

    # ---- consumer side --------------------------------------------------------------------
    def pop(self, n: int, out: np.ndarray | None = None) -> np.ndarray:
        """Read exactly n samples (caller must check `available` first). Returns [n, channels]."""
        if n > self.available:
            raise ValueError("pop: not enough samples")
        r = self._r % self.capacity
        first = min(n, self.capacity - r)
        if out is None:
            out = np.empty((n, self.channels), dtype=np.float32)
        out[:first] = self._buf[r:r + first]
        if first < n:
            out[first:] = self._buf[:n - first]
        self._r += n
        return out

    def skip(self, n: int) -> int:
        """Discard up to n oldest samples. Returns the number discarded."""
        n = min(n, self.available)
        self._r += n
        return n

    def clear(self) -> None:
        """Consumer-side reset: drop everything currently buffered."""
        self._r = self._w
