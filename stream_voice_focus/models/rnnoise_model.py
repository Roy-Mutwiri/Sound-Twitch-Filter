"""RNNoise backend (xiph.org, BSD-3-Clause) via the `pyrnnoise` wheel (Apache-2.0 wrapper).

48 kHz, 480-sample (10 ms) frames. RNNoise is far lighter than DeepFilterNet3 but is a
small noise suppressor designed for ambient/stationary noise; it leaves music and loud
game effects largely intact. Offered as a low-CPU fallback. Algorithmic delay: 960 samples
(measured by cross-correlation, two 10 ms hops), verified in tests/test_models.py.

Reduction control: time-domain blend with the input delayed by the same 960 samples, so both
paths are sample-aligned.
"""
from __future__ import annotations

import ctypes
import time
import numpy as np

from .base import StreamingEnhancer, ModelInfo

SR = 48000
HOP = 480
ALGO_DELAY = 960  # measured by cross-correlation (tools/model_sanity.py); current rnnoise uses a 2-hop overlap


class RNNoiseEnhancer(StreamingEnhancer):
    info = ModelInfo(
        key="rnnoise",
        display_name="RNNoise (light, very low CPU)",
        description="Classic lightweight RNN noise suppressor. Very low CPU, weak against music and "
                    "game effects. 20 ms algorithmic delay.",
        sample_rate=SR,
        frame_size=HOP,
        algorithmic_delay=ALGO_DELAY,
        supports_reduction=True,
        license="RNNoise: BSD-3-Clause (Xiph.Org); pyrnnoise wrapper: Apache-2.0",
    )

    def __init__(self):
        from pyrnnoise import rnnoise as r
        self._lib = r.lib
        self._create = r.create
        self._destroy = r.destroy
        self._state = self._create()
        self._reduction = 1.0
        self._delay_buf = np.zeros(ALGO_DELAY, np.float32)
        self.last_infer_ms = 0.0
        self.last_speech_prob = 0.0

    def reset(self) -> None:
        if self._state is not None:
            self._destroy(self._state)
        self._state = self._create()
        self._delay_buf[:] = 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        x = np.ascontiguousarray(frame, dtype=np.float32)
        buf = x * 32768.0
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.last_speech_prob = float(self._lib.rnnoise_process_frame(self._state, ptr, ptr))
        y = buf * (1.0 / 32768.0)
        delayed = self._delay_buf[:HOP].copy()
        self._delay_buf = np.concatenate([self._delay_buf[HOP:], x])
        if self._reduction < 0.999:
            y = (1.0 - self._reduction) * delayed + self._reduction * y
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return y

    def close(self) -> None:
        if self._state is not None:
            self._destroy(self._state)
            self._state = None
