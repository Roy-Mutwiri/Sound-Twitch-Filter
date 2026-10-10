"""DPDFNet backend (Ceva, Apache-2.0) via the `dpdfnet` package's StreamEnhancer.

48 kHz "hr" variants use a 960-sample window / 480-sample hop like DeepFilterNet3. The
package's streaming wrapper returns audio with a fixed 2400-sample (50 ms) delay relative to
the input (measured with an impulse and by cross-correlation in tools/model_sanity.py), i.e.
20 ms more than DeepFilterNet3. Per-frame cost on one CPU thread (Ryzen 9 7950X):
dpdfnet2_48khz_hr ~2.2 ms, dpdfnet8_48khz_hr ~6.3 ms (p95 ~8 ms, close to the 10 ms budget).

Weights are downloaded by the package from Hugging Face (Ceva-IP/DPDFNet) on first use and
cached locally; afterwards it runs fully offline.

Reduction control: time-domain blend with the input delayed by the same 2400 samples.
"""
from __future__ import annotations

import time
from typing import Callable, List

import numpy as np

from .base import StreamingEnhancer, ModelInfo

SR = 48000
HOP = 480
ALGO_DELAY = 2400

_LIC = "DPDFNet: Apache-2.0 (Ceva, Inc.)"

DPDFNET_INFOS: List[ModelInfo] = [
    ModelInfo(
        key="dpdfnet2_48k",
        display_name="DPDFNet-2 48 kHz (recommended: strongest on music, 50 ms delay)",
        description=("Dual-path DeepFilterNet variant (Ceva, 2025). 50 ms algorithmic delay, ~1.5x the CPU "
                     "of DeepFilterNet3. In file tests: about +3 dB more music suppression and higher PESQ/STOI "
                     "than DeepFilterNet3 on 0 dB speech+music, slightly less transparent on clean speech."),
        sample_rate=SR, frame_size=HOP, algorithmic_delay=ALGO_DELAY, supports_reduction=True, license=_LIC,
    ),
    ModelInfo(
        key="dpdfnet8_48k",
        display_name="DPDFNet-8 48 kHz (max suppression, heavy CPU, 50 ms delay)",
        description=("Largest DPDFNet. 50 ms algorithmic delay, ~5x the CPU of DeepFilterNet3 "
                     "(p95 ~8 ms per 10 ms frame on one thread: little headroom)."),
        sample_rate=SR, frame_size=HOP, algorithmic_delay=ALGO_DELAY, supports_reduction=True, license=_LIC,
    ),
]

_NAMES = {"dpdfnet2_48k": "dpdfnet2_48khz_hr", "dpdfnet8_48k": "dpdfnet8_48khz_hr"}


class DPDFNetEnhancer(StreamingEnhancer):
    def __init__(self, key: str):
        from dpdfnet import StreamEnhancer
        self.info = next(i for i in DPDFNET_INFOS if i.key == key)
        self._se = StreamEnhancer(model=_NAMES[key], verbose=False)
        self._reduction = 1.0
        self._delay_buf = np.zeros(ALGO_DELAY, np.float32)
        self._primed = False
        self.last_infer_ms = 0.0

    def reset(self) -> None:
        self._se.reset()
        self._delay_buf[:] = 0.0
        self._primed = False

    def process(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        x = np.ascontiguousarray(frame, dtype=np.float32)
        y = self._se.process(x, SR)
        if len(y) == 0:
            y = np.zeros(HOP, np.float32)   # first call: wrapper is filling its window
        elif len(y) != HOP:
            y = y[-HOP:] if len(y) > HOP else np.pad(y, (0, HOP - len(y)))
        # delay line for the sample-aligned original (2400 = 5 hops)
        delayed = self._delay_buf[:HOP].copy()
        self._delay_buf = np.concatenate([self._delay_buf[HOP:], x])
        if self._reduction < 0.999:
            y = (1.0 - self._reduction) * delayed + self._reduction * y
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return y


def dpdfnet_loader(key: str) -> Callable[[], StreamingEnhancer]:
    return lambda: DPDFNetEnhancer(key)
