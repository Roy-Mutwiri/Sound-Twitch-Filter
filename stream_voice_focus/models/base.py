"""Common interface for streaming speech-enhancement backends.

Every backend processes exactly one frame of mono float32 audio at its native
sample rate per call, carries its own internal state between calls, and declares
its algorithmic delay (how many samples the output lags the input by design,
independent of compute time). The audio engine uses `algorithmic_delay` to keep
the untouched "original" path time-aligned with the processed path.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ModelInfo:
    key: str                 # stable id used in settings
    display_name: str
    description: str
    sample_rate: int         # native rate
    frame_size: int          # samples per process() call at native rate
    algorithmic_delay: int   # samples at native rate (verified by tests/test_models.py)
    supports_reduction: bool # True if the reduction control is applied inside the backend
    license: str
    realtime: bool = True    # False = offline / high-delay quality mode


class StreamingEnhancer:
    """Base class. Instances are used from a single worker thread."""

    info: ModelInfo
    last_infer_ms: float = 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process one mono float32 frame of exactly info.frame_size samples; returns same length."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear all internal state (used after a dropout/recovery so stale audio is not replayed)."""
        raise NotImplementedError

    def set_reduction(self, amount: float) -> None:
        """Background reduction in [0, 1]. 0 = original passes through, 1 = full model output.

        Implemented as a *blend* between the model output and the original signal delayed by
        exactly the model's algorithmic delay (sample-aligned, so no echo / comb filtering).
        It is not a model-native "suppress harder" knob: the network always runs at full
        strength. For DeepFilterNet this is the same thing its own `atten_lim_db` does.
        """
        self._reduction = float(min(1.0, max(0.0, amount)))

    @property
    def reduction(self) -> float:
        return getattr(self, "_reduction", 1.0)

    def close(self) -> None:
        pass


class Passthrough(StreamingEnhancer):
    """No processing. For routing/latency tests and as a reference path."""

    info = ModelInfo(
        key="passthrough",
        display_name="Passthrough (no processing)",
        description="Routes audio through the engine untouched. For routing and latency tests only.",
        sample_rate=48000,
        frame_size=480,
        algorithmic_delay=0,
        supports_reduction=False,
        license="-",
    )

    def __init__(self):
        self._reduction = 1.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def reset(self) -> None:
        pass
