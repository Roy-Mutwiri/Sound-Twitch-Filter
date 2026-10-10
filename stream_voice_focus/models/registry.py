"""Model registry: stable keys -> lazy loaders (torch / onnxruntime imports are slow)."""
from __future__ import annotations

from typing import Callable, Dict, List
from .base import StreamingEnhancer, ModelInfo, Passthrough
from .dfn3 import DFN3_INFO, DeepFilterNet3Torch, load_dfn3
from .rnnoise_model import RNNoiseEnhancer

import importlib.util

_HAS_TORCH = importlib.util.find_spec("torch") is not None   # absent in the packaged exe

MODEL_INFOS: List[ModelInfo] = [
    DFN3_INFO,
    *([DeepFilterNet3Torch.info] if _HAS_TORCH else []),
    RNNoiseEnhancer.info,
    Passthrough.info,
]

_LOADERS: Dict[str, Callable[[], StreamingEnhancer]] = {
    "dfn3": lambda: load_dfn3("ort"),
    "dfn3_torch": lambda: DeepFilterNet3Torch(),
    "rnnoise": lambda: RNNoiseEnhancer(),
    "passthrough": lambda: Passthrough(),
}

try:  # optional extra backend
    from .dpdfnet_model import DPDFNetEnhancer, DPDFNET_INFOS, dpdfnet_loader
    for pos, _info in enumerate(DPDFNET_INFOS):
        MODEL_INFOS.insert(pos, _info)          # DPDFNet-2 first (default), then DPDFNet-8, DFN3, ...
        _LOADERS[_info.key] = dpdfnet_loader(_info.key)
except Exception:  # pragma: no cover - package not installed
    pass


def model_keys() -> List[str]:
    return [m.key for m in MODEL_INFOS]


def model_info(key: str) -> ModelInfo:
    for m in MODEL_INFOS:
        if m.key == key:
            return m
    raise KeyError(key)


def load_model(key: str) -> StreamingEnhancer:
    return _LOADERS[key]()
