"""Compatibility shims.

The `deepfilternet` 0.5.6 pip package (2023) imports `torchaudio.backend.common`,
which was removed in torchaudio >= 2.9. The DeepFilterNet model code itself only
uses it for file I/O type hints, which this app never calls (we read audio with
soundfile). Registering a tiny stand-in module lets the package import cleanly on
current torch/torchaudio without pinning an old torch.
"""
import sys
import types


def install_torchaudio_shim() -> None:
    try:
        import torchaudio.backend.common  # noqa: F401
        return
    except Exception:
        pass
    mod = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # minimal stand-in; never instantiated by this app
        def __init__(self, *a, **k):
            pass

    mod.AudioMetaData = AudioMetaData
    backend = sys.modules.get("torchaudio.backend")
    if backend is None:
        backend = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend"] = backend
    backend.common = mod
    sys.modules["torchaudio.backend.common"] = mod
