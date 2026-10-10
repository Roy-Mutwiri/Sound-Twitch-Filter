"""DeepFilterNet3 streaming backend (48 kHz, 480-sample / 10 ms frames, 30 ms algorithmic delay).

Two runtimes:
  * "ort"   - ONNX Runtime session built from `models_cache/dfn3_stream.onnx` (exported by
              tools/export_onnx.py from the official DeepFilterNet3 checkpoint). About 1.6 ms
              per 10 ms frame on one CPU thread (Ryzen 9 7950X).
  * "torch" - the pure-PyTorch streaming graph (vendored from the DeepFilterNet repository's
              torchDF port, file _torchdf_minimal.py). About 5 ms per frame. Used automatically
              when the ONNX file is missing.

Both carry the full model state (GRU hidden states, conv buffers, STFT overlap buffers and
feature-normalisation state) between frames, so the output is identical to running the
model over a whole file (verified in tools/align_test.py: max difference ~1e-8 vs torch and
rms 0.0035 vs the official offline `enhance()` once the 1440-sample delay is accounted for).

Background reduction uses DeepFilterNet's own `atten_lim_db` mechanism: the final enhanced
frame is blended with the noisy frame of the same STFT time index,
    out = lim * noisy + (1 - lim) * enhanced,   lim = 10 ** (-atten_lim_db / 20)
which is what `df.enhance.enhance(..., atten_lim_db=...)` does offline. Because both terms
are the same STFT frame there is no time offset between them.
"""
from __future__ import annotations

import os
import time
import numpy as np

from .base import StreamingEnhancer, ModelInfo

SR = 48000
HOP = 480
FFT = 960
ALGO_DELAY = 1440  # (FFT - HOP) + 2 lookahead frames * HOP; measured with an impulse in tests

LICENSE = "DeepFilterNet: MIT OR Apache-2.0 (Hendrik Schroeter et al.)"

DFN3_INFO = ModelInfo(
    key="dfn3",
    display_name="DeepFilterNet3 (lowest delay 30 ms, most transparent)",
    description=("Full-band 48 kHz deep speech enhancement. 10 ms frames, 30 ms algorithmic "
                 "delay. Trained to remove noise and reverb; music and game effects are treated "
                 "as noise with mixed success. Other voices are kept."),
    sample_rate=SR,
    frame_size=HOP,
    algorithmic_delay=ALGO_DELAY,
    supports_reduction=True,
    license=LICENSE,
)


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_onnx_path() -> str:
    import sys
    if os.environ.get("SVF_MODELS_DIR"):
        root = os.environ["SVF_MODELS_DIR"]
    elif getattr(sys, "frozen", False):   # PyInstaller bundle
        root = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "models_cache")
    else:
        root = os.path.join(project_root(), "models_cache")
    return os.path.join(root, "dfn3_stream.onnx")


def _initial_states_numpy():
    """Initial state tensors, identical to TorchDFMinimalPipeline, without importing torch."""
    nb_bands, nb_df, df_order, conv_la, emb = 32, 96, 5, 2, 256
    freq = FFT // 2 + 1
    return [
        np.linspace(-60.0, -90.0, nb_bands, dtype=np.float32),                      # erb_norm_state
        np.linspace(0.001, 0.0001, nb_df, dtype=np.float32).reshape(1, nb_df, 1),   # band_unit_norm_state
        np.zeros((HOP,), np.float32),                                                # analysis_mem
        np.zeros((HOP,), np.float32),                                                # synthesis_mem
        np.zeros((1, 1, conv_la + 1, nb_bands), np.float32),                         # rolling_erb_buf
        np.zeros((1, 2, conv_la + 1, nb_df), np.float32),                            # rolling_feat_spec_buf
        np.zeros((1, 64, df_order, nb_df), np.float32),                              # rolling_c0_buf
        np.zeros((max(df_order, conv_la), freq, 2), np.float32),                     # rolling_spec_buf_x
        np.zeros((df_order + conv_la, freq, 2), np.float32),                         # rolling_spec_buf_y
        np.zeros((1, 1, emb), np.float32),                                           # enc_hidden
        np.zeros((2, 1, emb), np.float32),                                           # erb_dec_hidden
        np.zeros((2, 1, emb), np.float32),                                           # df_dec_hidden
    ]


def _vorbis_window() -> np.ndarray:
    n = np.arange(FFT)
    w = np.sin(0.5 * np.pi * (n + 0.5) / (FFT // 2))
    return np.sin(0.5 * np.pi * w ** 2).astype(np.float32)


class _DFN3Common(StreamingEnhancer):
    info = DFN3_INFO

    def __init__(self):
        self._reduction = 1.0
        self.last_infer_ms = 0.0
        self.window = _vorbis_window()
        self._blend_mem = np.zeros(HOP, np.float32)

    def _blend(self, y: np.ndarray, noisy_spec: np.ndarray) -> np.ndarray:
        """Blend enhanced frame y with the noisy frame synthesised from the same STFT frame."""
        if self._reduction >= 0.999:
            self._blend_mem[:] = 0.0
            return y
        cs = noisy_spec[:, 0] + 1j * noisy_spec[:, 1]
        x = (np.fft.irfft(cs, n=FFT) * FFT).astype(np.float32) * self.window
        noisy_frame = x[:HOP] + self._blend_mem
        self._blend_mem = x[HOP:].copy()
        lim = 1.0 - self._reduction
        return lim * noisy_frame + (1.0 - lim) * y


class DeepFilterNet3ORT(_DFN3Common):
    """ONNX Runtime implementation."""

    def __init__(self, onnx_path: str | None = None, threads: int = 1):
        super().__init__()
        import onnxruntime as ort
        self.path = onnx_path or default_onnx_path()
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.states = _initial_states_numpy()

    def reset(self) -> None:
        self.states = _initial_states_numpy()
        self._blend_mem[:] = 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        feed = dict(zip(self.in_names, [np.ascontiguousarray(frame, dtype=np.float32), *self.states]))
        out = self.sess.run(None, feed)
        self.states = out[1:]
        y = self._blend(out[0], self.states[8][4])   # rolling_spec_buf_y[df_order-1]
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return y


class DeepFilterNet3Torch(_DFN3Common):
    """Pure-PyTorch streaming implementation (fallback when the ONNX file is missing)."""

    info = ModelInfo(
        key="dfn3_torch",
        display_name="DeepFilterNet3 (streaming, PyTorch runtime)",
        description=DFN3_INFO.description + " PyTorch eager runtime, about 3x the per-frame cost of ONNX Runtime.",
        sample_rate=SR, frame_size=HOP, algorithmic_delay=ALGO_DELAY, supports_reduction=True, license=LICENSE,
    )

    def __init__(self, threads: int = 1):
        super().__init__()
        import torch
        from stream_voice_focus.compat import install_torchaudio_shim
        install_torchaudio_shim()
        torch.set_num_threads(threads)
        from ._torchdf_minimal import TorchDFMinimalPipeline
        self.torch = torch
        self.pipe = TorchDFMinimalPipeline()
        self.model = self.pipe.torch_streaming_model.eval()
        self.states = [s.clone() for s in self.pipe.states]

    def reset(self) -> None:
        self.states = [s.clone() for s in self.pipe.states]
        self._blend_mem[:] = 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        torch = self.torch
        with torch.no_grad():
            y, *self.states = self.model(torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)), *self.states)
        y = self._blend(y.numpy(), self.states[8][4].numpy())
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return y


def load_dfn3(prefer: str = "ort", threads: int = 1) -> _DFN3Common:
    if prefer == "ort":
        try:
            return DeepFilterNet3ORT(threads=threads)
        except FileNotFoundError:
            pass
    return DeepFilterNet3Torch(threads=threads)
