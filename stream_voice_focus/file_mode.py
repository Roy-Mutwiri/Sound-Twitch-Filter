"""File-processing test mode: run a model over a WAV file frame by frame (the same streaming
path the real-time engine uses) and write the result, delay-compensated.

    python -m stream_voice_focus.file_mode input.wav output.wav [--model dfn3] [--reduction 1.0]

Any sample rate / channel count is accepted; audio is downmixed to mono and resampled to the
model rate with libsoxr, then written at the model rate. Use `--keep-rate` to resample back.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import soundfile as sf

from .models.registry import load_model, model_keys


def load_mono(path: str, target_sr: int) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != target_sr:
        import soxr
        x = soxr.resample(x, sr, target_sr, quality="HQ").astype(np.float32)
    return x


def process_array(model, x: np.ndarray, reduction: float = 1.0, progress=None):
    """Stream x (mono, model rate) through `model`; returns delay-compensated output of len(x)."""
    hop = model.info.frame_size
    d = model.info.algorithmic_delay
    model.reset()
    model.set_reduction(reduction)
    n = len(x)
    pad = (-n) % hop
    xp = np.concatenate([x, np.zeros(pad + d + hop, np.float32)])
    outs = []
    times = []
    nfr = len(xp) // hop
    for i in range(nfr):
        t0 = time.perf_counter()
        outs.append(model.process(xp[i * hop:(i + 1) * hop]))
        times.append(time.perf_counter() - t0)
        if progress and i % 100 == 0:
            progress(i / nfr)
    y = np.concatenate(outs)[d:d + n]
    stats = {"frames": nfr, "mean_ms": 1000 * float(np.mean(times)), "p95_ms": 1000 * float(np.percentile(times, 95)),
             "max_ms": 1000 * float(np.max(times)), "realtime_factor": float(np.sum(times)) / (n / model.info.sample_rate)}
    return y.astype(np.float32), stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--model", default="dfn3", choices=model_keys())
    ap.add_argument("--reduction", type=float, default=1.0, help="0..1 blend (1 = full model output)")
    ap.add_argument("--keep-rate", action="store_true", help="resample output back to the input file's rate")
    args = ap.parse_args(argv)

    model = load_model(args.model)
    sr_m = model.info.sample_rate
    x = load_mono(args.input, sr_m)
    y, st = process_array(model, x, args.reduction)
    out_sr = sr_m
    if args.keep_rate:
        in_sr = sf.info(args.input).samplerate
        if in_sr != sr_m:
            import soxr
            y = soxr.resample(y, sr_m, in_sr, quality="HQ").astype(np.float32)
            out_sr = in_sr
    sf.write(args.output, y, out_sr)
    print(f"{args.model}: {st['frames']} frames, {st['mean_ms']:.2f} ms/frame mean, p95 {st['p95_ms']:.2f}, "
          f"max {st['max_ms']:.2f}; realtime factor {st['realtime_factor']:.3f}; delay compensated "
          f"({model.info.algorithmic_delay} samples). Wrote {args.output}")
    model.close()


if __name__ == "__main__":
    sys.exit(main())
