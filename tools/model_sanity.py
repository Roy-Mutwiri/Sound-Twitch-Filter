"""Sanity check every backend: load time, measured algorithmic delay, reduction=0 identity, per-frame cost."""
import sys
import time
import numpy as np

from stream_voice_focus.models.registry import load_model, model_keys


def run(key: str):
    t = time.time()
    m = load_model(key)
    print(f"{key}: {type(m).__name__} loaded in {time.time() - t:.1f}s; declared delay {m.info.algorithmic_delay}")
    hop = m.info.frame_size
    n = 48000
    imp = np.zeros(n, np.float32)
    imp[24000] = 0.5
    outs = [m.process(imp[i:i + hop]) for i in range(0, n, hop)]
    y = np.concatenate(outs)
    pk = int(np.argmax(np.abs(y)))
    print(f"  impulse delay measured: {pk - 24000} samples")
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(n)).astype(np.float32)
    m.reset()
    m.set_reduction(0.0)
    y = np.concatenate([m.process(x[i:i + hop]) for i in range(0, n, hop)])
    d = m.info.algorithmic_delay
    err = float(np.abs(y[d + hop * 2:] - x[hop * 2:n - d]).max()) if m.info.supports_reduction else float("nan")
    print(f"  reduction=0 vs delayed input, max abs err: {err:.2e}")
    m.set_reduction(1.0)
    m.reset()
    ts = []
    for i in range(0, n, hop):
        t0 = time.perf_counter()
        m.process(x[i:i + hop])
        ts.append((time.perf_counter() - t0) * 1e3)
    print(f"  per-frame ms: mean {np.mean(ts):.2f}  p95 {np.percentile(ts, 95):.2f}  max {np.max(ts):.2f}  (frame = {1000 * hop / m.info.sample_rate:.1f} ms)")
    m.close()


if __name__ == "__main__":
    for k in (sys.argv[1:] or model_keys()):
        run(k)
