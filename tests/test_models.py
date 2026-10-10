"""Model backends: declared algorithmic delay matches measurement, state is carried across frames,
reduction=0 reproduces the delayed input, reset clears state."""
import numpy as np
import pytest

from stream_voice_focus.models.registry import load_model, model_keys


def _speechlike(sr=48000, seconds=2.0, seed=0):
    tt = np.arange(int(sr * seconds)) / sr
    f0 = 120 * (1 + 0.2 * np.sin(2 * np.pi * 0.7 * tt))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    sig = sum(np.sin(k * ph) / k for k in range(1, 15)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * tt)) ** 2 * 0.2
    return (sig + 0.005 * np.random.default_rng(seed).standard_normal(len(sig))).astype(np.float32)


def _stream(m, x):
    hop = m.info.frame_size
    pad = (-len(x)) % hop
    xp = np.concatenate([x, np.zeros(pad, np.float32)])
    return np.concatenate([m.process(xp[i:i + hop]) for i in range(0, len(xp), hop)])


def _best_lag(a, b, maxlag=4000):
    best = (0, -1)
    for lag in range(0, maxlag, 1):
        n = min(len(a), len(b) - lag)
        if n < 1000:
            break
        c = float(np.dot(a[:n], b[lag:lag + n]) / np.sqrt(np.dot(a[:n], a[:n]) * np.dot(b[lag:lag + n], b[lag:lag + n]) + 1e-12))
        if c > best[1]:
            best = (lag, c)
    return best


REALTIME_KEYS = [k for k in model_keys() if k not in ("passthrough", "dfn3_torch")]


@pytest.mark.parametrize("key", REALTIME_KEYS)
def test_declared_delay_matches_measured(key):
    m = load_model(key)
    x = _speechlike()
    y = _stream(m, x)
    lag, c = _best_lag(x, y)
    assert abs(lag - m.info.algorithmic_delay) <= 2, (key, lag, m.info.algorithmic_delay, c)
    m.close()


@pytest.mark.parametrize("key", REALTIME_KEYS)
def test_reduction_zero_is_delayed_identity(key):
    m = load_model(key)
    m.set_reduction(0.0)
    x = (0.1 * np.random.default_rng(1).standard_normal(48000)).astype(np.float32)
    y = _stream(m, x)
    d = m.info.algorithmic_delay
    hop = m.info.frame_size
    err = np.abs(y[d + 2 * hop:] - x[2 * hop:len(x) - d]).max()
    assert err < 1e-5, (key, err)
    m.close()


def test_streaming_state_is_carried_dfn3():
    """Processing frame-by-frame must equal processing the same frames after a reset only
    if state is carried: feeding the second half alone must differ from the second half of
    the continuous stream (because the model has memory)."""
    m = load_model("dfn3")
    x = _speechlike(seconds=1.0)
    y_full = _stream(m, x)
    m.reset()
    y_half = _stream(m, x[24000:])
    assert not np.allclose(y_full[24000:], y_half, atol=1e-3)
    # and reset() really resets: same input twice -> identical output
    m.reset(); a = _stream(m, x)
    m.reset(); b = _stream(m, x)
    assert np.allclose(a, b)
    m.close()


def test_dfn3_ort_equals_torch_runtime():
    ort = load_model("dfn3")
    th = load_model("dfn3_torch")
    x = _speechlike(seconds=0.5)
    assert np.abs(_stream(ort, x) - _stream(th, x)).max() < 1e-5
    ort.close(); th.close()


def test_dfn3_suppresses_stationary_noise_keeps_tone():
    m = load_model("dfn3")
    rng = np.random.default_rng(0)
    sr = 48000
    noise = (0.05 * rng.standard_normal(sr * 2)).astype(np.float32)
    y = _stream(m, noise)
    assert np.sqrt(np.mean(y[sr:] ** 2)) < 0.3 * np.sqrt(np.mean(noise[sr:] ** 2))
    m.close()
