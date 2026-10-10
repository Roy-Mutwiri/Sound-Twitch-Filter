"""Engine tests with simulated PortAudio streams (no real devices): bounded queues, overload
recovery, latency-matched bypass, clock drift handling, clean stop/start, device-failure events."""
import threading
import time
import types

import numpy as np
import pytest

from stream_voice_focus import engine as eng_mod
from stream_voice_focus.devices import DeviceInfo
from stream_voice_focus.engine import AudioEngine, EngineConfig
from stream_voice_focus.models.base import Passthrough, StreamingEnhancer, ModelInfo

SR = 48000
BLOCK = 480


class FakeClock:
    """One ticker drives both fake streams, like two devices on the same clock. `out_rate_ratio`
    > 1 makes the output device consume faster than the input produces (clock drift)."""

    def __init__(self):
        self.streams = {}
        self.out_rate_ratio = 1.0
        self._stop = threading.Event()
        self._t = None

    def start(self):
        if self._t is None:
            self._stop.clear()
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()

    def _run(self):
        period = BLOCK / SR
        t0 = time.perf_counter()
        n_in = 0
        n_out = 0.0
        tinfo = types.SimpleNamespace(inputBufferAdcTime=0.0, outputBufferDacTime=0.0, currentTime=0.0)
        while not self._stop.is_set():
            n_in += 1
            target = t0 + n_in * period
            while time.perf_counter() < target and not self._stop.is_set():
                time.sleep(0.0005)
            now = time.perf_counter()
            s_in = self.streams.get("in")
            if s_in and s_in.active:
                data = s_in.source(BLOCK) if s_in.source else np.zeros(BLOCK, np.float32)
                buf = np.repeat(data[:, None], s_in.channels, axis=1).astype(np.float32)
                tinfo.inputBufferAdcTime = now - s_in.latency
                s_in.cb(buf, BLOCK, tinfo, None)
            n_out += self.out_rate_ratio
            while n_out >= 1.0:
                n_out -= 1.0
                s_out = self.streams.get("out")
                if s_out and s_out.active:
                    buf = np.zeros((BLOCK, s_out.channels), np.float32)
                    tinfo.outputBufferDacTime = now + s_out.latency
                    s_out.cb(buf, BLOCK, tinfo, None)
                    s_out.sink.append(buf[:, 0].copy())

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1)
            self._t = None


CLOCK = FakeClock()


class FakeStream:
    def __init__(self, kind, device=None, samplerate=SR, channels=2, dtype="float32", blocksize=BLOCK,
                 latency="low", callback=None, finished_callback=None, extra_settings=None):
        self.kind, self.channels, self.cb, self.fin = kind, channels, callback, finished_callback
        self.latency = 0.02
        self.active = False
        self.source = None
        self.sink = []
        CLOCK.streams[kind] = self

    def start(self):
        self.active = True
        CLOCK.start()

    def stop(self):
        self.active = False

    def abort(self):
        self.active = False

    def close(self):
        self.active = False

    def simulate_disconnect(self):
        self.active = False
        if self.fin:
            self.fin()


@pytest.fixture
def fake_sd(monkeypatch):
    monkeypatch.setattr(eng_mod.sd, "InputStream", lambda **kw: FakeStream("in", **kw))
    monkeypatch.setattr(eng_mod.sd, "OutputStream", lambda **kw: FakeStream("out", **kw))
    monkeypatch.setattr(eng_mod, "wasapi_settings", lambda h: None)
    CLOCK.streams.clear()
    CLOCK.out_rate_ratio = 1.0
    yield CLOCK.streams
    CLOCK.stop()


IN_DEV = DeviceInfo(1, "CABLE Output (VB-Audio Virtual Cable)", "Windows WASAPI", 2, 0, 48000)
OUT_DEV = DeviceInfo(2, "Headphones (test)", "Windows WASAPI", 0, 2, 48000)


class DelayedIdentity(StreamingEnhancer):
    """Model stand-in with a known algorithmic delay and optional artificial compute time."""

    def __init__(self, sleep_ms=0.0, delay=960):
        self.info = ModelInfo("test", "test", "", SR, BLOCK, delay, True, "-")
        self._buf = np.zeros(delay, np.float32)
        self.sleep = sleep_ms / 1000
        self._reduction = 1.0

    def process(self, frame):
        if self.sleep:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < self.sleep:
                pass
        out = np.concatenate([self._buf, frame])
        self._buf = out[len(frame):]
        return out[:len(frame)] * 0.5   # distinguishable from the original path

    def reset(self):
        self._buf[:] = 0


def _sine_source(n_total, f=440.0):
    sig = (np.sin(2 * np.pi * f * np.arange(n_total) / SR) * 0.3).astype(np.float32)
    pos = [0]

    def src(n):
        i = pos[0]
        pos[0] += n
        return np.take(sig, np.arange(i, i + n) % len(sig))
    return sig, src


def test_passthrough_roundtrip_bounded(fake_sd):
    sig, src = _sine_source(SR * 4)
    eng = AudioEngine()
    cfg = EngineConfig(IN_DEV, OUT_DEV)
    eng.start(cfg, Passthrough())
    fake_sd["in"].source = src
    time.sleep(2.5)
    slips_mid = eng.stats().slips
    time.sleep(1.5)
    s = eng.stats()
    eng.stop()
    out = np.concatenate(fake_sd["out"].sink)
    assert s.frames_processed > 300
    assert s.backlog_drops == 0 and s.out_drops == 0 and s.underruns == 0
    assert s.backlog_ms < cfg.max_backlog_ms
    # the controller converges on the prefill target (start-up priming is trimmed) and then idles
    assert abs(s.slips - slips_mid) < 40, "no drift was simulated: controller should be idle once settled"
    assert abs(s.out_fill_ms - cfg.prefill_ms) < 12
    assert s.measured_latency_ms is not None and 20 < s.measured_latency_ms < 200
    # last second of output must be the input signal, intact (after settling)
    seg = out[-SR:]
    ref = sig
    best = max(abs(np.corrcoef(seg, np.take(ref, np.arange(k, k + SR) % len(ref)))[0, 1]) for k in range(0, 220, 2))
    assert best > 0.97


def test_overload_is_bounded_and_recovers(fake_sd):
    _, src = _sine_source(SR * 4)
    eng = AudioEngine()
    cfg = EngineConfig(IN_DEV, OUT_DEV, max_backlog_ms=80)
    model = DelayedIdentity(sleep_ms=14)    # slower than real time: 14 ms per 10 ms frame
    eng.start(cfg, model)
    fake_sd["in"].source = src
    time.sleep(2.0)
    s1 = eng.stats()
    assert s1.backlog_drops > 0, "overload must drop old audio instead of growing the queue"
    assert s1.backlog_ms <= cfg.max_backlog_ms + 10
    model.sleep = 0.0                       # load disappears -> no further drops
    time.sleep(0.5)
    d0 = eng.stats().backlog_drops
    time.sleep(1.0)
    s2 = eng.stats()
    eng.stop()
    assert s2.backlog_drops == d0
    assert eng.running is False


def test_bypass_is_latency_matched(fake_sd):
    """Toggling Original/Filtered must not shift time: with a model that outputs 0.5 x delayed input,
    output = 0.5*x (filtered) or x (original) at the same time positions, with no discontinuity."""
    sig, src = _sine_source(SR * 4)
    eng = AudioEngine()
    eng.start(EngineConfig(IN_DEV, OUT_DEV), DelayedIdentity(delay=1440))
    fake_sd["in"].source = src
    time.sleep(1.0)
    eng.set_bypass(True)
    time.sleep(1.0)
    eng.set_bypass(False)
    time.sleep(0.8)
    eng.stop()
    out = np.concatenate(fake_sd["out"].sink)
    nz = np.flatnonzero(np.abs(out) > 1e-4)
    out = out[nz[0]:]
    env = np.sqrt(np.convolve(out ** 2, np.ones(480) / 480, mode="same")) * np.sqrt(2)
    a, b, c = env[int(0.5 * SR)], env[int(1.5 * SR)], env[int(2.5 * SR)]
    assert abs(a - 0.15) < 0.03 and abs(b - 0.30) < 0.04 and abs(c - 0.15) < 0.03
    # phase continuity: a sine's second difference is tiny unless there is a time jump
    d2 = np.abs(np.diff(out[int(0.2 * SR):int(2.7 * SR)], 2))
    assert d2.max() < 0.02, "discontinuity detected when switching original/filtered"


@pytest.mark.parametrize("ratio", [1.003, 0.997])
def test_clock_drift_stays_bounded(fake_sd, ratio):
    """Output clock 0.3 % faster/slower than input (30x worse than real devices): the queue must stay
    bounded, no backlog drops, and the fill must settle near the prefill target."""
    _, src = _sine_source(SR * 4)
    CLOCK.out_rate_ratio = ratio
    eng = AudioEngine()
    cfg = EngineConfig(IN_DEV, OUT_DEV, prefill_ms=20)
    eng.start(cfg, Passthrough())
    fake_sd["in"].source = src
    time.sleep(3.0)                      # settle (start-up priming is trimmed first)
    slips_mid = eng.stats().slips
    window = 7.0                         # the fake clock applies the drift as one whole block every ~3.3 s
    time.sleep(window)
    s = eng.stats()
    eng.stop()
    assert s.backlog_drops == 0 and s.out_drops == 0
    assert s.out_fill_ms < 80
    corrected = s.slips - slips_mid      # samples corrected during the steady window
    expected = 0.003 * SR * window       # 0.3 % drift over the window = 1008 samples
    if ratio < 1:                        # output slower -> queue grows -> trimmed
        assert -expected * 1.6 < corrected < -expected * 0.3, corrected
    else:                                # output faster -> queue drains -> stretched
        assert expected * 0.3 < corrected < expected * 1.6, corrected
    assert s.underruns <= 5          # 0.3 % drift is ~30x a real device pair; a few underruns are tolerated


def test_stop_start_repeat_and_disconnect_event(fake_sd):
    events = []
    eng = AudioEngine(on_event=lambda k, m: events.append((k, m)))
    for _ in range(3):
        eng.start(EngineConfig(IN_DEV, OUT_DEV), Passthrough())
        time.sleep(0.3)
        eng.stop()
        assert not eng.running
    eng.start(EngineConfig(IN_DEV, OUT_DEV), Passthrough())
    time.sleep(0.3)
    fake_sd["in"].simulate_disconnect()
    time.sleep(0.3)
    assert events and events[-1][0] == "error"
    assert not eng.running
    eng.stop()


def test_soft_clip_and_resize():
    x = np.array([0.0, 0.5, 0.95, 1.5, -2.0], np.float32)
    y = eng_mod._soft_clip(x)
    assert np.all(np.abs(y) <= 1.0) and y[1] == 0.5 and y[2] < 0.95 and y[3] > 0.9
    r = eng_mod._resize_linear(np.arange(480, dtype=np.float32), 479)
    assert len(r) == 479 and r[0] == 0 and abs(r[-1] - 479) < 1e-4
