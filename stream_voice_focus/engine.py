"""Real-time audio engine.

Threads
-------
* PortAudio input callback  : copies the captured block into `in_ring` (mono downmix, level meter).
* Worker thread             : pops model-sized frames from `in_ring`, runs inference, keeps the
                              time-aligned "original" path, applies bypass/gain/limiter, pushes
                              to `out_ring`. All model work happens here.
* PortAudio output callback : copies from `out_ring` to the device; fills silence (with a short
                              fade) on underrun. Never blocks, never touches the model.

Bounded buffers: both rings have fixed capacity. If inference falls behind, the worker drops the
oldest backlog (counted as `backlog_drops`) so delay can never grow without bound. If the
playback clock runs slower than the capture clock the output ring fills; the worker then trims
one sample per frame (`slips`, inaudible 0.2 % resample) and, past a hard limit, skips a frame.
If the playback clock is faster the ring drains and the output callback inserts faded silence
(`underruns`) and the worker stretches by one sample per frame until the fill recovers.

Latency accounting (all reported separately, see EngineStats):
* algorithmic delay of the model (fixed, by design),
* inference execution time (measured per frame),
* capture/playback buffering: PortAudio reported stream latencies + measured ring fills,
* estimated total added latency = sum of the above,
* measured end-to-end: PortAudio ADC timestamp of a captured sample vs DAC timestamp of the
  output sample that carries it (plus the algorithmic delay). Reported only if the host API
  provides valid timestamps, otherwise "unavailable".
"""
from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from .devices import DeviceInfo, wasapi_settings
from .models.base import StreamingEnhancer
from .resampler import StreamResampler
from .ringbuffer import RingBuffer


@dataclass
class EngineConfig:
    input_device: DeviceInfo
    output_device: DeviceInfo
    sample_rate: int = 48000
    block_size: int = 480          # device callback block (10 ms at 48 kHz)
    reduction: float = 1.0
    output_gain: float = 1.0
    bypass: bool = False           # True = play the latency-matched original
    prefill_ms: float = 20.0       # output ring target fill (absorbs jitter)
    max_backlog_ms: float = 80.0   # input backlog before dropping (inference overload)
    latency: str = "low"           # PortAudio latency hint


@dataclass
class EngineStats:
    running: bool = False
    model_key: str = ""
    in_peak: float = 0.0
    in_rms: float = 0.0
    out_peak: float = 0.0
    out_rms: float = 0.0
    infer_ms_avg: float = 0.0
    infer_ms_p95: float = 0.0
    infer_ms_max: float = 0.0
    frame_ms: float = 10.0
    cpu_load: float = 0.0            # infer_ms_avg / frame_ms
    backlog_ms: float = 0.0          # input waiting for inference
    out_fill_ms: float = 0.0         # processed audio waiting for playback
    underruns: int = 0
    in_overflows: int = 0
    in_drops: int = 0
    backlog_drops: int = 0
    out_drops: int = 0
    slips: int = 0
    frames_processed: int = 0
    algo_delay_ms: float = 0.0
    input_latency_ms: float = 0.0    # PortAudio reported
    output_latency_ms: float = 0.0   # PortAudio reported
    est_total_ms: float = 0.0
    measured_latency_ms: Optional[float] = None
    measured_latency_note: str = ""
    uptime_s: float = 0.0
    last_error: str = ""


def _soft_clip(x: np.ndarray, t: float = 0.9) -> np.ndarray:
    """Transparent below `t`, smooth tanh knee above; never exceeds 1.0."""
    a = np.abs(x)
    over = a > t
    if not over.any():
        return x
    y = x.copy()
    y[over] = np.sign(x[over]) * (t + (1.0 - t) * np.tanh((a[over] - t) / (1.0 - t)))
    return y


def _resize_linear(x: np.ndarray, n: int) -> np.ndarray:
    if n == len(x):
        return x
    return np.interp(np.linspace(0.0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


class DelayLine:
    def __init__(self, delay: int):
        self.delay = int(delay)
        self._buf = np.zeros(self.delay, np.float32)

    def push(self, x: np.ndarray) -> np.ndarray:
        if self.delay == 0:
            return x
        n = len(x)
        if n >= self.delay:
            out = np.concatenate([self._buf, x[:n - self.delay]])
            self._buf = x[n - self.delay:].copy()
        else:
            out = self._buf[:n].copy()
            self._buf = np.concatenate([self._buf[n:], x])
        return out

    def reset(self):
        self._buf[:] = 0


class AudioEngine:
    def __init__(self, on_event: Optional[Callable[[str, str], None]] = None):
        self.on_event = on_event or (lambda kind, msg: None)
        self._lock = threading.Lock()
        self._running = False
        self._stop_evt = threading.Event()
        self._in_event = threading.Event()
        self.model: Optional[StreamingEnhancer] = None
        self.cfg: Optional[EngineConfig] = None
        self._in_stream = None
        self._out_stream = None
        self._worker: Optional[threading.Thread] = None
        self._reset_counters()
        # live controls (written by UI thread, read by worker)
        self._gain_target = 1.0
        self._bypass = False
        self._reduction_target = 1.0
        self._fade_out = False

    # ------------------------------------------------------------------ controls
    def set_gain(self, g: float):
        self._gain_target = float(max(0.0, min(2.0, g)))

    def set_bypass(self, b: bool):
        self._bypass = bool(b)

    def set_reduction(self, r: float):
        self._reduction_target = float(max(0.0, min(1.0, r)))

    @property
    def running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ lifecycle
    def _reset_counters(self):
        self.in_peak = self.in_rms = self.out_peak = self.out_rms = 0.0
        self.underruns = self.in_overflows = self.in_drops = self.backlog_drops = self.out_drops = 0
        self.slips = 0
        self.frames_processed = 0
        self._infer_hist = collections.deque(maxlen=300)
        self._infer_avg = 0.0
        self._infer_max = 0.0
        self._backlog_avg = 0.0
        self._fill_avg = 0.0
        self._in_stamps = collections.deque(maxlen=128)   # (start_idx, n, adc_time)
        self._tags = collections.deque(maxlen=256)        # (out_start_idx, in_start_idx)
        self._meas_latency = None
        self._meas_note = "no valid device timestamps yet"
        self._timestamps_seen = False
        self._t_start = 0.0
        self.last_error = ""
        self._out_abs_read = 0

    def start(self, cfg: EngineConfig, model: StreamingEnhancer) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("engine already running")
            self.cfg = cfg
            self.model = model
            self._reset_counters()
            sr = cfg.sample_rate
            self._gain_target = cfg.output_gain
            self._gain_cur = cfg.output_gain
            self._bypass = cfg.bypass
            self._mix = 0.0 if cfg.bypass else 1.0
            self._reduction_target = cfg.reduction
            self._reduction_cur = cfg.reduction
            model.set_reduction(cfg.reduction)
            self._fade_out = False
            self._discontinuity = False
            self._need_fadein = True

            in_ch = max(1, min(2, cfg.input_device.max_input_channels))
            out_ch = max(1, min(2, cfg.output_device.max_output_channels))
            self._out_ch = out_ch
            self._in_ring = RingBuffer(sr, 1)                  # 1 s hard cap
            self._out_ring = RingBuffer(sr, out_ch)
            self._out_tmp = np.zeros((4096, out_ch), np.float32)
            self._prefill = int(sr * cfg.prefill_ms / 1000)
            self._max_backlog = int(sr * cfg.max_backlog_ms / 1000)
            # drift controller thresholds on the *averaged* output fill
            self._hi_wm = self._prefill + cfg.block_size
            self._lo_wm = max(cfg.block_size // 2, self._prefill - cfg.block_size // 2)
            self._hard_hi = self._prefill + max(6 * cfg.block_size, int(sr * 0.08))

            # model-rate adaptation
            self._rs_in = StreamResampler(sr, model.info.sample_rate)
            self._rs_out = StreamResampler(model.info.sample_rate, sr)
            self._delay = DelayLine(int(round(model.info.algorithmic_delay * sr / model.info.sample_rate)))

            try:
                self._out_stream = sd.OutputStream(
                    device=cfg.output_device.index, samplerate=sr, channels=out_ch, dtype="float32",
                    blocksize=cfg.block_size, latency=cfg.latency, callback=self._out_cb,
                    finished_callback=self._on_out_finished,
                    extra_settings=wasapi_settings(cfg.output_device.hostapi))
                self._in_stream = sd.InputStream(
                    device=cfg.input_device.index, samplerate=sr, channels=in_ch, dtype="float32",
                    blocksize=cfg.block_size, latency=cfg.latency, callback=self._in_cb,
                    finished_callback=self._on_in_finished,
                    extra_settings=wasapi_settings(cfg.input_device.hostapi))
            except Exception as e:
                self._close_streams()
                raise RuntimeError(f"Could not open audio device: {e}") from e

            self._in_lat = float(self._in_stream.latency)
            self._out_lat = float(self._out_stream.latency)
            # prime output with silence so the first callbacks have data. WASAPI pulls about two
            # extra blocks when a stream starts, so prime target + 2 blocks to land on target.
            self._out_ring.push(np.zeros((self._prefill + 2 * cfg.block_size, out_ch), np.float32))
            self._stop_evt.clear()
            self._in_event.clear()
            self._running = True
            self._t_start = time.time()
            self._worker = threading.Thread(target=self._worker_loop, name="svf-worker", daemon=True)
            self._worker.start()
            try:
                # start capture first; begin playback once the first processed frame is queued so
                # the output ring starts at its target fill instead of ramping up through slips
                self._in_stream.start()
                deadline = time.time() + 0.5
                while self._out_ring.available < self._prefill + cfg.block_size and time.time() < deadline:
                    time.sleep(0.002)
                self._out_stream.start()
            except Exception as e:
                self._running = False
                self._stop_evt.set()
                self._close_streams()
                raise RuntimeError(f"Could not start audio stream: {e}") from e

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            # short fade-out in the output callback avoids a pop
            self._fade_out = True
            time.sleep(0.03)
            self._stop_evt.set()
            self._in_event.set()
            try:
                if self._in_stream is not None:
                    self._in_stream.stop()
            except Exception:
                pass
            if self._worker is not None:
                self._worker.join(timeout=2.0)
            try:
                if self._out_stream is not None:
                    self._out_stream.stop()
            except Exception:
                pass
            self._close_streams()
            if self.model is not None:
                try:
                    self.model.reset()
                except Exception:
                    pass

    def _close_streams(self):
        for s in (self._in_stream, self._out_stream):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
        self._in_stream = None
        self._out_stream = None

    def _on_in_finished(self):
        if self._running:
            self._fail("Input device stopped (disconnected or reconfigured).")

    def _on_out_finished(self):
        if self._running:
            self._fail("Output device stopped (disconnected or reconfigured).")

    def _fail(self, msg: str):
        self.last_error = msg
        self._running = False
        self._stop_evt.set()
        self._in_event.set()
        threading.Thread(target=self._cleanup_after_fail, daemon=True).start()
        self.on_event("error", msg)

    def _cleanup_after_fail(self):
        time.sleep(0.05)
        with self._lock:
            try:
                for s in (self._in_stream, self._out_stream):
                    if s is not None:
                        try:
                            s.abort()
                        except Exception:
                            pass
            finally:
                self._close_streams()

    # ------------------------------------------------------------------ callbacks
    def _in_cb(self, indata, frames, t, status):
        if status:
            if status.input_overflow:
                self.in_overflows += 1
        mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
        if frames:
            self.in_peak = float(np.max(np.abs(mono)))
            self.in_rms = float(math.sqrt(float(np.mean(mono * mono))))
        start = self._in_ring.total_written
        dropped = self._in_ring.push(mono)
        if dropped:
            self.in_drops += dropped
        adc = getattr(t, "inputBufferAdcTime", 0.0)
        if adc:
            self._in_stamps.append((start, frames - dropped, adc))
        self._in_event.set()

    def _out_cb(self, outdata, frames, t, status):
        if status and status.output_underflow:
            self.underruns += 1
        ring = self._out_ring
        avail = ring.available
        n = min(avail, frames)
        out_idx = ring.total_read
        if n > 0:
            chunk = self._out_tmp[:n]
            ring.pop(n, out=chunk)
            if self._need_fadein:
                k = min(n, 64)
                chunk[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
                self._need_fadein = False
            if n < frames:
                k = min(n, 64)
                chunk[n - k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)[:, None]
            if self._fade_out:
                chunk *= np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
            outdata[:n] = chunk
        if n < frames:
            outdata[n:] = 0.0
            if self._running and not self._fade_out:
                self.underruns += 1
                self._need_fadein = True
        if self._fade_out:
            self._fade_out = False
            self._need_fadein = True
        if frames:
            self.out_peak = float(np.max(np.abs(outdata)))
            self.out_rms = float(math.sqrt(float(np.mean(outdata * outdata))))
        # measured latency via device timestamps
        dac = getattr(t, "outputBufferDacTime", 0.0)
        if dac and n > 0:
            self._timestamps_seen = True
            self._measure_latency(out_idx, dac)

    def _measure_latency(self, out_idx: int, dac: float):
        try:
            tags = tuple(self._tags)
            stamps = tuple(self._in_stamps)
        except RuntimeError:
            return
        in_idx = None
        for o_start, i_start in reversed(tags):
            if o_start <= out_idx:
                in_idx = i_start + (out_idx - o_start)
                break
        if in_idx is None:
            return
        for s_start, s_n, adc in reversed(stamps):
            if s_start <= in_idx < s_start + s_n:
                sr = self.cfg.sample_rate
                adc_t = adc + (in_idx - s_start) / sr
                lat = (dac - adc_t) + self._delay.delay / sr
                if 0 < lat < 2.0:
                    lat_ms = lat * 1000.0
                    self._meas_latency = lat_ms if self._meas_latency is None else 0.9 * self._meas_latency + 0.1 * lat_ms
                    self._meas_note = "ADC->DAC device timestamps (PortAudio), incl. algorithmic delay"
                return

    # ------------------------------------------------------------------ worker
    def _worker_loop(self):
        model = self.model
        cfg = self.cfg
        sr = cfg.sample_rate
        hop_m = model.info.frame_size                       # model frame at model rate
        hop_e = int(round(hop_m * sr / model.info.sample_rate))  # engine samples per model frame
        in_ring, out_ring = self._in_ring, self._out_ring
        rs_in, rs_out, delay = self._rs_in, self._rs_out, self._delay
        out_ch = self._out_ch
        ramp = np.linspace(0.0, 1.0, hop_e, dtype=np.float32)
        fade_n = min(96, hop_e)
        fade_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        loops = 0
        try:
            while not self._stop_evt.is_set():
                if in_ring.available < hop_e:
                    self._in_event.wait(0.02)
                    self._in_event.clear()
                    loops += 1
                    if loops % 25 == 0 and self._running:
                        s_in, s_out = self._in_stream, self._out_stream
                        if (s_in is not None and not s_in.active) or (s_out is not None and not s_out.active):
                            self._fail("An audio stream stopped unexpectedly (device removed or changed).")
                            return
                    continue
                backlog = in_ring.available
                if backlog > self._max_backlog:
                    # inference cannot keep up (or input burst): drop oldest so delay stays bounded
                    drop = backlog - 2 * hop_e
                    in_ring.skip(drop)
                    self.backlog_drops += drop
                    self._discontinuity = True
                    backlog = in_ring.available
                self._backlog_avg = 0.9 * self._backlog_avg + 0.1 * backlog

                in_start = in_ring.total_read
                x = in_ring.pop(hop_e)[:, 0]

                # smooth reduction changes
                if abs(self._reduction_cur - self._reduction_target) > 1e-4:
                    self._reduction_cur += np.clip(self._reduction_target - self._reduction_cur, -0.05, 0.05)
                    model.set_reduction(self._reduction_cur)

                # inference (with optional model-rate resampling)
                t0 = time.perf_counter()
                if rs_in.identity:
                    y = model.process(x)
                else:
                    rs_in.push(x)
                    while True:
                        f = rs_in.pop(hop_m)
                        if f is None:
                            break
                        rs_out.push(model.process(f))
                    y = rs_out.pop(hop_e)
                    if y is None:
                        y = np.zeros(hop_e, np.float32)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                self._infer_hist.append(infer_ms)
                self._infer_avg = 0.95 * self._infer_avg + 0.05 * infer_ms if self._infer_avg else infer_ms
                self._infer_max = max(self._infer_max, infer_ms)

                # latency-matched original path and original/filtered crossfade
                orig = delay.push(x)
                target_mix = 0.0 if self._bypass else 1.0
                if target_mix != self._mix:
                    mix = self._mix + (target_mix - self._mix) * ramp
                    self._mix = target_mix
                    out = orig + mix * (y - orig)
                else:
                    out = y if target_mix == 1.0 else orig

                if self._discontinuity:
                    out = out.copy()
                    out[:fade_n] *= fade_in
                    self._discontinuity = False

                # smoothed gain
                if self._gain_cur != self._gain_target:
                    g = self._gain_cur + (self._gain_target - self._gain_cur) * ramp
                    self._gain_cur = self._gain_target
                    out = out * g
                elif self._gain_cur != 1.0:
                    out = out * self._gain_cur
                out = _soft_clip(np.asarray(out, dtype=np.float32))

                # clock drift handling via gentle slips
                fill = out_ring.available
                self._fill_avg = fill if self.frames_processed < 5 else 0.9 * self._fill_avg + 0.1 * fill
                if fill > self._hard_hi:
                    # far too much queued (e.g. after a stall): skip this frame entirely
                    self.slips -= hop_e
                    self._discontinuity = True
                    self.frames_processed += 1
                    continue
                if 5 < self.frames_processed < 100 and fill - self._prefill >= hop_e:
                    # start-up: the host pulled fewer blocks than we primed for; drop the excess in
                    # whole frames now (nothing meaningful is playing yet) instead of trimming slowly.
                    # Uses the instantaneous fill, and re-seeds the average, to avoid over-correcting.
                    self.slips -= hop_e
                    self._discontinuity = True
                    self._fill_avg = fill - hop_e
                    self.frames_processed += 1
                    continue
                dev = self._fill_avg - self._prefill
                if dev > hop_e // 2:
                    k = int(min(3, 1 + dev // hop_e))          # 1 sample/frame normally; up to 3 (0.6 %) when far off
                    out = _resize_linear(out, hop_e - k)
                    self.slips -= k
                elif dev < -hop_e // 2 and self.frames_processed > 20:
                    k = int(min(3, 1 + (-dev) // hop_e))
                    out = _resize_linear(out, hop_e + k)
                    self.slips += k

                stereo = np.repeat(out[:, None], out_ch, axis=1) if out_ch > 1 else out[:, None]
                out_start = out_ring.total_written
                dropped = out_ring.push(stereo)
                if dropped:
                    self.out_drops += dropped
                self._tags.append((out_start, in_start))
                self.frames_processed += 1
        except Exception as e:  # never let the worker die silently
            self._fail(f"Processing error: {e!r}")

    # ------------------------------------------------------------------ stats
    def stats(self) -> EngineStats:
        s = EngineStats()
        s.running = self._running
        s.model_key = self.model.info.key if self.model else ""
        s.in_peak, s.in_rms, s.out_peak, s.out_rms = self.in_peak, self.in_rms, self.out_peak, self.out_rms
        hist = list(self._infer_hist)
        s.infer_ms_avg = self._infer_avg
        s.infer_ms_p95 = float(np.percentile(hist, 95)) if hist else 0.0
        s.infer_ms_max = self._infer_max
        if self.model and self.cfg:
            sr = self.cfg.sample_rate
            s.frame_ms = 1000.0 * self.model.info.frame_size / self.model.info.sample_rate
            s.cpu_load = s.infer_ms_avg / s.frame_ms if s.frame_ms else 0.0
            s.backlog_ms = 1000.0 * self._backlog_avg / sr
            s.out_fill_ms = 1000.0 * self._fill_avg / sr
            s.algo_delay_ms = 1000.0 * self._delay.delay / sr
            s.input_latency_ms = 1000.0 * getattr(self, "_in_lat", 0.0)
            s.output_latency_ms = 1000.0 * getattr(self, "_out_lat", 0.0)
            s.est_total_ms = (s.input_latency_ms + s.backlog_ms + s.infer_ms_avg + s.algo_delay_ms
                              + s.out_fill_ms + s.output_latency_ms)
        s.underruns, s.in_overflows, s.in_drops = self.underruns, self.in_overflows, self.in_drops
        s.backlog_drops, s.out_drops, s.slips = self.backlog_drops, self.out_drops, self.slips
        s.frames_processed = self.frames_processed
        s.measured_latency_ms = self._meas_latency
        s.measured_latency_note = self._meas_note if self._timestamps_seen else "device timestamps not provided by host API"
        s.uptime_s = (time.time() - self._t_start) if self._running else 0.0
        s.last_error = self.last_error
        return s
