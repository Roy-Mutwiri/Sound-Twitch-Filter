"""End-to-end device test without touching the headphones.

Plays a test signal (file or synthetic speech-like tone) into the VB-Cable playback endpoint
("CABLE Input", exactly like the browser would), runs the engine capturing "CABLE Output",
and sends the processed audio to --out (default: the Realtek speakers) at --gain (default 0.0
= silent, so the full pipeline runs on real devices without audible output).

Usage: python tools/engine_smoke.py [--seconds 20] [--model dfn3] [--out "Speakers"] [--gain 0] [--file x.wav]
"""
import argparse
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from stream_voice_focus import devices as dv
from stream_voice_focus.engine import AudioEngine, EngineConfig
from stream_voice_focus.models.registry import load_model


def pick(devs, hint, is_out):
    for d in devs:
        if hint.lower() in d.name.lower() and (d.is_output if is_out else d.is_input):
            return d
    raise SystemExit(f"device with '{hint}' not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--model", default="dfn3")
    ap.add_argument("--out", default="Speakers (Realtek")
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--file", default=None)
    ap.add_argument("--toggle", action="store_true", help="toggle bypass/reduction/gain during the run")
    ap.add_argument("--stress", action="store_true", help="burn CPU in the worker to provoke overload recovery")
    args = ap.parse_args()

    devs = dv.list_devices()
    cap, play = dv.find_vb_cable(devs)
    if cap is None or play is None:
        raise SystemExit(dv.vb_cable_missing_message())
    out = pick(devs, args.out, True)
    print(f"source -> {play.name} | engine in <- {cap.name} | engine out -> {out.name} (gain {args.gain})")
    for issue in dv.check_routing(cap, out):
        print("  routing:", issue.level, issue.message)

    sr = 48000
    if args.file:
        sig, fsr = sf.read(args.file, dtype="float32", always_2d=True)
        sig = sig.mean(axis=1)
        if fsr != sr:
            import soxr
            sig = soxr.resample(sig, fsr, sr)
    else:
        tt = np.arange(int(sr * 4)) / sr
        f0 = 130 * (1 + 0.15 * np.sin(2 * np.pi * 0.5 * tt))
        ph = 2 * np.pi * np.cumsum(f0) / sr
        sig = (sum(np.sin(k * ph) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * tt)) ** 2 * 0.2).astype(np.float32)
    pos = [0]

    def src_cb(outdata, frames, t, status):
        i = pos[0]
        chunk = np.take(sig, np.arange(i, i + frames) % len(sig))
        outdata[:] = np.repeat(chunk[:, None], outdata.shape[1], axis=1)
        pos[0] = (i + frames) % len(sig)

    src = sd.OutputStream(device=play.index, samplerate=sr, channels=2, dtype="float32", blocksize=480,
                          callback=src_cb, extra_settings=dv.wasapi_settings(play.hostapi))

    model = load_model(args.model)
    if args.stress:
        real = model.process

        def slow(frame):
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.012:
                pass
            return real(frame)
        model.process = slow

    events = []
    eng = AudioEngine(on_event=lambda k, m: events.append((k, m)) or print("EVENT", k, m))
    cfg = EngineConfig(input_device=cap, output_device=out, output_gain=args.gain, reduction=1.0)
    src.start()
    t0 = time.time()
    eng.start(cfg, model)
    print(f"started; reported latencies in={eng._in_lat*1000:.1f}ms out={eng._out_lat*1000:.1f}ms")
    last = 0
    try:
        while time.time() - t0 < args.seconds and eng.running:
            time.sleep(1.0)
            s = eng.stats()
            if args.toggle:
                k = int(time.time() - t0)
                eng.set_bypass(k % 4 in (1, 2))
                eng.set_reduction(0.3 if k % 6 >= 3 else 1.0)
                eng.set_gain(args.gain * (0.5 if k % 2 else 1.0))
            print(f"t={time.time()-t0:5.1f}s in_rms={s.in_rms:.3f} out_rms={s.out_rms:.3f} infer={s.infer_ms_avg:.2f}/{s.infer_ms_p95:.2f}ms "
                  f"backlog={s.backlog_ms:.1f}ms fill={s.out_fill_ms:.1f}ms under={s.underruns} ovf={s.in_overflows} "
                  f"bdrop={s.backlog_drops} odrop={s.out_drops} slips={s.slips} est={s.est_total_ms:.0f}ms "
                  f"meas={'%.0f' % s.measured_latency_ms if s.measured_latency_ms else 'n/a'}ms frames={s.frames_processed}")
    finally:
        eng.stop()
        src.stop(); src.close()
    s = eng.stats()
    print("STOPPED. events:", events)
    print(f"SUMMARY frames={s.frames_processed} underruns={s.underruns} overflows={s.in_overflows} backlog_drops={s.backlog_drops} "
          f"out_drops={s.out_drops} slips={s.slips} infer_avg={s.infer_ms_avg:.2f} p95={s.infer_ms_p95:.2f} max={s.infer_ms_max:.2f} "
          f"est_total={s.est_total_ms:.0f}ms measured={s.measured_latency_ms} note={s.measured_latency_note}")


if __name__ == "__main__":
    main()
