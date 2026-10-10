"""External end-to-end latency measurement (needs a second loopback path).

The in-app figure shown in the UI comes from PortAudio ADC/DAC timestamps. This tool measures
the real path with audio only:

    this tool --(chirp)--> CABLE Input  ==> app engine (CABLE Output -> model -> OUT_DEVICE)
                                           ==> OUT_DEVICE is looped back to LOOP_CAPTURE ==> this tool

Requirements: OUT_DEVICE must be a playback endpoint whose audio can be captured again, e.g.
 * a second virtual cable (VB-Cable A/B: play to "CABLE-A Input", capture "CABLE-A Output"),
 * Voicemeeter running with its "Voicemeeter Input" routed to a "Voicemeeter Out B" bus,
 * a sound card "Stereo Mix" endpoint that is enabled in Windows.
Never use the same cable the browser plays into (feedback loop).

    python tools/measure_latency_loopback.py --out "CABLE-A Input" --loop "CABLE-A Output" [--model dfn3]

Prints the measured delay between the chirp played into CABLE Input and the processed chirp
captured on the loopback, i.e. the full added latency including both device buffers and the
model's algorithmic delay (and the loopback's own latency, which this tool also measures alone
and subtracts).
"""
import argparse
import time

import numpy as np
import sounddevice as sd

from stream_voice_focus import devices as dv
from stream_voice_focus.engine import AudioEngine, EngineConfig
from stream_voice_focus.models.registry import load_model

SR = 48000


def chirp(seconds=0.5, f0=300, f1=6000, level=0.3):
    t = np.arange(int(SR * seconds)) / SR
    k = (f1 - f0) / seconds
    x = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t)) * level
    w = np.hanning(len(x))
    return (x * w).astype(np.float32)


def play_and_record(play_dev, rec_dev, signal, seconds=3.0):
    rec = []
    sig = np.concatenate([np.zeros(SR // 2, np.float32), signal, np.zeros(int(SR * seconds) - len(signal) - SR // 2, np.float32)])
    pos = [0]

    def ocb(outdata, frames, t, status):
        i = pos[0]
        chunk = sig[i:i + frames]
        outdata[:len(chunk)] = chunk[:, None]
        outdata[len(chunk):] = 0
        pos[0] += frames

    def icb(indata, frames, t, status):
        rec.append(indata[:, 0].copy())

    with sd.InputStream(device=rec_dev.index, samplerate=SR, channels=1, dtype="float32", blocksize=480,
                        callback=icb, extra_settings=dv.wasapi_settings(rec_dev.hostapi)), \
         sd.OutputStream(device=play_dev.index, samplerate=SR, channels=1, dtype="float32", blocksize=480,
                         callback=ocb, extra_settings=dv.wasapi_settings(play_dev.hostapi)):
        time.sleep(seconds + 0.2)
    return sig, np.concatenate(rec)


def delay_ms(played, recorded):
    n = min(len(played), len(recorded))
    c = np.correlate(recorded[:n], played[:n][SR // 2:SR // 2 + SR // 2], mode="valid")
    lag = int(np.argmax(np.abs(c))) - SR // 2
    return 1000.0 * lag / SR, float(np.max(np.abs(c)) / (np.linalg.norm(played[SR // 2:SR]) * np.linalg.norm(recorded[max(0, lag + SR // 2):max(0, lag + SR // 2) + SR // 2]) + 1e-9))


def pick(devs, hint, is_out):
    for d in devs:
        if hint.lower() in d.name.lower() and (d.is_output if is_out else d.is_input):
            return d
    raise SystemExit(f"device containing '{hint}' not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="engine output playback endpoint (loopback-capable)")
    ap.add_argument("--loop", required=True, help="capture endpoint that returns --out audio")
    ap.add_argument("--model", default="dfn3")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    devs = dv.list_devices(None)
    cap, play = dv.find_vb_cable(devs)
    out = pick(devs, args.out, True)
    loop = pick(devs, args.loop, False)
    if dv.is_vb_cable_playback(out.name):
        raise SystemExit("refusing: --out is the cable the source plays into (feedback loop)")
    x = chirp()
    # 1) loopback path alone (play directly to --out, capture --loop)
    base = []
    for _ in range(args.runs):
        p, r = play_and_record(out, loop, x)
        d, q = delay_ms(p, r)
        base.append(d)
        print(f"loopback alone: {d:.1f} ms (corr {q:.2f})")
    # 2) full path through the engine
    model = load_model(args.model)
    eng = AudioEngine()
    eng.start(EngineConfig(cap, out), model)
    time.sleep(1.0)
    full = []
    for _ in range(args.runs):
        p, r = play_and_record(play, loop, x)
        d, q = delay_ms(p, r)
        full.append(d)
        s = eng.stats()
        print(f"through engine: {d:.1f} ms (corr {q:.2f}); in-app estimate {s.est_total_ms:.0f} ms, timestamp measurement "
              f"{s.measured_latency_ms if s.measured_latency_ms is None else round(s.measured_latency_ms)} ms")
    eng.stop()
    b, f = float(np.median(base)), float(np.median(full))
    print(f"\nRESULT: loopback alone {b:.1f} ms, through engine {f:.1f} ms -> added latency ≈ {f - b:.1f} ms "
          f"(includes VB-Cable, capture+playback buffers, model delay {1000 * model.info.algorithmic_delay / SR:.0f} ms)")


if __name__ == "__main__":
    main()
