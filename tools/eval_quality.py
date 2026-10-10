"""Objective quality evaluation of the streaming models on synthetic mixtures.

Builds test cases at 48 kHz from the licensed samples in samples/raw:
  clean        : speech only (LibriSpeech CC-BY 4.0 clip)
  music_0dB    : speech + music at 0 dB SNR (music: librosa examples, CC-BY / CC0)
  music_5dB    : speech + music at 5 dB SNR
  sfx_0dB      : speech + synthetic gunshots/explosions (noise bursts + low booms) at 0 dB
  bg_music     : music only, no speech   (how much is removed)
  bg_sfx       : effects only, no speech
  overlap_0dB  : two talkers at 0 dB (the model is expected to keep both)

Metrics (intrusive ones use the clean target speech, which we have because we mixed the audio):
  SI-SDR improvement (dB), PESQ-WB (16 kHz), STOI, residual level for no-speech cases, and
  DNSMOS P.835 (SIG/BAK/OVRL, non-intrusive, Microsoft DNS-Challenge model, MIT) when available.

Outputs loudness-matched WAVs in samples/out/<case>_<model>.wav next to <case>_input.wav for listening.
This script measures; it does not listen. See README "Quality results" for the interpretation.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import soxr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "samples", "raw")
OUT = os.path.join(ROOT, "samples", "out")
SR = 48000


def load48(name: str, seconds: float | None = None) -> np.ndarray:
    x, sr = sf.read(os.path.join(RAW, name), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality="HQ").astype(np.float32)
    if seconds:
        x = x[: int(seconds * SR)]
    return x


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix_at_snr(speech, noise, snr_db):
    noise = np.resize(noise, len(speech))
    g = rms(speech) / (rms(noise) * 10 ** (snr_db / 20))
    return speech + g * noise, g * noise


def synth_sfx(seconds: float, seed=3) -> np.ndarray:
    """Synthetic gunshots/explosions: broadband bursts with fast decay plus low-frequency booms."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = np.zeros(n, np.float32)
    t = 0.3
    while t < seconds - 0.5:
        i = int(t * SR)
        L = int(SR * rng.uniform(0.08, 0.35))
        env = np.exp(-np.arange(L) / (L / 6))
        burst = rng.standard_normal(L) * env
        boom_f = rng.uniform(40, 90)
        boom = np.sin(2 * np.pi * boom_f * np.arange(L) / SR) * np.exp(-np.arange(L) / (L / 3)) * 0.8
        y[i:i + L] += (burst * rng.uniform(0.4, 1.0) + boom).astype(np.float32)[: len(y[i:i + L])]
        t += rng.uniform(0.35, 1.2)
    return y / (np.max(np.abs(y)) + 1e-9)


def si_sdr(est, ref):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    tgt = a * ref
    return 10 * np.log10(np.dot(tgt, tgt) / (np.dot(est - tgt, est - tgt) + 1e-12) + 1e-12)


def pesq_stoi(est, ref):
    from pesq import pesq
    from pystoi import stoi
    r16 = soxr.resample(ref, SR, 16000)
    e16 = soxr.resample(est, SR, 16000)
    try:
        p = float(pesq(16000, r16, e16, "wb"))
    except Exception:
        p = float("nan")
    s = float(stoi(r16, e16, 16000, extended=False))
    return p, s


class DNSMOS:
    def __init__(self):
        import onnxruntime as ort
        p = os.path.join(ROOT, "models_cache", "dnsmos", "sig_bak_ovr.onnx")
        self.ok = os.path.exists(p)
        if self.ok:
            self.sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])

    def __call__(self, x48):
        """DNSMOS P.835 (sig_bak_ovr.onnx takes 9.01 s of raw 16 kHz audio, shape [1, 144160])."""
        if not self.ok:
            return (float("nan"),) * 3
        x = soxr.resample(x48, SR, 16000).astype(np.float32)
        seg = int(9.01 * 16000)
        hop = 16000
        if len(x) < seg:
            x = np.pad(x, (0, seg - len(x)))
        name = self.sess.get_inputs()[0].name
        res = []
        for s in range(0, len(x) - seg + 1, hop):
            out = self.sess.run(None, {name: x[s:s + seg][None, :]})[0][0]
            res.append(out)
        r = np.mean(res, axis=0)
        # polynomial mapping from DNSMOS dnsmos_local.py (personalized=False)
        sig_poly = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        bak_poly = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        ovr_poly = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        return float(sig_poly(r[0])), float(bak_poly(r[1])), float(ovr_poly(r[2]))


def build_cases():
    sp1 = load48("speech_libri1.wav", 12)
    sp2 = load48("speech_libri2.wav", 12)
    sp1 = sp1 / (np.max(np.abs(sp1)) + 1e-9) * 0.5
    sp2 = sp2 / (np.max(np.abs(sp2)) + 1e-9) * 0.5
    mus = load48("music_choice.wav")           # drum & bass (CC-BY, Admiral Bob)
    mus2 = load48("music_nutcracker.wav")      # orchestral (Kevin MacLeod, CC-BY)
    sfx = synth_sfx(12.0)
    cases = {}
    m, n = mix_at_snr(sp1, mus, 0); cases["music_dnb_0dB"] = (m, sp1, True)
    m, n = mix_at_snr(sp1, mus2, 0); cases["music_orch_0dB"] = (m, sp1, True)
    m, n = mix_at_snr(sp1, mus, 5); cases["music_dnb_5dB"] = (m, sp1, True)
    m, n = mix_at_snr(sp1, sfx, 0); cases["sfx_0dB"] = (m, sp1, True)
    m, n = mix_at_snr(sp1, sfx, -5); cases["sfx_-5dB"] = (m, sp1, True)
    cases["clean"] = (sp1, sp1, True)
    cases["bg_music_only"] = (np.resize(mus, len(sp1)) * (rms(sp1) / rms(mus)), None, False)
    cases["bg_sfx_only"] = (sfx * (rms(sp1) / rms(sfx)), None, False)
    m, n = mix_at_snr(sp1, sp2, 0); cases["overlap_2talkers_0dB"] = (m, sp1 + n, True)
    return cases


def main():
    from stream_voice_focus.file_mode import process_array
    from stream_voice_focus.models.registry import load_model
    models = sys.argv[1:] or ["dfn3", "dpdfnet2_48k", "dpdfnet8_48k", "rnnoise"]
    os.makedirs(OUT, exist_ok=True)
    cases = build_cases()
    dns = DNSMOS()
    results = {}
    for name, (mix, ref, has_speech) in cases.items():
        sf.write(os.path.join(OUT, f"{name}_input.wav"), mix / max(1.0, np.max(np.abs(mix))), SR)
        row = {"input": {}}
        if has_speech:
            p, s = pesq_stoi(mix, ref)
            row["input"] = {"si_sdr": si_sdr(mix, ref), "pesq": p, "stoi": s}
        d = dns(mix)
        row["input"].update({"dnsmos_sig": d[0], "dnsmos_bak": d[1], "dnsmos_ovr": d[2], "level_dbfs": 20 * np.log10(rms(mix))})
        results[name] = row
    for mk in models:
        t = time.time()
        model = load_model(mk)
        print(f"== {mk} loaded in {time.time()-t:.1f}s", flush=True)
        for name, (mix, ref, has_speech) in cases.items():
            y, st = process_array(model, mix, 1.0)
            r = {"rtf": st["realtime_factor"], "ms_per_frame": st["mean_ms"]}
            if has_speech:
                p, s = pesq_stoi(y, ref)
                sd_in = results[name]["input"]["si_sdr"]
                r.update({"si_sdr": si_sdr(y, ref), "pesq": p, "stoi": s,
                          "si_sdr_impr": (si_sdr(y, ref) - sd_in) if sd_in < 60 else float("nan")})
            d = dns(y)
            r.update({"dnsmos_sig": d[0], "dnsmos_bak": d[1], "dnsmos_ovr": d[2]})
            r["residual_db"] = 20 * np.log10(rms(y) / rms(mix))   # output level relative to input
            results[name][mk] = r
            # loudness-matched listening copy (same RMS as input, peak-safe)
            yl = y * (rms(mix) / rms(y)) if rms(y) > 1e-6 else y
            yl = yl / max(1.0, np.max(np.abs(yl)))
            sf.write(os.path.join(OUT, f"{name}_{mk}.wav"), yl, SR)
            print(f"  {name:22s} " + " ".join(f"{k}={v:.2f}" for k, v in r.items()), flush=True)
        model.close()
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    # summary table
    print("\n=== SUMMARY (speech cases: SI-SDR improvement dB / PESQ / STOI / DNSMOS BAK; no-speech: residual dB) ===")
    for name, row in results.items():
        line = f"{name:22s} in: "
        inp = row["input"]
        if "pesq" in inp:
            line += f"PESQ {inp['pesq']:.2f} STOI {inp['stoi']:.2f} SIG{inp['dnsmos_sig']:.2f} B{inp['dnsmos_bak']:.2f} O{inp['dnsmos_ovr']:.2f} | "
        else:
            line += f"BAK {inp['dnsmos_bak']:.2f} | "
        for mk in models:
            r = row.get(mk, {})
            if "pesq" in r:
                sdr = f"+{r['si_sdr_impr']:.1f}dB" if r['si_sdr_impr'] == r['si_sdr_impr'] else f"SDR {r['si_sdr']:.1f}dB"
                line += f"{mk}: {sdr} P{r['pesq']:.2f} S{r['stoi']:.2f} SIG{r['dnsmos_sig']:.2f} B{r['dnsmos_bak']:.2f} O{r['dnsmos_ovr']:.2f} | "
            else:
                line += f"{mk}: resid {r.get('residual_db', float('nan')):.1f}dB | "
        print(line)


if __name__ == "__main__":
    main()
