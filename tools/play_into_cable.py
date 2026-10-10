"""Loop a WAV file into the VB-Cable playback endpoint (what the browser does) for N seconds."""
import sys, time, numpy as np, sounddevice as sd, soundfile as sf
from stream_voice_focus import devices as dv
path, seconds = sys.argv[1], float(sys.argv[2])
devs = dv.list_devices(); cap, play = dv.find_vb_cable(devs)
sig, sr = sf.read(path, dtype="float32", always_2d=True); sig = sig.mean(axis=1)
if sr != 48000:
    import soxr; sig = soxr.resample(sig, sr, 48000)
pos = [0]
def cb(out, frames, t, status):
    i = pos[0]; out[:] = np.repeat(np.take(sig, np.arange(i, i + frames) % len(sig))[:, None], out.shape[1], axis=1); pos[0] = (i + frames) % len(sig)
with sd.OutputStream(device=play.index, samplerate=48000, channels=2, dtype="float32", blocksize=480, callback=cb,
                     extra_settings=dv.wasapi_settings(play.hostapi)):
    time.sleep(seconds)
