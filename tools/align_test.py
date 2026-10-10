import time, numpy as np, torch
torch.set_num_threads(1)
from stream_voice_focus.compat import install_torchaudio_shim; install_torchaudio_shim()
from stream_voice_focus.models._torchdf_minimal import TorchDFMinimalPipeline
from df.enhance import init_df, enhance
pipe = TorchDFMinimalPipeline(); m = pipe.torch_streaming_model
model, state, _ = init_df(log_level="WARNING")
sr=48000; rng=np.random.default_rng(0); tt=np.arange(sr*2)/sr
# speech-like: harmonic buzz with syllable envelope
f0 = 120*(1+0.2*np.sin(2*np.pi*0.7*tt)); ph = 2*np.pi*np.cumsum(f0)/sr
sig = sum(np.sin(k*ph)/k for k in range(1,15)) * (0.5+0.5*np.sin(2*np.pi*4*tt))**2 * 0.2
sig = sig + 0.01*rng.standard_normal(len(sig))
x = torch.from_numpy(sig.astype(np.float32))[None]
off = enhance(model, state, x).squeeze(0).numpy()
# streaming raw (no trimming): feed frames directly
states=[s.clone() for s in pipe.states]; outs=[]
xs = np.concatenate([sig.astype(np.float32), np.zeros(4800,np.float32)])
with torch.no_grad():
    for i in range(0, len(xs)-480+1, 480):
        o,*states = m(torch.from_numpy(xs[i:i+480]),*states); outs.append(o.numpy())
st = np.concatenate(outs)
def best_lag(a,b,maxlag=4800):
    best=None
    for lag in range(0,maxlag+1,1):
        n=min(len(a),len(b)-lag); e=np.sqrt(np.mean((a[:n]-b[lag:lag+n])**2))
        if best is None or e<best[1]: best=(lag,e)
    return best
print("input->streaming output best lag (samples, rms err):", best_lag(sig, st))
print("offline->streaming output best lag:", best_lag(off, st))
print("input->offline best lag:", best_lag(sig, off))
# impulse response delay
states=[s.clone() for s in pipe.states]; outs=[]
imp = np.zeros(48000, np.float32); imp[24000]=0.5
with torch.no_grad():
    for i in range(0, len(imp), 480):
        o,*states = m(torch.from_numpy(imp[i:i+480]),*states); outs.append(o.numpy())
y=np.concatenate(outs); print("impulse in at 24000, out peak at", int(np.argmax(np.abs(y))), "-> delay samples", int(np.argmax(np.abs(y)))-24000, "=", (int(np.argmax(np.abs(y)))-24000)/48, "ms")
