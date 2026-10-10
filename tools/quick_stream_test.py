import time, sys, numpy as np, torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
from stream_voice_focus.compat import install_torchaudio_shim; install_torchaudio_shim()
t=time.time()
from stream_voice_focus.models._torchdf_minimal import TorchDFMinimalPipeline
pipe = TorchDFMinimalPipeline()
print("load secs", round(time.time()-t,1), "hop", pipe.hop_size, "fft", pipe.fft_size, "sr", pipe.sample_rate)
m = pipe.torch_streaming_model
print("lookahead", m.lookahead, "df_order", m.df_order)
# correctness vs offline enhance
from df.enhance import init_df, enhance
model, state, _ = init_df(log_level="WARNING")
sr=48000; rng=np.random.default_rng(0)
tt=np.arange(sr*2)/sr
sig = 0.3*np.sin(2*np.pi*220*tt)*(np.sin(2*np.pi*3*tt)>0) + 0.05*rng.standard_normal(sr*2)
x = torch.from_numpy(sig.astype(np.float32))[None]
off = enhance(model, state, x).squeeze(0).numpy()
st = pipe(x, sr).squeeze(0).detach().numpy()
n=min(len(off),len(st)); print("offline vs streaming max abs diff", np.abs(off[:n]-st[:n]).max(), "rms", np.sqrt(np.mean((off[:n]-st[:n])**2)))
# per-frame benchmark
states=[s.clone() for s in pipe.states]
frame=torch.zeros(480)
with torch.no_grad():
    for _ in range(20): out,*states = m(frame,*states)
    ts=[]
    for i in range(300):
        t0=time.perf_counter(); out,*states = m(frame,*states); ts.append(time.perf_counter()-t0)
ts=np.array(ts)*1000
print(f"torch per-frame ms: mean {ts.mean():.2f} p50 {np.median(ts):.2f} p95 {np.percentile(ts,95):.2f} max {ts.max():.2f}  (frame = 10 ms)")
