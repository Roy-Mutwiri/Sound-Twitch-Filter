"""Export the streaming DeepFilterNet3 model to ONNX and benchmark ONNX Runtime vs torch."""
import time, sys, os, numpy as np, torch
torch.set_num_threads(1)
from stream_voice_focus.compat import install_torchaudio_shim; install_torchaudio_shim()
from stream_voice_focus.models._torchdf_minimal import TorchDFMinimalPipeline
out_path = sys.argv[1] if len(sys.argv) > 1 else "models_cache/dfn3_stream.onnx"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
pipe = TorchDFMinimalPipeline(); m = pipe.torch_streaming_model.eval()
states = [s.clone() for s in pipe.states]
frame = torch.zeros(480)
with torch.no_grad():
    torch.onnx.export(m, (frame, *states), out_path, input_names=pipe.input_names, output_names=pipe.output_names,
                      opset_version=17, dynamo=False, do_constant_folding=True)
print("exported", out_path, os.path.getsize(out_path)/1e6, "MB")
import onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(out_path, so, providers=["CPUExecutionProvider"])
names=[i.name for i in sess.get_inputs()]
st=[s.numpy() for s in states]
# correctness vs torch
rng=np.random.default_rng(1); x=(0.1*rng.standard_normal(48000)).astype(np.float32)
tst=[s.clone() for s in states]; ost=[s.copy() for s in st]; maxd=0
with torch.no_grad():
    for i in range(0,48000,480):
        f=x[i:i+480]
        to,*tst = m(torch.from_numpy(f),*tst)
        res = sess.run(None, dict(zip(names,[f,*ost]))); oo, ost = res[0], res[1:]
        maxd=max(maxd, float(np.abs(to.numpy()-oo).max()))
print("torch vs ort max abs diff over 1s:", maxd)
ts=[]
for i in range(300):
    t0=time.perf_counter(); res=sess.run(None, dict(zip(names,[frame.numpy(),*ost]))); ost=res[1:]; ts.append(time.perf_counter()-t0)
ts=np.array(ts)*1000
print(f"ORT per-frame ms: mean {ts.mean():.2f} p50 {np.median(ts):.2f} p95 {np.percentile(ts,95):.2f} max {ts.max():.2f}")
