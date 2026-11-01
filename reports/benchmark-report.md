# Benchmark report

TinyLlama, Bark, TinySD, and Shap-E completed real CUDA inference. Initial load times may include checkpoint transfer; TinySD's existing partial cache was resumed in place. Stable Video Diffusion remains blocked pending approval for a memory-sensitive inference attempt.

| Model | Load time | Inference time | Device | Peak VRAM | Output | Status |
|---|---:|---:|---|---:|---|---|
| TinyLlama | 2,272.131 s* | 13.468 s | CUDA | 2,219,912,704 B | Two non-empty text responses; see absolute paths below | PASS |
| Bark | 11.338 s (cached) | 17.141 s | CUDA | 984,007,168 B | Valid WAV, 24 kHz, 4.667 s; warm WAV also validated | PASS |
| TinySD | 374.254 s* | 5.931 s | CUDA | 2,007,410,176 B | Two valid 512x512 PNGs; warm request validated | PASS |
| Shap-E | 1,844.395 s* | 27.118 s | CUDA | 2,369,751,040 B | Valid OBJ; warm OBJ validated | PASS |
| Stable Video Diffusion | N/A | N/A | N/A | N/A | Awaiting approval for optimized inference; snapshot incomplete, no MP4 | BLOCKED |

*TinyLlama load time includes the interrupted/retried 2,200,119,864-byte checkpoint transfer and model initialization. First request end-to-end: 2,288.040 s. The subsequent warm request completed in 13.346 s (13.342 s inference, `load_seconds=0`). Peak VRAM is PyTorch's measured `torch.cuda.max_memory_allocated`, not an estimate.

Bark's initial download run measured 857.262 s load and 847.411 s inference, with a 984,006,656-byte peak; the 1,676,663,913-byte checkpoint was being downloaded during that load interval. A later cached run loaded the same checkpoint in 11.338 s, generated a 4.667-second WAV in 17.141 s, and generated a 1.613-second warm WAV in 6.821 s (`load_seconds=0`). Both responses were CUDA-backed and parsed successfully with SoundFile.

*TinySD's model load took 374.254 s and its first request completed in 382.8 s end-to-end (5.931 s inference). The warm request completed in 4.318 s (4.277 s inference, `load_seconds=0`). Both requests ran on CUDA with CPU fallback disabled. Peak allocated VRAM was 2,007,410,176 bytes. The 512x512 PNG responses were each 412,473 bytes.

*Shap-E's 1,844.395 s load included completing the checkpoint download and loading the model. The first real text-to-3D request took 1,889.985 s end-to-end (27.118 s inference); the warm request took 35.627 s (26.029 s inference, `load_seconds=0`). Both ran on CUDA with CPU fallback disabled. Peak allocated VRAM was 2,369,751,040 bytes. The first OBJ is 3,132,869 bytes (30,524 vertices, 61,060 faces); the warm OBJ is 1,851,031 bytes (18,350 vertices, 36,708 faces).

TinyLlama checkpoint: `D:\FastAPI-AI\.cache\huggingface\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6\model.safetensors`; SHA-256 verified as `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`.

Generated files:

- `D:\FastAPI-AI\outputs\text\72e3beaf36f24734b4b0e12caf91dc74.json`
- `D:\FastAPI-AI\outputs\text\ea9123f12a5a4f45a14fc7ade6a76b20.json`
- `D:\FastAPI-AI\outputs\audio\36dc70a2b60e4d93bb5d2af90dfe809e.wav`
- `D:\FastAPI-AI\outputs\audio\1bba003277e0446db03cea96ace5a57e.wav`
- `D:\FastAPI-AI\outputs\audio\80caff2c1b4241d5894e9c5d3a2b0392.wav` (initial download run)
- `D:\FastAPI-AI\outputs\images\114a0fbf95844197a8472d461dd881ea.png`
- `D:\FastAPI-AI\outputs\images\e675957a371143b5b13be99d67ea7798.png`
- `D:\FastAPI-AI\outputs\models_3d\416acd4248fb4dbcbc3f2007ffc18472.obj`
- `D:\FastAPI-AI\outputs\models_3d\574e58715d854abb8233414cfb25a46c.obj`

Bark checkpoint: `D:\FastAPI-AI\.cache\huggingface\models--suno--bark-small\snapshots\1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd\pytorch_model.bin`; SHA-256 verified as `f0f7f16b24f65789ce42b3c491aa6a1cdf219f7ef425066fcd194485245e65d9`.

TinySD checkpoint snapshot: `D:\FastAPI-AI\.cache\huggingface\models--segmind--tiny-sd\snapshots\cad0bd7495fa6c4bcca01b19a723dc91627fe84f`; the text encoder, U-Net, and VAE weight files total 1,060,519,363 bytes and were all loaded successfully by the real CUDA inference run.

Shap-E checkpoint snapshot: `D:\FastAPI-AI\.cache\huggingface\models--openai--shap-e\snapshots\7bd337afdea1c17842e1c3cc45c4e268356dba40`; the prior, renderer, Shap-E renderer, and text encoder weight files total 3,568,035,918 bytes and were all loaded successfully by the real CUDA inference run.

The `/system/info` endpoint uses `nvidia-smi` for GPU utilization/used-memory snapshots where available. Per-inference allocated/peak CUDA memory metrics are collected once inference runs. No utilization or memory values are estimated. Transfer investigation and evidence are documented in [acceptance-report.md](./acceptance-report.md).
