# Hardware and environment audit

## Initial observations

| Item | Observed value |
|---|---|
| OS | Windows PowerShell environment |
| NVIDIA GPU | Quadro P4000, 8192 MiB |
| NVIDIA driver | 572.76 |
| `nvidia-smi` CUDA compatibility | 12.8 |
| Free space on `D:` | 73,105,034,240 bytes at initial audit |
| Free space on `C:` | 9,820,753,920 bytes at initial audit |
| Project Python | Python 3.13.2 at `D:\python.exe`; `ensurepip` available |
| Pre-existing project `.venv` | Python 3.14.4; left unchanged |
| Pre-existing global Python packages | FastAPI and Transformers present in Python 3.14; PyTorch absent |
| PyTorch in the original project environment | Not installed |

The stated Pascal compute capability `6.1` is user-provided hardware information. The installed PyTorch binary must still report a compatible compiled architecture and pass a real CUDA tensor operation before `sm_61` support is considered verified. `nvidia-smi` reporting CUDA 12.8 does not determine the PyTorch CUDA runtime.

## Selected isolated environment

A new environment was created at `D:\FastAPI-AI\.venv-genai` with Python 3.13.2. It does not replace `.venv` or modify either system Python installation.

The initial GPU package selection is PyTorch 2.7.1 / torchvision 0.22.1 with CUDA 12.6 (`cu126`). This is an intentionally pinned CUDA 12.x candidate for the Python 3.13 Windows wheel and installed driver. It passed the real tensor test described below. `hf_xet` is pinned to support Hugging Face large-file transfer. No `bfloat16` is configured. No driver, system CUDA Toolkit, or global Python packages are changed.

## Validation status

| Check | Status |
|---|---|
| Driver sees Quadro P4000 | Verified by `nvidia-smi` |
| PyTorch CUDA runtime/version | 12.6 (`torch 2.7.1+cu126`) |
| `torch.cuda.is_available()` | `True` |
| PyTorch compiled architecture compatibility | `sm_61` present in the binary architecture list |
| Real CUDA tensor operation | Passed on `cuda:0`; sum of squares of 1..1024 was 358,438,400.0 |
| Dependency conflict resolution | `pip check` reported no broken requirements |
| TinyLlama real inference | PASS on `cuda`; the model and inputs were moved to CUDA, with 2,219,912,704 bytes measured peak allocated VRAM. |
| Bark real inference | PASS on `cuda`; valid WAV outputs from cold and warm requests, with 984,007,168 bytes measured peak allocated VRAM. |
| TinySD real inference | PASS on `cuda`; two valid 512x512 PNGs from cold and warm requests, with 2,007,410,176 bytes measured peak allocated VRAM. |
| Shap-E real inference | PASS on `cuda`; two valid OBJ meshes from cold and warm requests, with 2,369,751,040 bytes measured peak allocated VRAM. |
| Other model statuses | Tracked in [acceptance-report.md](./acceptance-report.md). |

Full JSON evidence: `D:\FastAPI-AI\reports\gpu-check.json`. This verifies a small CUDA operation, not model inference or model-level compatibility. Do not treat this report or driver detection alone as proof of model inference compatibility.

## Model-level verification

TinyLlama completed two real text-generation requests on the NVIDIA Quadro P4000 with CPU fallback disabled. The checkpoint was `D:\FastAPI-AI\.cache\huggingface\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6\model.safetensors`; its SHA-256 matched the expected cache blob. PyTorch measured a 2,219,912,704-byte allocated-memory peak during inference, and the warm request reported `device=cuda` with zero model load time. Outputs are `D:\FastAPI-AI\outputs\text\72e3beaf36f24734b4b0e12caf91dc74.json` and `D:\FastAPI-AI\outputs\text\ea9123f12a5a4f45a14fc7ade6a76b20.json`.

TinyLlama, TinySD, and Shap-E compatibility is confirmed on this GPU. Stable Video Diffusion remains untested; the official Diffusers guide recommends CPU model offload, U-Net forward chunking, and a small VAE decode chunk, and says using the options together should reduce VRAM use below 8 GB. The app already enables model CPU offload, VAE slicing, U-Net forward chunking, and `decode_chunk_size=2`, but the 8 GiB GPU has only 8,068 MiB free while idle. That leaves limited safety margin, so no SVD inference has been launched pending user approval. See the download-troubleshooting evidence in [acceptance-report.md](./acceptance-report.md).

Bark also passed two real text-to-speech requests on CUDA with CPU fallback disabled. The checkpoint was `D:\FastAPI-AI\.cache\huggingface\models--suno--bark-small\snapshots\1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd\pytorch_model.bin` (1,676,663,913 bytes; SHA-256 `f0f7f16b24f65789ce42b3c491aa6a1cdf219f7ef425066fcd194485245e65d9`). The cached model load took 11.338 s, the first 24 kHz WAV was 4.667 s long, and the warm request took 6.821 s inference with `load_seconds=0`; peak allocated VRAM was 984,007,168 bytes. Validated outputs are `D:\FastAPI-AI\outputs\audio\36dc70a2b60e4d93bb5d2af90dfe809e.wav` and `D:\FastAPI-AI\outputs\audio\1bba003277e0446db03cea96ace5a57e.wav`.

TinySD passed two real 512x512 image-generation requests on CUDA with CPU fallback disabled. The complete loaded weight files reside under `D:\FastAPI-AI\.cache\huggingface\models--segmind--tiny-sd\snapshots\cad0bd7495fa6c4bcca01b19a723dc91627fe84f`; text encoder, U-Net, and VAE weights total 1,060,519,363 bytes. Load took 374.254 s including checkpoint resume, first end-to-end request took 382.8 s (5.931 s inference), and the warm request took 4.318 s (4.277 s inference, `load_seconds=0`). Measured peak allocated VRAM was 2,007,410,176 bytes. Both PNG outputs are valid, 412,473 bytes each: `D:\FastAPI-AI\outputs\images\114a0fbf95844197a8472d461dd881ea.png` and `D:\FastAPI-AI\outputs\images\e675957a371143b5b13be99d67ea7798.png`.

Shap-E passed two real text-to-3D requests on CUDA with CPU fallback disabled. The complete prior, renderer, Shap-E renderer, and text encoder weights reside under `D:\FastAPI-AI\.cache\huggingface\models--openai--shap-e\snapshots\7bd337afdea1c17842e1c3cc45c4e268356dba40` and total 3,568,035,918 bytes. Load took 1,844.395 s, including transfer; the first request took 1,889.985 s end-to-end (27.118 s inference), and the warm request took 35.627 s (26.029 s inference, `load_seconds=0`). Measured peak allocated VRAM was 2,369,751,040 bytes. Both OBJ meshes passed structural validation: `D:\FastAPI-AI\outputs\models_3d\416acd4248fb4dbcbc3f2007ffc18472.obj` and `D:\FastAPI-AI\outputs\models_3d\574e58715d854abb8233414cfb25a46c.obj`.

Stable Video Diffusion was not run: its snapshot is incomplete, with two `.incomplete` weight blobs of 440,401,920 and 555,745,280 bytes; the current run's available free RAM is 46,076,174,336 bytes and idle free VRAM is 8,068 MiB. The installed SVD loader uses the official memory optimization strategy described at https://huggingface.co/docs/diffusers/main/en/using-diffusers/svd. Because the documented target is below 8 GB and this GPU has little headroom below that threshold, request approval before attempting full inference.
