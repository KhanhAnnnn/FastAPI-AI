# Real inference acceptance report

Created: 2026-10-09T18:51:02.330713+00:00

| Model | Status | Load (s) | Inference (s) | Device | Peak VRAM | Output validation |
|---|---:|---:|---:|---|---:|---|
| TinyLlama | PASS | 2272.130606099963 | 13.46789460000582 | cuda | 2219912704 | non-empty assistant text from two requests |
| Bark | PASS | 11.338 | 17.141 | cuda | 984007168 | validated output |
| TinySD | PASS | 374.254 | 5.931 | cuda | 2007410176 | validated output |
| Shap-E | PASS | 1844.395 | 27.118 | cuda | 2369751040 | validated output |
| Stable Video Diffusion | BLOCKED | None | N/A | N/A | N/A | Inference not started pending user approval for a memory-sensitive run; previous checkpoint download stalled and the snapshot remains incomplete. |

## TinyLlama artifacts
- Output: `D:\FastAPI-AI\outputs\text\72e3beaf36f24734b4b0e12caf91dc74.json`
- Output: `D:\FastAPI-AI\outputs\text\ea9123f12a5a4f45a14fc7ade6a76b20.json`
- Checkpoint: `D:\FastAPI-AI\.cache\huggingface\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6\model.safetensors` (2200119864 bytes; SHA-256 `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`).
- Warm request: 13.346 s; inference 13.341825500014238 s; device `cuda`.
- Content note: The first response incorrectly described FastAPI as Flask-based; this acceptance verifies real inference and artifact validity, not factual accuracy.

## Bark artifacts
- Output: `D:\FastAPI-AI\outputs\audio\36dc70a2b60e4d93bb5d2af90dfe809e.wav`
- Output: `D:\FastAPI-AI\outputs\audio\1bba003277e0446db03cea96ace5a57e.wav`
- Checkpoint: `D:\FastAPI-AI\.cache\huggingface\models--suno--bark-small\snapshots\1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd\pytorch_model.bin` (1676663913 bytes; SHA-256 `f0f7f16b24f65789ce42b3c491aa6a1cdf219f7ef425066fcd194485245e65d9`).
- Warm request: 6.888 s; inference 6.821 s; device `cuda`.
- WAV: 224044 bytes, 24000 Hz, 4.666666666666667 s.
- Warm WAV: 77484 bytes, 24000 Hz, 1.6133333333333333 s.

## Bark prior successful run
- Load: 857.262 s; inference: 847.411 s; device: `cuda`.

## Bark prior attempt
- Status: BLOCKED; error: OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models/suno/bark-small: offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable.

## TinySD artifacts
- Output: `D:\FastAPI-AI\outputs\images\114a0fbf95844197a8472d461dd881ea.png`
- Output: `D:\FastAPI-AI\outputs\images\e675957a371143b5b13be99d67ea7798.png`
- Checkpoint snapshot: `D:\FastAPI-AI\.cache\huggingface\models--segmind--tiny-sd\snapshots\cad0bd7495fa6c4bcca01b19a723dc91627fe84f` (text encoder, U-Net, and VAE weight files total 1,060,519,363 bytes; all loaded during CUDA inference).
- Warm request: 4.318 s; inference 4.277 s; device `cuda`.

## TinySD prior attempt
- Status: BLOCKED; error: Checkpoint transfer interrupted: 515516399 bytes received, 131407238 remaining; no PNG generated.

## Shap-E artifacts
- Output: `D:\FastAPI-AI\outputs\models_3d\416acd4248fb4dbcbc3f2007ffc18472.obj`
- Output: `D:\FastAPI-AI\outputs\models_3d\574e58715d854abb8233414cfb25a46c.obj`
- Checkpoint snapshot: `D:\FastAPI-AI\.cache\huggingface\models--openai--shap-e\snapshots\7bd337afdea1c17842e1c3cc45c4e268356dba40` (prior, renderer, Shap-E renderer, and text encoder weights total 3,568,035,918 bytes; all loaded during CUDA inference).
- OBJ validation: first output 30,524 vertices and 61,060 faces; warm output 18,350 vertices and 36,708 faces.
- Warm request: 35.627 s; inference 26.029 s; device `cuda`.

## Shap-E prior attempt
- Status: NOT TESTED; error: Not reached by the sequential inference run.

## Download troubleshooting
- Root cause: Intermittent DNS resolution/reachability of the Hugging Face CDN host used for Xet ranges; this is not a repository permission or gated-model failure.
- The Hub metadata/API request succeeded and all five repositories were public, private=false, gated=false.
- The active Xet log recorded 14 'No such host is known' request errors for us.aws.cdn.hf.co.
- A system DNS lookup for us.aws.cdn.hf.co timed out once, then resolved on a later check; huggingface.co and cas-server.xethub.hf.co resolved.
- Earlier transfers also recorded IncompleteRead and Windows ConnectionResetError 10054. A prior Xet 416 occurred during range reconstruction and was followed by successful 206 range responses.
- No 401, 403, 429, or 5xx response was observed in the available transfer evidence.
- An explicit HF_HUB_OFFLINE probe for Bark was blocked at the model-metadata API call; it did not reach inference. With offline mode unset, the completed Bark checkpoint loaded from cache in 11.338 s and both inference requests passed.
- Attempt settings: `HF_HUB_DOWNLOAD_TIMEOUT=300`, `HF_HUB_ETAG_TIMEOUT=60`, `HF_XET_HIGH_PERFORMANCE=1`.
- Resolution: TinyLlama and Bark checkpoints completed, passed SHA-256 verification, and produced real CUDA inference outputs. TinySD's existing partial transfer was resumed in place using the 300 s download timeout; its complete text encoder, U-Net, and VAE weights then loaded and produced two valid CUDA PNGs. Shap-E's full 3,568,035,918-byte weight set loaded and produced two valid CUDA OBJ files. No cache was deleted and no dependency versions changed.
- Acceptance-run log: `D:\FastAPI-AI\reports\tinyllama-download-inference.log`
- TinySD run log: `D:\FastAPI-AI\reports\tinysd-download-inference.log`
- Shap-E run log: `D:\FastAPI-AI\reports\shape-download-inference.log`

## Stable Video Diffusion deferred
- Status: BLOCKED by user decision; do not resume the checkpoint download or run inference until the user explicitly asks to continue.
- Reason: the checkpoint snapshot is incomplete and no real SVD inference or MP4 output has been produced. The prior download attempt stalled with no transfer progress for more than 15 minutes.
- Existing partial weight blobs are preserved at `D:\FastAPI-AI\.cache\huggingface\models--stabilityai--stable-video-diffusion-img2vid\blobs\ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a.incomplete` (440,401,920 bytes) and `D:\FastAPI-AI\.cache\huggingface\models--stabilityai--stable-video-diffusion-img2vid\blobs\cb552818963e736506c6693ffc279d59df423d63aced38902ea4373ab1fd2932.incomplete` (555,745,280 bytes). Do not delete or replace these files.
- To continue later, first obtain explicit user approval to resume the existing checkpoint transfer. Use the existing Python environment and model ID; keep the current cache and use bounded download timeouts. Confirm the snapshot is complete before attempting inference.
- Before a future inference attempt, obtain explicit user approval for the memory-sensitive run. The installed pipeline already enables CPU model offload, VAE slicing, U-Net forward chunking, and `decode_chunk_size=2`. The official Diffusers guidance says its offload/chunking optimizations can reduce VRAM use below 8 GB, but this remains close to the Quadro P4000's 8 GiB capacity. Acceptance must still require a valid MP4, measured CUDA device/timings/peak VRAM, and a warm second request; never report PASS based on download or import alone.
