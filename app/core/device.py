from __future__ import annotations

import os
import subprocess

from app.config import CPU_FALLBACK_ENABLED


def get_torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if CPU_FALLBACK_ENABLED:
        return torch.device("cpu")
    reason = (
        "CUDA is unavailable in this Python environment. "
        "CPU fallback is disabled; see /system/info for diagnostics."
    )
    raise RuntimeError(reason)


def environment_summary() -> dict[str, object]:
    import importlib.metadata
    import platform

    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_fallback_enabled": CPU_FALLBACK_ENABLED,
        "cache_directory": os.environ.get("HF_HOME", ""),
        "packages": {},
    }
    for package in ("torch", "transformers", "diffusers", "fastapi"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = None
    try:
        import torch

        info["torch_cuda_runtime"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "architecture": f"sm_{props.major}{props.minor}",
                "total_memory_bytes": props.total_memory,
                "compiled_architectures": torch.cuda.get_arch_list(),
            }
        else:
            info["cuda_error"] = "torch.cuda.is_available() returned False."
    except ImportError:
        info["torch_cuda_runtime"] = None
        info["cuda_available"] = False
        info["gpu_error"] = "PyTorch is not installed."
    except Exception as exc:
        info["cuda_available"] = False
        info["gpu_error"] = f"{type(exc).__name__}: {exc}"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        info["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        info["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return info
