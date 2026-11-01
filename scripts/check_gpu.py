import json

import torch


def main() -> None:
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "compiled_architectures": torch.cuda.get_arch_list(),
    }
    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA is unavailable; GPU test did not run: {report}")
    props = torch.cuda.get_device_properties(0)
    report["gpu"] = props.name
    report["compute_capability"] = f"{props.major}.{props.minor}"
    arch = f"sm_{props.major}{props.minor}"
    report["required_architecture"] = arch
    report["architecture_in_binary"] = arch in torch.cuda.get_arch_list()
    compatible_arches = [
        item
        for item in torch.cuda.get_arch_list()
        if item.startswith(f"sm_{props.major}")
        and int(item.removeprefix(f"sm_{props.major}")) <= props.minor
    ]
    report["compatible_compiled_architectures"] = compatible_arches
    if not compatible_arches:
        raise SystemExit(f"Installed PyTorch has no compatible Pascal kernel: {report}")
    a = torch.arange(1, 1025, device="cuda", dtype=torch.float32)
    result = (a * a).sum()
    torch.cuda.synchronize()
    report["tensor_test_result"] = result.item()
    report["tensor_test_device"] = str(result.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
