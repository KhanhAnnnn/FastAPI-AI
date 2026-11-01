def response_metric_headers(
    metrics: dict[str, object], serialization_seconds: float = 0.0
) -> dict[str, str]:
    headers = {
        "X-Model-Id": str(metrics["model_id"]),
        "X-Device": str(metrics["device"]),
        "X-Load-Seconds": f"{metrics['load_seconds']:.3f}",
        "X-Inference-Seconds": f"{metrics['inference_seconds']:.3f}",
        "X-Serialization-Seconds": f"{serialization_seconds:.3f}",
        "X-Total-Seconds": f"{metrics['total_seconds'] + serialization_seconds:.3f}",
        "X-Process-RSS-Bytes": str(metrics["process_rss_after_bytes"]),
        "X-Process-CPU-Seconds": f"{metrics['process_cpu_seconds']:.3f}",
        "X-VRAM-Before-Bytes": str(metrics["vram_before_bytes"] or "N/A"),
        "X-VRAM-After-Bytes": str(metrics["vram_after_bytes"] or "N/A"),
        "X-VRAM-Peak-Bytes": str(metrics["peak_vram_bytes"] or "N/A"),
    }
    return headers
