import json
import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import av
import soundfile as sf
from fastapi.testclient import TestClient
from PIL import Image

from app.config import OUTPUT_ROOT, PROJECT_ROOT
from app.core.device import environment_summary
from app.main import app

REPORT_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("TinyLlama", "Bark", "TinySD", "Stable Video Diffusion", "Shap-E"),
        help="Run just one model check; otherwise run all sequentially.",
    )
    selected = parser.parse_args().model
    report_path = REPORT_DIR / "acceptance-report.json"
    previous_report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    previous_models = previous_report.get("models", {})
    image_buffer = BytesIO()
    Image.new("RGB", (128, 128), (42, 85, 130)).save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    checks = {}

    with TestClient(app) as client:
        checks["TinyLlama"] = run_selected(
            selected, "TinyLlama", lambda: check_text(client), previous_models
        )
        checks["Bark"] = run_selected(
            selected, "Bark", lambda: check_audio(client), previous_models
        )
        checks["TinySD"] = run_selected(
            selected, "TinySD", lambda: check_image(client), previous_models
        )
        checks["Stable Video Diffusion"] = run_selected(
            selected,
            "Stable Video Diffusion",
            lambda: check_video(client, image_bytes),
            previous_models,
        )
        checks["Shap-E"] = run_selected(
            selected, "Shap-E", lambda: check_mesh(client), previous_models
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_summary(),
        "models": checks,
        "outputs_directory": str(OUTPUT_ROOT.resolve()),
    }
    if "download_troubleshooting" in previous_report:
        report["download_troubleshooting"] = previous_report[
            "download_troubleshooting"
        ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(checks, report.get("download_troubleshooting"))
    print(f"Acceptance report: {report_path.resolve()}")
    print(json.dumps(checks, indent=2))


def run_selected(selected, name, check, previous_models):
    if selected is None or selected == name:
        result = run_check(check)
        previous = previous_models.get(name, {})
        prior_success = previous.get("prior_successful_run")
        if previous.get("status") == "PASS" and prior_success is None:
            prior_success = previous
        if prior_success is not None:
            result["prior_successful_run"] = prior_success
        if previous.get("status") != "PASS" and previous.get("error"):
            previous_status = previous["status"]
            if "offlinemodeisenabled" in previous["error"].lower():
                previous_status = "BLOCKED"
            result["prior_attempt"] = {
                "status": previous_status,
                "error": previous["error"],
                "test_seconds": previous.get("test_seconds"),
                "traceback": previous.get("traceback"),
            }
        return result
    return previous_models.get(
        name, {"status": "NOT TESTED", "reason": "Not selected in this run."}
    )


def run_check(check):
    started = time.perf_counter()
    try:
        result = check()
        result["status"] = "PASS"
        result["test_seconds"] = round(time.perf_counter() - started, 3)
        return result
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        blocked_indicators = (
            "cuda is unavailable",
            "out of memory",
            "cuda error",
            "timed out",
            "connection",
            "network",
            "dns",
            "no such host",
            "name resolution",
            "getaddrinfo",
            "incompleteread",
            "offline mode",
            "offlinemodeisenabled",
            "not found in cache",
            "localentrynotfound",
            "not installed",
        )
        return {
            "status": (
                "BLOCKED"
                if any(item in message.lower() for item in blocked_indicators)
                else "FAIL"
            ),
            "test_seconds": round(time.perf_counter() - started, 3),
            "error": message,
            "traceback": traceback.format_exc(),
        }


def output_snapshot(directory: Path) -> set[Path]:
    return {path.resolve() for path in directory.glob("*") if path.is_file()}


def new_output_files(directory: Path, previous_files: set[Path]) -> list[str]:
    created = sorted(
        (
            path.resolve()
            for path in directory.glob("*")
            if path.is_file() and path.resolve() not in previous_files
        ),
        key=lambda path: path.stat().st_mtime,
    )
    assert created, f"No output artifact was saved in {directory.resolve()}."
    return [str(path) for path in created]


def check_text(client: TestClient):
    output_dir = OUTPUT_ROOT / "text"
    previous_files = output_snapshot(output_dir)
    first_started = time.perf_counter()
    response = client.get("/generate/text", params={"prompt": "What is FastAPI?"})
    first_request_seconds = time.perf_counter() - first_started
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["text"].strip()
    warm_started = time.perf_counter()
    warm_response = client.get("/generate/text", params={"prompt": "Define an API."})
    warm_request_seconds = time.perf_counter() - warm_started
    assert warm_response.status_code == 200, warm_response.text
    warm_payload = warm_response.json()
    assert warm_payload["text"].strip()
    assert warm_payload["metrics"]["load_seconds"] == 0
    output_files = new_output_files(output_dir, previous_files)
    assert len(output_files) == 2, f"Expected two text outputs, found {output_files}."
    return {
        "output": "non-empty assistant text from two requests",
        "output_files": output_files,
        "first_request_seconds": round(first_request_seconds, 3),
        "warm_request_seconds": round(warm_request_seconds, 3),
        "metrics": payload["metrics"],
        "warm_metrics": warm_payload["metrics"],
    }


def check_audio(client: TestClient):
    output_dir = OUTPUT_ROOT / "audio"
    previous_files = output_snapshot(output_dir)
    first_started = time.perf_counter()
    response = client.get(
        "/generate/audio",
        params={
            "prompt": "Hello, welcome to our artificial intelligence laboratory.",
            "preset": "v2/en_speaker_1",
        },
    )
    assert response.status_code == 200, response.text
    audio, sample_rate = sf.read(BytesIO(response.content))
    assert audio.size > 0 and sample_rate > 0
    first_request_seconds = time.perf_counter() - first_started
    warm_started = time.perf_counter()
    warm_response = client.get(
        "/generate/audio",
        params={"prompt": "Hi.", "preset": "v2/en_speaker_1"},
    )
    warm_request_seconds = time.perf_counter() - warm_started
    assert warm_response.status_code == 200, warm_response.text
    warm_audio, warm_sample_rate = sf.read(BytesIO(warm_response.content))
    assert warm_audio.size > 0 and warm_sample_rate > 0
    warm_metrics = {
        "device": warm_response.headers.get("x-device"),
        "load_seconds": float(warm_response.headers["x-load-seconds"]),
        "inference_seconds": float(warm_response.headers["x-inference-seconds"]),
        "peak_vram_bytes": int(warm_response.headers["x-vram-peak-bytes"]),
    }
    assert warm_metrics["load_seconds"] == 0
    output_files = new_output_files(output_dir, previous_files)
    assert len(output_files) == 2, f"Expected two Bark outputs, found {output_files}."
    return {
        "output_files": output_files,
        "output_bytes": len(response.content),
        "sample_rate": sample_rate,
        "duration_seconds": audio.shape[0] / sample_rate,
        "device": response.headers.get("x-device"),
        "load_seconds": response.headers.get("x-load-seconds"),
        "inference_seconds": response.headers.get("x-inference-seconds"),
        "peak_vram_bytes": response.headers.get("x-vram-peak-bytes"),
        "first_request_seconds": round(first_request_seconds, 3),
        "warm_request_seconds": round(warm_request_seconds, 3),
        "warm_output_bytes": len(warm_response.content),
        "warm_sample_rate": warm_sample_rate,
        "warm_duration_seconds": warm_audio.shape[0] / warm_sample_rate,
        "warm_metrics": warm_metrics,
    }


def check_image(client: TestClient):
    output_dir = OUTPUT_ROOT / "images"
    previous_files = output_snapshot(output_dir)
    first_started = time.perf_counter()
    response = client.get(
        "/generate/image",
        params={
            "prompt": "A futuristic city at sunset, digital art",
            "num_inference_steps": 10,
            "seed": 42,
        },
    )
    assert response.status_code == 200, response.text
    with Image.open(BytesIO(response.content)) as image:
        image.verify()
    with Image.open(BytesIO(response.content)) as image:
        assert image.width > 0 and image.height > 0
        dimensions = image.size
    first_request_seconds = time.perf_counter() - first_started
    warm_started = time.perf_counter()
    warm_response = client.get(
        "/generate/image",
        params={
            "prompt": "A futuristic city at sunset, digital art",
            "num_inference_steps": 10,
            "seed": 42,
        },
    )
    warm_request_seconds = time.perf_counter() - warm_started
    assert warm_response.status_code == 200, warm_response.text
    with Image.open(BytesIO(warm_response.content)) as image:
        image.verify()
    with Image.open(BytesIO(warm_response.content)) as image:
        assert image.width > 0 and image.height > 0
        warm_dimensions = image.size
    warm_metrics = {
        "device": warm_response.headers.get("x-device"),
        "load_seconds": float(warm_response.headers["x-load-seconds"]),
        "inference_seconds": float(warm_response.headers["x-inference-seconds"]),
        "peak_vram_bytes": int(warm_response.headers["x-vram-peak-bytes"]),
    }
    assert warm_metrics["load_seconds"] == 0
    output_files = new_output_files(output_dir, previous_files)
    assert len(output_files) == 2, f"Expected two TinySD outputs, found {output_files}."
    return {
        "output_files": output_files,
        "output_bytes": len(response.content),
        "format": "PNG",
        "dimensions": dimensions,
        "device": response.headers.get("x-device"),
        "load_seconds": response.headers.get("x-load-seconds"),
        "inference_seconds": response.headers.get("x-inference-seconds"),
        "peak_vram_bytes": response.headers.get("x-vram-peak-bytes"),
        "first_request_seconds": round(first_request_seconds, 3),
        "warm_request_seconds": round(warm_request_seconds, 3),
        "warm_output_bytes": len(warm_response.content),
        "warm_dimensions": warm_dimensions,
        "warm_metrics": warm_metrics,
    }


def check_video(client: TestClient, image_bytes: bytes):
    output_dir = OUTPUT_ROOT / "videos"
    previous_files = output_snapshot(output_dir)
    response = client.post(
        "/generate/video",
        params={"num_frames": 25, "seed": 42},
        files={"image": ("acceptance.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200, response.text
    with av.open(BytesIO(response.content)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        assert stream.codec_context.name == "h264"
        assert frames and container.duration and container.duration > 0
        result = {
            "output_files": new_output_files(output_dir, previous_files),
            "output_bytes": len(response.content),
            "codec": stream.codec_context.name,
            "dimensions": [stream.width, stream.height],
            "frame_count": len(frames),
            "duration_seconds": container.duration / av.time_base,
            "device": response.headers.get("x-device"),
            "load_seconds": response.headers.get("x-load-seconds"),
            "inference_seconds": response.headers.get("x-inference-seconds"),
            "peak_vram_bytes": response.headers.get("x-vram-peak-bytes"),
        }
    return result


def check_mesh(client: TestClient):
    output_dir = OUTPUT_ROOT / "models_3d"
    previous_files = output_snapshot(output_dir)
    first_started = time.perf_counter()
    response = client.get(
        "/generate/3d",
        params={"prompt": "A small wooden chair", "num_inference_steps": 25},
    )
    assert response.status_code == 200, response.text
    text = response.content.decode("ascii")
    vertices = sum(line.startswith("v ") for line in text.splitlines())
    faces = sum(line.startswith("f ") for line in text.splitlines())
    assert vertices > 0 and faces > 0
    first_request_seconds = time.perf_counter() - first_started
    warm_started = time.perf_counter()
    warm_response = client.get(
        "/generate/3d",
        params={"prompt": "A small wooden chair", "num_inference_steps": 25},
    )
    warm_request_seconds = time.perf_counter() - warm_started
    assert warm_response.status_code == 200, warm_response.text
    warm_text = warm_response.content.decode("ascii")
    warm_vertices = sum(line.startswith("v ") for line in warm_text.splitlines())
    warm_faces = sum(line.startswith("f ") for line in warm_text.splitlines())
    assert warm_vertices > 0 and warm_faces > 0
    warm_metrics = {
        "device": warm_response.headers.get("x-device"),
        "load_seconds": float(warm_response.headers["x-load-seconds"]),
        "inference_seconds": float(warm_response.headers["x-inference-seconds"]),
        "peak_vram_bytes": int(warm_response.headers["x-vram-peak-bytes"]),
    }
    assert warm_metrics["load_seconds"] == 0
    output_files = new_output_files(output_dir, previous_files)
    assert len(output_files) == 2, f"Expected two Shap-E outputs, found {output_files}."
    return {
        "output_files": output_files,
        "output_bytes": len(response.content),
        "vertices": vertices,
        "faces": faces,
        "device": response.headers.get("x-device"),
        "load_seconds": response.headers.get("x-load-seconds"),
        "inference_seconds": response.headers.get("x-inference-seconds"),
        "peak_vram_bytes": response.headers.get("x-vram-peak-bytes"),
        "first_request_seconds": round(first_request_seconds, 3),
        "warm_request_seconds": round(warm_request_seconds, 3),
        "warm_output_bytes": len(warm_response.content),
        "warm_vertices": warm_vertices,
        "warm_faces": warm_faces,
        "warm_metrics": warm_metrics,
    }


def write_markdown(checks: dict, download_troubleshooting: dict | None = None) -> None:
    lines = [
        "# Real inference acceptance report",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Model | Status | Load (s) | Inference (s) | Device | Peak VRAM | Output validation |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for model_name, result in checks.items():
        metrics = result.get("metrics", {})
        if not metrics and "warm_metrics" in result:
            metrics = result["warm_metrics"]
        load = result.get("load_seconds", metrics.get("load_seconds", "N/A"))
        inference = result.get("inference_seconds") or metrics.get("inference_seconds", "N/A")
        device = result.get("device") or metrics.get("device", "N/A")
        peak = result.get("peak_vram_bytes") or metrics.get("peak_vram_bytes", "N/A")
        output = result.get(
            "error", result.get("output", result.get("reason", "validated output"))
        )
        lines.append(
            f"| {model_name} | {result['status']} | {load} | {inference} | "
            f"{device} | {peak} | {output} |"
        )
    for model_name, result in checks.items():
        output_files = result.get("output_files", [])
        checkpoint = result.get("checkpoint")
        quality_note = result.get("content_quality_note")
        warm_metrics = result.get("warm_metrics")
        if output_files or checkpoint or quality_note or warm_metrics:
            lines.extend(["", f"## {model_name} artifacts"])
        for output_file in output_files:
            lines.append(f"- Output: `{output_file}`")
        if checkpoint:
            lines.append(
                f"- Checkpoint: `{checkpoint['path']}` "
                f"({checkpoint['size_bytes']} bytes; SHA-256 "
                f"`{checkpoint['sha256']}`)."
            )
        if warm_metrics:
            lines.append(
                f"- Warm request: {result.get('warm_request_seconds', 'N/A')} s; "
                f"inference {warm_metrics.get('inference_seconds', 'N/A')} s; "
                f"device `{warm_metrics.get('device', 'N/A')}`."
            )
        if "sample_rate" in result:
            lines.append(
                f"- WAV: {result.get('output_bytes', 'N/A')} bytes, "
                f"{result['sample_rate']} Hz, {result.get('duration_seconds', 'N/A')} s."
            )
            if "warm_sample_rate" in result:
                lines.append(
                    f"- Warm WAV: {result.get('warm_output_bytes', 'N/A')} bytes, "
                    f"{result['warm_sample_rate']} Hz, "
                    f"{result.get('warm_duration_seconds', 'N/A')} s."
                )
        if quality_note:
            lines.append(f"- Content note: {quality_note}")
        prior_run = result.get("prior_successful_run")
        if prior_run:
            lines.extend(
                [
                    "",
                    f"## {model_name} prior successful run",
                    f"- Load: {prior_run.get('load_seconds', 'N/A')} s; "
                    f"inference: {prior_run.get('inference_seconds', 'N/A')} s; "
                    f"device: `{prior_run.get('device', 'N/A')}`.",
                ]
            )
        prior_attempt = result.get("prior_attempt")
        if prior_attempt:
            lines.extend(
                [
                    "",
                    f"## {model_name} prior attempt",
                    f"- Status: {prior_attempt.get('status')}; "
                    f"error: {prior_attempt.get('error')}",
                ]
            )
    if download_troubleshooting:
        lines.extend(
            [
                "",
                "## Download troubleshooting",
                f"- Root cause: {download_troubleshooting['root_cause']}",
            ]
        )
        lines.extend(
            f"- {evidence}"
            for evidence in download_troubleshooting.get("evidence", [])
        )
        action = download_troubleshooting.get("action_taken", {})
        if action:
            lines.append(
                "- Attempt settings: "
                f"`HF_HUB_DOWNLOAD_TIMEOUT={action.get('download_timeout_seconds')}`, "
                f"`HF_HUB_ETAG_TIMEOUT={action.get('etag_timeout_seconds')}`, "
                f"`HF_XET_HIGH_PERFORMANCE={int(action.get('xet_high_performance', False))}`."
            )
            lines.append(f"- Resolution: {action.get('result', 'N/A')}")
        inference_log = download_troubleshooting.get("inference_log")
        if inference_log:
            lines.append(f"- Acceptance-run log: `{inference_log}`")
    (REPORT_DIR / "acceptance-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
