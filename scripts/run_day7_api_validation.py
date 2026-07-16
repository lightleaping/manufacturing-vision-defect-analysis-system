"""실제 Uvicorn Process와 ResNet18 Checkpoint를 사용한 Day 7 HTTP 검증.

이 Script는 단위·통합 테스트가 통과한 뒤 실행한다.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = Path(
    "reports/artifacts/day7_fastapi_inference_validation.json"
)
DEFAULT_NORMAL_IMAGE_PATH = Path(
    "data/raw/casting_product_images/casting_data/casting_data/"
    "test/ok_front/cast_ok_0_7631.jpeg"
)
DEFAULT_DEFECT_IMAGE_PATH = Path(
    "data/raw/casting_product_images/casting_data/casting_data/"
    "test/def_front/cast_def_0_1414.jpeg"
)


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Day 7 real FastAPI image inference validation."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--normal-image-path",
        type=Path,
        default=DEFAULT_NORMAL_IMAGE_PATH,
    )
    parser.add_argument(
        "--defect-image-path",
        type=Path,
        default=DEFAULT_DEFECT_IMAGE_PATH,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args(arguments)


def ensure_port_available(*, host: str, port: int) -> None:
    """검증용 Uvicorn이 사용할 Port 충돌을 사전에 확인한다."""

    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as current_socket:
        current_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            current_socket.bind((host, port))
        except OSError as exc:
            raise RuntimeError(
                f"Validation port is already in use: {host}:{port}"
            ) from exc


def validate_prediction_payload(
    payload: Mapping[str, Any],
    *,
    expected_prediction: int,
    expected_class_name: str,
) -> None:
    """실제 HTTP Prediction 응답의 핵심 계약을 검증한다."""

    required_keys = {
        "prediction",
        "prediction_class_name",
        "defect_probability",
        "normal_probability",
        "raw_logit",
        "classification_threshold",
        "model_name",
        "model_version",
        "positive_class",
        "original_filename",
        "content_type",
        "image_width",
        "image_height",
        "image_mode",
        "inference_time_ms",
    }
    missing = required_keys - set(payload)
    if missing:
        raise KeyError(f"Prediction response is missing keys: {sorted(missing)}")

    if payload["prediction"] != expected_prediction:
        raise ValueError(
            "Unexpected prediction. "
            f"Expected: {expected_prediction}. Received: {payload['prediction']}."
        )

    if payload["prediction_class_name"] != expected_class_name:
        raise ValueError("Unexpected prediction_class_name")

    defect_probability = float(payload["defect_probability"])
    normal_probability = float(payload["normal_probability"])
    raw_logit = float(payload["raw_logit"])
    inference_time_ms = float(payload["inference_time_ms"])

    for name, value in {
        "defect_probability": defect_probability,
        "normal_probability": normal_probability,
        "raw_logit": raw_logit,
        "inference_time_ms": inference_time_ms,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if not 0.0 <= defect_probability <= 1.0:
        raise ValueError("defect_probability must be in [0, 1]")

    if not 0.0 <= normal_probability <= 1.0:
        raise ValueError("normal_probability must be in [0, 1]")

    if not math.isclose(
        defect_probability + normal_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("normal and defect probabilities must sum to 1")

    if float(payload["classification_threshold"]) != 0.5:
        raise ValueError("classification_threshold must be 0.5")

    if payload["positive_class"] != "DEFECT":
        raise ValueError("positive_class must be DEFECT")


def write_json_atomically(*, payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")

    try:
        temporary_path.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _wait_until_ready(
    *,
    client: httpx.Client,
    health_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            response = client.get(health_url)
            if response.status_code == 200:
                payload = response.json()
                if payload.get("model_loaded") is True:
                    return dict(payload)
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc

        time.sleep(0.25)

    raise RuntimeError(
        "FastAPI did not become ready with model_loaded=true."
    ) from last_error


def _post_image(
    *,
    client: httpx.Client,
    prediction_url: str,
    image_path: Path,
) -> dict[str, Any]:
    with image_path.open("rb") as image_file:
        response = client.post(
            prediction_url,
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Prediction response JSON must be an object")
    return payload


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = parse_arguments(arguments)

    normal_image_path = resolve_project_path(parsed.normal_image_path)
    defect_image_path = resolve_project_path(parsed.defect_image_path)
    output_path = resolve_project_path(parsed.output_path)

    for name, path in {
        "normal_image_path": normal_image_path,
        "defect_image_path": defect_image_path,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    ensure_port_available(host=parsed.host, port=parsed.port)

    base_url = f"http://{parsed.host}:{parsed.port}"
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.app:app",
        "--host",
        parsed.host,
        "--port",
        str(parsed.port),
        "--log-level",
        "warning",
    ]

    started_at = time.perf_counter()

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            with httpx.Client(timeout=30.0) as client:
                health_payload = _wait_until_ready(
                    client=client,
                    health_url=f"{base_url}/api/v1/health",
                    timeout_seconds=45.0,
                )

                normal_payload = _post_image(
                    client=client,
                    prediction_url=f"{base_url}/api/v1/predictions",
                    image_path=normal_image_path,
                )
                validate_prediction_payload(
                    normal_payload,
                    expected_prediction=0,
                    expected_class_name="NORMAL",
                )

                defect_payload = _post_image(
                    client=client,
                    prediction_url=f"{base_url}/api/v1/predictions",
                    image_path=defect_image_path,
                )
                validate_prediction_payload(
                    defect_payload,
                    expected_prediction=1,
                    expected_class_name="DEFECT",
                )

            runtime_seconds = time.perf_counter() - started_at

            artifact = {
                "project": "Manufacturing Vision Defect Analysis System",
                "run_name": "day7_fastapi_inference_validation",
                "base_url": base_url,
                "health": health_payload,
                "normal_image": {
                    "path": normal_image_path.relative_to(PROJECT_ROOT).as_posix(),
                    "response": normal_payload,
                },
                "defect_image": {
                    "path": defect_image_path.relative_to(PROJECT_ROOT).as_posix(),
                    "response": defect_payload,
                },
                "runtime_seconds": runtime_seconds,
            }

            write_json_atomically(payload=artifact, path=output_path)

            print("=" * 100)
            print("DAY 7 - FASTAPI REAL HTTP VALIDATION")
            print("=" * 100)
            print(f"Health model loaded        : {health_payload['model_loaded']}")
            print(
                "NORMAL prediction          : "
                f"{normal_payload['prediction_class_name']}"
            )
            print(
                "NORMAL P(DEFECT)            : "
                f"{normal_payload['defect_probability']:.12f}"
            )
            print(
                "DEFECT prediction          : "
                f"{defect_payload['prediction_class_name']}"
            )
            print(
                "DEFECT P(DEFECT)            : "
                f"{defect_payload['defect_probability']:.12f}"
            )
            print(f"Runtime seconds            : {runtime_seconds:.2f}")
            print(f"Artifact                   : {output_path}")
            print()
            print("[PASS] Day 7 FastAPI real HTTP validation")

        except Exception:
            log_file.seek(0)
            server_log = log_file.read()
            if server_log:
                print("[UVICORN LOG]")
                print(server_log[-5000:])
            raise

        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
