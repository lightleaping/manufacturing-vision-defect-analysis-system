"""실제 Day 12 Best Checkpoint로 Detection FastAPI Endpoint를 Smoke 검증한다.

단위·통합 테스트가 먼저 통과한 뒤 실행한다. 사용자가 지정한 JPEG 또는 PNG 한 장을
실제 Production Detection Service로 추론하고 JSON Artifact를 저장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.detection_model_loader import (
    create_production_detection_inference_service,
)


DEFAULT_ARTIFACT = Path(
    "reports/artifacts/day13_detection_api_smoke_test.json"
)


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    raise ValueError("image_path must use .jpg, .jpeg, or .png extension.")


def run_smoke_test(
    *,
    project_root: Path,
    image_path: Path,
    score_threshold: float,
    artifact_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    image = image_path if image_path.is_absolute() else root / image_path
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Smoke image does not exist: {image}.")

    service = create_production_detection_inference_service()
    application = create_app(
        service_factory=None,
        detection_service_factory=lambda: service,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/detection/predictions",
            params={"score_threshold": score_threshold},
            files={
                "file": (
                    image.name,
                    image.read_bytes(),
                    _content_type(image),
                )
            },
        )

    try:
        response_payload = response.json()
    except ValueError as error:
        raise RuntimeError("Detection API returned invalid JSON.") from error

    result: dict[str, Any] = {
        "stage": "day13_detection_api_smoke_test",
        "request": {
            "endpoint": "/api/v1/detection/predictions",
            "image_path": str(image),
            "score_threshold": score_threshold,
        },
        "response": {
            "status_code": response.status_code,
            "payload": response_payload,
        },
        "validation_passed": bool(
            response.status_code == 200
            and isinstance(response_payload, dict)
            and response_payload.get("checkpoint_epoch") == 3
            and response_payload.get("architecture")
            == "fasterrcnn_mobilenet_v3_large_320_fpn"
            and response_payload.get("detection_count")
            == len(response_payload.get("detections", []))
        ),
    }

    output = artifact_path if artifact_path.is_absolute() else root / artifact_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("DAY 13 - REAL BEST CHECKPOINT DETECTION API SMOKE TEST")
    print("=" * 100)
    print(f"Image                    : {image}")
    print(f"Score threshold          : {score_threshold:.2f}")
    print(f"HTTP status              : {response.status_code}")
    print(f"Detection count          : {response_payload.get('detection_count')}")
    print(f"Inference time (ms)      : {response_payload.get('inference_time_ms')}")
    print(f"Artifact                 : {output}")
    print("[RESULT]                 : " + ("PASS" if result["validation_passed"] else "FAIL"))
    print("=" * 100)

    if not result["validation_passed"]:
        raise RuntimeError("Day 13 Detection API smoke test failed.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    run_smoke_test(
        project_root=args.project_root,
        image_path=args.image_path,
        score_threshold=args.score_threshold,
        artifact_path=args.artifact_path,
    )


if __name__ == "__main__":
    main()
