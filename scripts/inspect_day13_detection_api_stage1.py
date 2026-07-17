"""Day 13 Detection API Stage 1 적용 상태와 Best Checkpoint 계약을 점검한다."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

from src.api.detection_config import DetectionApiSettings
from src.detection.checkpoint import load_detection_checkpoint_payload
from src.detection.model_config import DetectionModelConfig


REQUIRED_FILES = (
    Path("src/api/detection_config.py"),
    Path("src/api/detection_inference_service.py"),
    Path("src/api/detection_model_loader.py"),
    Path("src/api/schemas.py"),
    Path("src/api/app.py"),
    Path("tests/test_detection_api_config.py"),
    Path("tests/test_detection_inference_service.py"),
    Path("tests/test_detection_model_loader.py"),
    Path("tests/test_detection_prediction_api.py"),
)

DEFAULT_ARTIFACT = Path(
    "reports/artifacts/day13_detection_api_stage1_inspection.json"
)


def canonical_expected_class_mapping(
    index_to_class: Mapping[int, str],
) -> dict[str, int]:
    """Model Config Mapping을 Checkpoint 비교용 Canonical 형식으로 변환한다.

    Day 11 Model Config는 기존 Dataset Mapping에 따라 Label 0 이름을
    ``background``로 노출할 수 있다. Day 12 Checkpoint 저장 정책은
    Background Label을 ``BACKGROUND=0``으로 고정한다.

    따라서 비교 경계에서는 Label 0만 ``BACKGROUND``로 정규화하고,
    Label 1~6의 NEU-DET Class 이름과 번호는 그대로 유지한다.
    """

    if not isinstance(index_to_class, Mapping):
        raise TypeError("index_to_class must be a mapping.")
    if not index_to_class:
        raise ValueError("index_to_class must not be empty.")

    normalized: dict[str, int] = {}

    for raw_index, raw_name in index_to_class.items():
        if (
            not isinstance(raw_index, int)
            or isinstance(raw_index, bool)
            or raw_index < 0
        ):
            raise ValueError(
                "Every index_to_class key must be a non-negative int."
            )
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(
                "Every index_to_class value must be a non-empty str."
            )

        canonical_name = (
            "BACKGROUND"
            if raw_index == 0
            else raw_name
        )
        if canonical_name in normalized:
            raise ValueError(
                "index_to_class contains duplicate canonical class names."
            )
        normalized[canonical_name] = raw_index

    if normalized.get("BACKGROUND") != 0:
        raise ValueError(
            "index_to_class must contain the background label at index 0."
        )
    if set(normalized.values()) != set(range(len(normalized))):
        raise ValueError(
            "index_to_class indexes must be contiguous from 0."
        )

    return normalized


def run_inspection(
    *,
    project_root: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    settings = DetectionApiSettings()
    checkpoint_path = settings.checkpoint_path

    payload = load_detection_checkpoint_payload(
        checkpoint_path,
        map_location="cpu",
    )
    model_config = DetectionModelConfig()

    # Loader Hotfix 1과 같은 Background Canonicalization 정책을 사용한다.
    expected_mapping = canonical_expected_class_mapping(
        model_config.index_to_class
    )
    checkpoint_mapping = {
        str(class_name): int(class_index)
        for class_name, class_index in dict(
            payload["class_mapping"]
        ).items()
    }

    file_checks = {
        path.as_posix(): (root / path).is_file()
        for path in REQUIRED_FILES
    }
    checks = {
        "required_files_exist": all(file_checks.values()),
        "checkpoint_exists": checkpoint_path.is_file(),
        "checkpoint_epoch_is_2": payload["epoch"] == 2,
        "checkpoint_best_metric_is_finite": math.isfinite(
            float(payload["best_metric"]),
        ),
        "class_mapping_matches": (
            checkpoint_mapping == expected_mapping
        ),
        "architecture_matches": (
            settings.architecture == model_config.architecture
        ),
        "num_classes_is_7": model_config.num_classes == 7,
        "score_threshold_is_0_5": (
            settings.default_score_threshold == 0.5
        ),
        "iou_threshold_is_0_5": (
            settings.iou_threshold == 0.5
        ),
    }

    result: dict[str, Any] = {
        "stage": "day13_detection_api_stage1_inspection",
        "project_root": str(root),
        "files": file_checks,
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "epoch_index": int(payload["epoch"]),
            "epoch_number": int(payload["epoch"]) + 1,
            "best_metric": float(payload["best_metric"]),
            "class_mapping": checkpoint_mapping,
        },
        "mapping_validation": {
            "policy": (
                "Label 0 is canonicalized to BACKGROUND; "
                "NEU-DET labels 1-6 remain unchanged."
            ),
            "expected_class_mapping": expected_mapping,
            "checkpoint_class_mapping": checkpoint_mapping,
            "matches": checkpoint_mapping == expected_mapping,
        },
        "api_policy": {
            "endpoint": "/api/v1/detection/predictions",
            "default_score_threshold": (
                settings.default_score_threshold
            ),
            "minimum_score_threshold": (
                settings.minimum_score_threshold
            ),
            "maximum_score_threshold": (
                settings.maximum_score_threshold
            ),
            "iou_threshold": settings.iou_threshold,
            "device": settings.device,
            "architecture": settings.architecture,
        },
        "checks": checks,
        "validation_passed": all(checks.values()),
    }

    output = (
        artifact_path
        if artifact_path.is_absolute()
        else root / artifact_path
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("DAY 13 - DETECTION API STAGE 1 INSPECTION")
    print("=" * 100)
    print(f"Project root             : {root}")
    print(f"Checkpoint               : {checkpoint_path}")
    print(
        "Checkpoint epoch         : "
        f"{int(payload['epoch']) + 1}"
    )
    print(
        "Best validation mAP@0.50 : "
        f"{float(payload['best_metric']):.6f}"
    )
    print(
        "Endpoint                 : "
        "/api/v1/detection/predictions"
    )
    print(
        "Score threshold          : "
        f"{settings.default_score_threshold:.2f}"
    )
    print(
        "IoU threshold            : "
        f"{settings.iou_threshold:.2f}"
    )

    for name, passed in checks.items():
        print(
            f"[{'PASS' if passed else 'FAIL'}] "
            f"{name}"
        )

    if not checks["class_mapping_matches"]:
        print(
            "[MAPPING EXPECTED]        : "
            f"{expected_mapping}"
        )
        print(
            "[MAPPING CHECKPOINT]      : "
            f"{checkpoint_mapping}"
        )

    print(f"Artifact                 : {output}")
    print(
        "[RESULT]                 : "
        + (
            "PASS"
            if result["validation_passed"]
            else "FAIL"
        )
    )
    print("=" * 100)

    if not result["validation_passed"]:
        raise RuntimeError(
            "Day 13 Detection API Stage 1 inspection failed."
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_ARTIFACT,
    )
    args = parser.parse_args()

    run_inspection(
        project_root=args.project_root,
        artifact_path=args.artifact_path,
    )


if __name__ == "__main__":
    main()
