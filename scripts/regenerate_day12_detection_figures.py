"""Day 12 Detection Figure V2 재생성 Script.

이 Script는 이미 확정된 Best Checkpoint와 평가 Artifact를 그대로 사용한다.
Test Prediction을 다시 계산하는 이유는 기존 JSON Artifact에 원본 Box Tensor를
저장하지 않았기 때문이다.

중요 정책
---------
- Checkpoint를 변경하지 않는다.
- 평가·실패 분석 JSON을 덮어쓰지 않는다.
- Test 결과를 모델 선택이나 추가 학습에 사용하지 않는다.
- 기존 Figure는 reports/backups/day12_visualization_v2 아래에 백업한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import urlparse

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)

from src.detection.checkpoint import load_detection_checkpoint_payload
from src.detection.data_loader import detection_collate_fn
from src.detection.detection_dataset import NeuDetDetectionDataset
from src.detection.epoch_runner import run_detection_evaluation_epoch
from src.detection.failure_analysis import analyze_detection_failures
from src.detection.model_config import DetectionModelConfig
from src.detection.model_factory import create_detection_model
from src.detection.training_visualization import (
    create_detection_failure_montage,
    create_detection_prediction_montage,
    plot_detection_class_metrics,
    plot_detection_training_history,
)
from src.detection.transforms import create_detection_transform


DEFAULT_MANIFEST = Path("data/processed/neu_det/splits.json")
DEFAULT_BEST_CHECKPOINT = Path("models/detection/day12_detection_best.pt")
DEFAULT_EVALUATION_ARTIFACT = Path(
    "reports/artifacts/day12_detection_evaluation.json"
)
DEFAULT_FAILURE_ARTIFACT = Path(
    "reports/artifacts/day12_detection_failure_analysis.json"
)
DEFAULT_BACKUP_DIRECTORY = Path("reports/backups/day12_visualization_v2")
FIGURE_PATHS = {
    "training_history": Path(
        "reports/figures/day12_detection_training_history.png"
    ),
    "class_metrics": Path(
        "reports/figures/day12_detection_class_metrics.png"
    ),
    "predictions": Path(
        "reports/figures/day12_detection_predictions.png"
    ),
    "failure_analysis": Path(
        "reports/figures/day12_detection_failure_analysis.png"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON Artifact does not exist: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}.")
    return payload


def _torch_cache_checkpoint_directory() -> Path:
    torch_home = os.getenv("TORCH_HOME")
    root = (
        Path(torch_home).expanduser().resolve()
        if torch_home
        else (Path.home() / ".cache" / "torch").resolve()
    )
    return root / "hub" / "checkpoints"


def _expected_weight_path() -> Path:
    weight = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    filename = Path(urlparse(weight.url).path).name
    return _torch_cache_checkpoint_directory() / filename


def _create_loader(dataset: NeuDetDetectionDataset) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
        collate_fn=detection_collate_fn,
    )


def _backup_existing_figures(project_root: Path) -> list[Path]:
    backup_directory = project_root / DEFAULT_BACKUP_DIRECTORY
    backup_directory.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name, relative_path in FIGURE_PATHS.items():
        source = project_root / relative_path
        destination = backup_directory / f"{name}.before_v2.png"
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
            copied.append(destination)
    return copied


def _verify_metric_consistency(
    *,
    expected: dict[str, Any],
    actual: dict[str, Any],
    tolerance: float = 1e-6,
) -> None:
    expected_overall = expected["test"]["metrics"]["overall"]
    actual_overall = actual["metrics"]["overall"]
    for key in ("precision", "recall", "f1", "map_50"):
        expected_value = expected_overall[key]
        actual_value = actual_overall[key]
        if expected_value is None or actual_value is None:
            if expected_value is not actual_value:
                raise ValueError(f"Metric mismatch for {key}.")
            continue
        if abs(float(expected_value) - float(actual_value)) > tolerance:
            raise ValueError(
                f"Recomputed Test metric does not match Artifact: {key}."
            )


def regenerate_day12_detection_figures(
    *,
    project_root: Path,
    torch_num_threads: int = 2,
    log_interval: int = 50,
) -> dict[str, Any]:
    """확정된 평가 결과를 변경하지 않고 Figure만 V2로 재생성한다."""
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path.")
    for name, value in (
        ("torch_num_threads", torch_num_threads),
        ("log_interval", log_interval),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive int.")

    project_root = project_root.resolve()
    manifest_path = project_root / DEFAULT_MANIFEST
    checkpoint_path = project_root / DEFAULT_BEST_CHECKPOINT
    evaluation_path = project_root / DEFAULT_EVALUATION_ARTIFACT
    failure_path = project_root / DEFAULT_FAILURE_ARTIFACT
    for path, description in (
        (manifest_path, "Split manifest"),
        (checkpoint_path, "Best checkpoint"),
        (evaluation_path, "Evaluation Artifact"),
        (failure_path, "Failure Artifact"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}.")
    if not _expected_weight_path().is_file():
        raise FileNotFoundError("COCO pretrained Detection weight is not cached.")

    evaluation_payload = _read_json(evaluation_path)
    failure_payload = _read_json(failure_path)
    score_threshold = float(
        evaluation_payload["evaluation_policy"]["score_threshold"]
    )
    iou_threshold = float(
        evaluation_payload["evaluation_policy"]["iou_threshold"]
    )
    checkpoint_payload = load_detection_checkpoint_payload(
        checkpoint_path,
        map_location="cpu",
    )
    model_config = DetectionModelConfig(
        min_size=320,
        max_size=320,
        use_pretrained_weights=True,
        use_pretrained_backbone=False,
        progress=False,
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(torch_num_threads)
    try:
        dataset = NeuDetDetectionDataset.from_manifest(
            manifest_path=manifest_path,
            project_root=project_root,
            split="test",
            transform=create_detection_transform(training=False),
            duplicate_box_policy="preserve",
        )
        loader = _create_loader(dataset)
        model_result = create_detection_model(
            config=model_config,
            device="cpu",
            training=False,
            proposal_limits=None,
        )
        model = model_result.model
        model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
        model.eval()

        print("=" * 100)
        print("DAY 12 - DETECTION FIGURE V2 REGENERATION")
        print("=" * 100)
        print(f"Project root             : {project_root}")
        print(f"Best checkpoint epoch    : {checkpoint_payload['epoch']}")
        print(f"Test images              : {len(dataset)}")
        print(f"Score threshold          : {score_threshold:.2f}")
        print(f"IoU threshold            : {iou_threshold:.2f}")
        print("[FIGURE ONLY] [CHECKPOINT AND JSON ARTIFACTS ARE NOT MODIFIED]")

        def progress(payload: dict[str, Any]) -> None:
            if payload.get("event") != "evaluation_progress":
                return
            print(
                "[TEST FIGURE REBUILD] "
                f"batch={payload['batch_count']} "
                f"samples={payload['sample_count']} "
                f"boxes={payload['prediction_box_count']} "
                f"elapsed={payload['elapsed_seconds']:.2f}s",
                flush=True,
            )

        result = run_detection_evaluation_epoch(
            model=model,
            data_loader=loader,
            split="test",
            num_classes=model_config.num_classes,
            index_to_class=model_config.index_to_class,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            device="cpu",
            log_interval=log_interval,
            progress_callback=progress,
        )
        _verify_metric_consistency(
            expected=evaluation_payload,
            actual=result.summary(),
        )
        sample_ids = [sample.record_id for sample in dataset.samples]
        analysis = analyze_detection_failures(
            predictions=result.predictions,
            targets=result.targets,
            index_to_class=model_config.index_to_class,
            sample_ids=sample_ids,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )
        expected_event_count = int(
            failure_payload["analysis"]["summary"]["event_count"]
        )
        if int(analysis["summary"]["event_count"]) != expected_event_count:
            raise ValueError("Recomputed failure event count does not match Artifact.")

        backups = _backup_existing_figures(project_root)
        paths = {name: project_root / relative for name, relative in FIGURE_PATHS.items()}
        plot_detection_training_history(
            history=checkpoint_payload["history"],
            output_path=paths["training_history"],
        )
        plot_detection_class_metrics(
            class_metrics=evaluation_payload["test"]["metrics"]["class_metrics"],
            output_path=paths["class_metrics"],
        )
        create_detection_prediction_montage(
            dataset=dataset,
            predictions=result.predictions,
            targets=result.targets,
            index_to_class=model_config.index_to_class,
            output_path=paths["predictions"],
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_images=4,
            max_predictions_per_image=8,
        )
        create_detection_failure_montage(
            dataset=dataset,
            predictions=result.predictions,
            targets=result.targets,
            failure_analysis=analysis,
            index_to_class=model_config.index_to_class,
            output_path=paths["failure_analysis"],
            max_images=6,
        )

        print("-" * 100)
        print(f"Backed-up figures        : {len(backups)}")
        for name, path in paths.items():
            with Image.open(path) as image:
                width, height = image.size
            print(
                f"{name:24}: {path} "
                f"({width}x{height}, {path.stat().st_size / 1024:.1f} KiB)"
            )
        print("[RESULT]                 : PASS")
        print("[FIGURE V2 ONLY] [METRICS, CHECKPOINT, ARTIFACTS UNCHANGED]")
        return {
            "result": "PASS",
            "backup_count": len(backups),
            "figures": {name: str(path) for name, path in paths.items()},
        }
    finally:
        torch.set_num_threads(previous_threads)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate Day 12 Detection figures with readable V2 layout."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    regenerate_day12_detection_figures(
        project_root=arguments.project_root,
        torch_num_threads=arguments.torch_num_threads,
        log_interval=arguments.log_interval,
    )


if __name__ == "__main__":
    main()
