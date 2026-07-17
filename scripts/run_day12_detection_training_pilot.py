"""COCO pretrained Faster R-CNN으로 Day 12 CPU Pilot Training을 실행한다.

기본 실행은 Cache에 Detection Weight가 없으면 중단한다. 최초 다운로드를
허용하려면 반드시 ``--allow-pretrained-download``를 명시해야 한다.

실행 예:
    python -m scripts.run_day12_detection_training_pilot `
        --allow-pretrained-download `
        --overfit-steps 3
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any
from urllib.parse import urlparse

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)

from src.detection.checkpoint import (
    build_detection_checkpoint_payload,
    save_detection_checkpoint,
)
from src.detection.data_loader import detection_collate_fn
from src.detection.detection_dataset import NeuDetDetectionDataset
from src.detection.metrics import calculate_detection_metrics
from src.detection.model_config import DetectionModelConfig
from src.detection.model_factory import create_detection_model
from src.detection.optimization import (
    build_detection_optimization,
    set_detection_backbone_trainable,
)
from src.detection.trainer import (
    run_detection_inference_step,
    run_tiny_overfit_diagnostic,
)
from src.detection.training_config import DetectionTrainingConfig
from src.detection.transforms import create_detection_transform


DEFAULT_MANIFEST = Path("data/processed/neu_det/splits.json")
DEFAULT_CONFIG_ARTIFACT = Path(
    "reports/artifacts/day12_detection_training_config.json"
)
DEFAULT_PILOT_ARTIFACT = Path(
    "reports/artifacts/day12_detection_training_pilot.json"
)
DEFAULT_PILOT_CHECKPOINT = Path(
    "models/detection/pilot/day12_detection_pilot.pt"
)
MINIMUM_FREE_GIB = 3.0
FULL_TRAIN_IMAGE_COUNT = 1440
FULL_VALIDATION_IMAGE_COUNT = 178


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int.")


def _canonical_class_mapping(
    index_to_class: Mapping[int, str],
) -> dict[str, int]:
    """Model Label Mapping을 Checkpoint의 고정 계약으로 정규화한다.

    일부 기존 구성은 Background 이름을 ``background``로 반환할 수 있다.
    Detection Label 0의 의미는 같으므로 저장 경계에서만 ``BACKGROUND``로
    통일한다. 나머지 결함 Class 이름은 원본 Canonical 이름을 유지한다.
    """
    if not isinstance(index_to_class, Mapping):
        raise TypeError("index_to_class must be a mapping.")

    class_mapping: dict[str, int] = {}
    for class_index, class_name in sorted(index_to_class.items()):
        if (
            not isinstance(class_index, int)
            or isinstance(class_index, bool)
            or class_index < 0
        ):
            raise ValueError(
                "Every index_to_class key must be a non-negative int."
            )
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(
                "Every index_to_class value must be a non-empty str."
            )

        normalized_name = (
            "BACKGROUND" if class_index == 0 else class_name
        )
        if normalized_name in class_mapping:
            raise ValueError(
                f"Duplicate normalized class name: {normalized_name}."
            )
        class_mapping[normalized_name] = class_index

    if class_mapping.get("BACKGROUND") != 0:
        raise ValueError(
            "index_to_class must contain the background label at index 0."
        )
    return class_mapping


def _create_subset_loader(
    *,
    dataset: NeuDetDetectionDataset,
    sample_count: int,
    batch_size: int,
) -> DataLoader[Any]:
    if sample_count > len(dataset):
        raise ValueError("sample_count exceeds dataset length.")
    subset = Subset(dataset, list(range(sample_count)))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
        collate_fn=detection_collate_fn,
    )


def run_day12_detection_training_pilot(
    *,
    project_root: Path,
    allow_pretrained_download: bool = False,
    train_samples: int = 1,
    validation_samples: int = 1,
    overfit_steps: int = 3,
    torch_num_threads: int = 2,
) -> dict[str, Any]:
    """Weight 준비·실제 Backward·Inference·Checkpoint를 작은 범위에서 검증한다."""
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path.")
    if not isinstance(allow_pretrained_download, bool):
        raise TypeError("allow_pretrained_download must be bool.")
    for name, value in (
        ("train_samples", train_samples),
        ("validation_samples", validation_samples),
        ("overfit_steps", overfit_steps),
        ("torch_num_threads", torch_num_threads),
    ):
        _validate_positive_int(name, value)

    project_root = project_root.resolve()
    manifest_path = project_root / DEFAULT_MANIFEST
    config_artifact = project_root / DEFAULT_CONFIG_ARTIFACT
    pilot_artifact = project_root / DEFAULT_PILOT_ARTIFACT
    pilot_checkpoint = project_root / DEFAULT_PILOT_CHECKPOINT
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}.")

    disk_free_gib_before = shutil.disk_usage(project_root).free / (1024**3)
    if disk_free_gib_before < MINIMUM_FREE_GIB:
        raise OSError(
            f"At least {MINIMUM_FREE_GIB:.1f} GiB free space is required. "
            f"Current: {disk_free_gib_before:.3f} GiB."
        )

    expected_weight_path = _expected_weight_path()
    cache_existed_before = expected_weight_path.is_file()
    if not cache_existed_before and not allow_pretrained_download:
        raise RuntimeError(
            "Detection pretrained weight is not cached. Re-run with "
            "--allow-pretrained-download after confirming disk and network policy."
        )

    training_config = DetectionTrainingConfig(
        batch_size=1,
        epochs=1,
        freeze_backbone_epochs=1,
        horizontal_flip_probability=0.0,
        torch_num_threads=torch_num_threads,
    )
    model_config = DetectionModelConfig(
        min_size=training_config.min_size,
        max_size=training_config.max_size,
        use_pretrained_weights=True,
        use_pretrained_backbone=False,
        progress=True,
    )
    class_mapping = _canonical_class_mapping(
        model_config.index_to_class
    )
    _write_json(
        config_artifact,
        {
            "schema_version": 1,
            "stage": "day12_training_pilot",
            "training_config": training_config.to_dict(),
            "model_config": {
                "architecture": model_config.architecture,
                "num_classes_with_background": model_config.num_classes,
                "class_to_index": class_mapping,
                "min_size": model_config.min_size,
                "max_size": model_config.max_size,
                "use_pretrained_weights": model_config.use_pretrained_weights,
            },
            "execution_scope": {
                "train_samples": train_samples,
                "validation_samples": validation_samples,
                "overfit_steps": overfit_steps,
                "full_training_executed": False,
                "test_split_evaluation_executed": False,
            },
        },
    )

    torch.manual_seed(training_config.random_seed)
    previous_num_threads = torch.get_num_threads()
    torch.set_num_threads(torch_num_threads)
    pilot_started = time.perf_counter()

    try:
        train_dataset = NeuDetDetectionDataset.from_manifest(
            manifest_path=manifest_path,
            project_root=project_root,
            split="train",
            transform=create_detection_transform(
                training=True,
                horizontal_flip_probability=0.0,
            ),
            duplicate_box_policy=training_config.duplicate_box_policy,
        )
        validation_dataset = NeuDetDetectionDataset.from_manifest(
            manifest_path=manifest_path,
            project_root=project_root,
            split="validation",
            transform=create_detection_transform(training=False),
            duplicate_box_policy=training_config.duplicate_box_policy,
        )
        train_loader = _create_subset_loader(
            dataset=train_dataset,
            sample_count=train_samples,
            batch_size=training_config.batch_size,
        )
        validation_loader = _create_subset_loader(
            dataset=validation_dataset,
            sample_count=validation_samples,
            batch_size=training_config.batch_size,
        )

        model_build_started = time.perf_counter()
        model_result = create_detection_model(
            config=model_config,
            device="cpu",
            training=True,
            proposal_limits=None,
        )
        model_build_seconds = time.perf_counter() - model_build_started
        model = model_result.model

        freeze_metadata = set_detection_backbone_trainable(
            model,
            trainable=False,
        )
        optimization = build_detection_optimization(
            model=model,
            config=training_config,
        )

        train_images, train_targets = next(iter(train_loader))
        overfit = run_tiny_overfit_diagnostic(
            model=model,
            optimizer=optimization.optimizer,
            images=train_images,
            targets=train_targets,
            steps=overfit_steps,
            device="cpu",
        )
        if optimization.scheduler is not None:
            optimization.scheduler.step()

        validation_predictions: list[dict[str, torch.Tensor]] = []
        validation_targets: list[dict[str, torch.Tensor]] = []
        inference_summaries: list[dict[str, Any]] = []
        for images, targets in validation_loader:
            inference = run_detection_inference_step(
                model=model,
                images=images,
                targets=targets,
                num_classes=model_config.num_classes,
                device="cpu",
            )
            validation_predictions.extend(inference.predictions)
            validation_targets.extend(
                [
                    {
                        "boxes": target["boxes"].detach().cpu().clone(),
                        "labels": target["labels"].detach().cpu().clone(),
                    }
                    for target in targets
                ]
            )
            inference_summaries.append(inference.summary())

        metrics = calculate_detection_metrics(
            predictions=validation_predictions,
            targets=validation_targets,
            index_to_class=model_config.index_to_class,
            score_threshold=training_config.score_threshold,
            iou_threshold=training_config.iou_threshold,
        )
        map_50 = metrics["overall"]["map_50"]
        best_metric = 0.0 if map_50 is None else float(map_50)

        history_entry = {
            "epoch": 0,
            "stage": "pilot",
            "train_total_losses": [
                item["total_loss"] for item in overfit["step_results"]
            ],
            "validation_map_50": best_metric,
            "backbone_trainable": False,
        }
        checkpoint_payload = build_detection_checkpoint_payload(
            epoch=0,
            model=model,
            optimizer=optimization.optimizer,
            scheduler=optimization.scheduler,
            training_config=training_config.to_dict(),
            class_mapping=class_mapping,
            best_metric=best_metric,
            history=[history_entry],
        )
        saved_checkpoint, _ = save_detection_checkpoint(
            payload=checkpoint_payload,
            latest_path=pilot_checkpoint,
            best_path=None,
            is_best=False,
        )

        average_train_step_seconds = float(overfit["average_step_seconds"])
        average_inference_batch_seconds = sum(
            item["elapsed_seconds"] for item in inference_summaries
        ) / len(inference_summaries)
        estimated_train_batches = math.ceil(
            FULL_TRAIN_IMAGE_COUNT / training_config.batch_size
        )
        estimated_validation_batches = math.ceil(
            FULL_VALIDATION_IMAGE_COUNT / training_config.batch_size
        )
        estimated_epoch_seconds = (
            average_train_step_seconds * estimated_train_batches
            + average_inference_batch_seconds * estimated_validation_batches
        )

        cache_exists_after = expected_weight_path.is_file()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(project_root),
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
                "device": "cpu",
                "torch_num_threads": torch_num_threads,
            },
            "execution_policy": {
                "allow_pretrained_download": allow_pretrained_download,
                "pretrained_cache_existed_before": cache_existed_before,
                "pretrained_cache_exists_after": cache_exists_after,
                "full_training_executed": False,
                "test_split_evaluation_executed": False,
                "proposal_limits_applied": False,
                "augmentation_probability": 0.0,
            },
            "storage": {
                "disk_free_gib_before": round(disk_free_gib_before, 3),
                "disk_free_gib_after": round(
                    shutil.disk_usage(project_root).free / (1024**3),
                    3,
                ),
                "expected_weight_path": str(expected_weight_path),
                "weight_size_mib": (
                    round(expected_weight_path.stat().st_size / (1024**2), 3)
                    if expected_weight_path.is_file()
                    else None
                ),
                "checkpoint_path": _relative_or_absolute(
                    saved_checkpoint,
                    project_root,
                ),
                "checkpoint_size_mib": round(
                    saved_checkpoint.stat().st_size / (1024**2),
                    3,
                ),
            },
            "data": {
                "train_dataset_size": len(train_dataset),
                "validation_dataset_size": len(validation_dataset),
                "pilot_train_samples": train_samples,
                "pilot_validation_samples": validation_samples,
                "train_record_ids": [
                    train_dataset.samples[index].record_id
                    for index in range(train_samples)
                ],
                "validation_record_ids": [
                    validation_dataset.samples[index].record_id
                    for index in range(validation_samples)
                ],
            },
            "model": model_result.metadata,
            "freeze": freeze_metadata,
            "optimization": optimization.metadata,
            "tiny_overfit": overfit,
            "validation_inference": inference_summaries,
            "validation_metrics": metrics,
            "timing": {
                "model_build_and_weight_prepare_seconds": round(
                    model_build_seconds,
                    6,
                ),
                "pilot_total_seconds": round(
                    time.perf_counter() - pilot_started,
                    6,
                ),
                "estimated_full_epoch_seconds": round(
                    estimated_epoch_seconds,
                    3,
                ),
                "estimated_full_epoch_minutes": round(
                    estimated_epoch_seconds / 60.0,
                    3,
                ),
                "estimate_note": (
                    "Single-batch CPU extrapolation; actual full epoch can differ "
                    "because of data loading, box counts, cache, and later unfreeze."
                ),
            },
            "artifacts": {
                "training_config": _relative_or_absolute(
                    config_artifact,
                    project_root,
                ),
                "pilot_analysis": _relative_or_absolute(
                    pilot_artifact,
                    project_root,
                ),
                "pilot_checkpoint": _relative_or_absolute(
                    saved_checkpoint,
                    project_root,
                ),
            },
            "validation_passed": bool(
                cache_exists_after
                and overfit["all_losses_finite"]
                and all(
                    item["inputs_unchanged"]
                    for item in overfit["step_results"]
                )
                and all(
                    item["inputs_unchanged"] for item in inference_summaries
                )
                and saved_checkpoint.is_file()
            ),
        }
        _write_json(pilot_artifact, payload)
    finally:
        torch.set_num_threads(previous_num_threads)

    print("=" * 100)
    print("DAY 12 - DETECTION PRETRAINED CPU TRAINING PILOT")
    print("=" * 100)
    print(f"Project root             : {project_root}")
    print(f"Disk free before         : {disk_free_gib_before:.3f} GiB")
    print(f"Weight cached before     : {cache_existed_before}")
    print(f"Weight cached after      : {payload['execution_policy']['pretrained_cache_exists_after']}")
    print(f"Weight size              : {payload['storage']['weight_size_mib']} MiB")
    print(f"Model build / prepare    : {payload['timing']['model_build_and_weight_prepare_seconds']:.3f}s")
    print(f"Backbone trainable       : {payload['freeze']['backbone_trainable']}")
    print(f"Trainable parameters     : {payload['freeze']['trainable_parameters']:,}")
    print(f"Pilot train samples      : {train_samples}")
    print(f"Tiny overfit steps       : {overfit_steps}")
    print(f"Initial total loss       : {payload['tiny_overfit']['initial_total_loss']:.6f}")
    print(f"Final total loss         : {payload['tiny_overfit']['final_total_loss']:.6f}")
    print(f"Loss decrease observed   : {payload['tiny_overfit']['loss_decrease_observed']}")
    print(f"Average training step    : {payload['tiny_overfit']['average_step_seconds']:.3f}s")
    print(f"Validation mAP@0.50      : {payload['validation_metrics']['overall']['map_50']}")
    print(f"Estimated full epoch     : {payload['timing']['estimated_full_epoch_minutes']:.2f} min")
    print(f"Checkpoint               : {pilot_checkpoint}")
    print(f"Checkpoint size          : {payload['storage']['checkpoint_size_mib']:.3f} MiB")
    print(f"Artifact                 : {pilot_artifact}")
    print("[RESULT]                 : " + ("PASS" if payload["validation_passed"] else "FAIL"))
    print("[PILOT ONLY] [FULL TRAINING NOT EXECUTED] [TEST SPLIT NOT USED]")

    if not payload["validation_passed"]:
        raise RuntimeError("Day 12 Detection training pilot failed.")
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Day 12 pretrained Detection CPU training pilot."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--allow-pretrained-download",
        action="store_true",
        help="Allow torchvision to download the COCO detection weight if absent.",
    )
    parser.add_argument("--train-samples", type=int, default=1)
    parser.add_argument("--validation-samples", type=int, default=1)
    parser.add_argument("--overfit-steps", type=int, default=3)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    run_day12_detection_training_pilot(
        project_root=args.project_root,
        allow_pretrained_download=args.allow_pretrained_download,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        overfit_steps=args.overfit_steps,
        torch_num_threads=args.torch_num_threads,
    )


if __name__ == "__main__":
    main()
