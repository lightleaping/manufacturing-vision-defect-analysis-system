"""Day 12 전체 Train·Validation Split으로 Frozen-Backbone 1 Epoch를 실행한다.

이 단계는 Pilot의 단일 Batch 추정치를 실제 전체 Split 시간으로 검증한다.
Test Split은 사용하지 않으며, 전체 Day 12 Fine-tuning 완료를 의미하지 않는다.

실행 예:
    python -m scripts.run_day12_detection_one_epoch_pilot `
        --project-root . `
        --torch-num-threads 2
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any
from urllib.parse import urlparse

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)

from src.detection.checkpoint import (
    build_detection_checkpoint_payload,
    save_detection_checkpoint,
)
from src.detection.data_loader import detection_collate_fn
from src.detection.detection_dataset import NeuDetDetectionDataset
from src.detection.epoch_runner import (
    build_detection_checkpoint_class_mapping,
    run_detection_evaluation_epoch,
    run_detection_training_epoch,
)
from src.detection.model_config import DetectionModelConfig
from src.detection.model_factory import create_detection_model
from src.detection.optimization import (
    build_detection_optimization,
    set_detection_backbone_trainable,
)
from src.detection.training_config import DetectionTrainingConfig
from src.detection.transforms import create_detection_transform


DEFAULT_MANIFEST = Path("data/processed/neu_det/splits.json")
DEFAULT_ARTIFACT = Path(
    "reports/artifacts/day12_detection_one_epoch_pilot.json"
)
DEFAULT_LATEST_CHECKPOINT = Path(
    "models/detection/day12_detection_latest.pt"
)
DEFAULT_BEST_CHECKPOINT = Path(
    "models/detection/day12_detection_best.pt"
)
MINIMUM_FREE_GIB = 3.0


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
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _create_loader(
    *,
    dataset: NeuDetDetectionDataset,
    batch_size: int,
    shuffle: bool,
    random_seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
        collate_fn=detection_collate_fn,
    )


def _progress_printer(payload: dict[str, Any]) -> None:
    event = payload.get("event")
    if event == "train_progress":
        print(
            "[TRAIN] "
            f"batch={payload['batch_count']} "
            f"samples={payload['sample_count']} "
            f"latest_loss={payload['latest_total_loss']:.6f} "
            f"avg_loss={payload['average_total_loss']:.6f} "
            f"elapsed={payload['elapsed_seconds'] / 60.0:.2f}m",
            flush=True,
        )
    elif event == "evaluation_progress":
        print(
            "[VALIDATION] "
            f"batch={payload['batch_count']} "
            f"samples={payload['sample_count']} "
            f"boxes={payload['prediction_box_count']} "
            f"elapsed={payload['elapsed_seconds']:.2f}s",
            flush=True,
        )


def run_day12_detection_one_epoch_pilot(
    *,
    project_root: Path,
    torch_num_threads: int = 2,
    log_interval: int = 100,
) -> dict[str, Any]:
    """전체 Train 1 Epoch와 Validation 평가 후 latest·best를 저장한다."""
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path.")
    for name, value in (
        ("torch_num_threads", torch_num_threads),
        ("log_interval", log_interval),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive int.")

    project_root = project_root.resolve()
    manifest_path = project_root / DEFAULT_MANIFEST
    artifact_path = project_root / DEFAULT_ARTIFACT
    latest_path = project_root / DEFAULT_LATEST_CHECKPOINT
    best_path = project_root / DEFAULT_BEST_CHECKPOINT
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}.")

    free_before = shutil.disk_usage(project_root).free / (1024**3)
    if free_before < MINIMUM_FREE_GIB:
        raise OSError(
            f"At least {MINIMUM_FREE_GIB:.1f} GiB free space is required. "
            f"Current: {free_before:.3f} GiB."
        )
    expected_weight = _expected_weight_path()
    if not expected_weight.is_file():
        raise FileNotFoundError(
            "COCO pretrained Detection weight is not cached. "
            "Run the Stage 2 pretrained pilot first."
        )

    training_config = DetectionTrainingConfig(
        batch_size=1,
        epochs=3,
        learning_rate=0.005,
        scheduler_name="step_lr",
        scheduler_step_size=2,
        scheduler_gamma=0.1,
        freeze_backbone_epochs=1,
        horizontal_flip_probability=0.5,
        torch_num_threads=torch_num_threads,
    )
    model_config = DetectionModelConfig(
        min_size=training_config.min_size,
        max_size=training_config.max_size,
        use_pretrained_weights=True,
        use_pretrained_backbone=False,
        progress=False,
    )
    class_mapping = build_detection_checkpoint_class_mapping(
        model_config.index_to_class
    )

    torch.manual_seed(training_config.random_seed)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(torch_num_threads)
    execution_started = time.perf_counter()
    interrupted = False

    try:
        train_dataset = NeuDetDetectionDataset.from_manifest(
            manifest_path=manifest_path,
            project_root=project_root,
            split="train",
            transform=create_detection_transform(
                training=True,
                horizontal_flip_probability=(
                    training_config.horizontal_flip_probability
                ),
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
        train_loader = _create_loader(
            dataset=train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            random_seed=training_config.random_seed,
        )
        validation_loader = _create_loader(
            dataset=validation_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            random_seed=training_config.random_seed,
        )

        model_started = time.perf_counter()
        model_result = create_detection_model(
            config=model_config,
            device="cpu",
            training=True,
            proposal_limits=None,
        )
        model = model_result.model
        model_prepare_seconds = time.perf_counter() - model_started
        freeze_metadata = set_detection_backbone_trainable(
            model,
            trainable=False,
        )
        optimization = build_detection_optimization(
            model=model,
            config=training_config,
        )

        print("=" * 100)
        print("DAY 12 - DETECTION FULL-SPLIT ONE-EPOCH PILOT")
        print("=" * 100)
        print(f"Project root             : {project_root}")
        print(f"Train images             : {len(train_dataset)}")
        print(f"Validation images        : {len(validation_dataset)}")
        print(f"Batch size               : {training_config.batch_size}")
        print(f"Backbone trainable       : {freeze_metadata['backbone_trainable']}")
        print(f"Trainable parameters     : {freeze_metadata['trainable_parameters']:,}")
        print(f"Horizontal flip          : {training_config.horizontal_flip_probability}")
        print(f"Learning rate            : {training_config.learning_rate}")
        print(f"Best metric              : {training_config.best_metric_name}")
        print("[TEST SPLIT NOT USED] [ONE EPOCH PILOT ONLY]")

        train_result = run_detection_training_epoch(
            model=model,
            optimizer=optimization.optimizer,
            data_loader=train_loader,
            epoch_index=0,
            device="cpu",
            log_interval=log_interval,
            progress_callback=_progress_printer,
        )
        validation_result = run_detection_evaluation_epoch(
            model=model,
            data_loader=validation_loader,
            split="validation",
            num_classes=model_config.num_classes,
            index_to_class=model_config.index_to_class,
            score_threshold=training_config.score_threshold,
            iou_threshold=training_config.iou_threshold,
            device="cpu",
            log_interval=max(1, min(50, log_interval)),
            progress_callback=_progress_printer,
        )

        map_50 = validation_result.metrics["overall"]["map_50"]
        best_metric = 0.0 if map_50 is None else float(map_50)
        lr_before_scheduler = [
            float(group["lr"]) for group in optimization.optimizer.param_groups
        ]
        if optimization.scheduler is not None:
            optimization.scheduler.step()
        lr_after_scheduler = [
            float(group["lr"]) for group in optimization.optimizer.param_groups
        ]

        history_entry = {
            "epoch": 0,
            "stage": "frozen_backbone_full_split_pilot",
            "backbone_trainable": False,
            "train": train_result.to_dict(),
            "validation": validation_result.summary(),
            "learning_rates_before_scheduler": lr_before_scheduler,
            "learning_rates_after_scheduler": lr_after_scheduler,
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
        saved_latest, saved_best = save_detection_checkpoint(
            payload=checkpoint_payload,
            latest_path=latest_path,
            best_path=best_path,
            is_best=True,
        )

        payload: dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "day12_full_split_one_epoch_pilot",
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
                "device": "cpu",
                "torch_num_threads": torch_num_threads,
            },
            "execution_policy": {
                "full_training_completed": False,
                "executed_epochs": 1,
                "configured_total_epochs": training_config.epochs,
                "backbone_trainable": False,
                "test_split_used": False,
                "proposal_limits_applied": False,
                "duplicate_box_policy": (
                    training_config.duplicate_box_policy
                ),
            },
            "storage": {
                "disk_free_gib_before": round(free_before, 3),
                "disk_free_gib_after": round(
                    shutil.disk_usage(project_root).free / (1024**3),
                    3,
                ),
                "pretrained_weight_path": str(expected_weight),
                "pretrained_weight_size_mib": round(
                    expected_weight.stat().st_size / (1024**2),
                    3,
                ),
                "latest_checkpoint": _relative_or_absolute(
                    saved_latest,
                    project_root,
                ),
                "latest_checkpoint_size_mib": round(
                    saved_latest.stat().st_size / (1024**2),
                    3,
                ),
                "best_checkpoint": (
                    None
                    if saved_best is None
                    else _relative_or_absolute(saved_best, project_root)
                ),
                "best_checkpoint_size_mib": (
                    None
                    if saved_best is None
                    else round(saved_best.stat().st_size / (1024**2), 3)
                ),
            },
            "data": {
                "train_images": len(train_dataset),
                "validation_images": len(validation_dataset),
                "test_images_used": 0,
            },
            "training_config": training_config.to_dict(),
            "model": model_result.metadata,
            "freeze": freeze_metadata,
            "optimization": optimization.metadata,
            "train_epoch": train_result.to_dict(),
            "validation": validation_result.summary(),
            "best_metric": {
                "name": training_config.best_metric_name,
                "value": best_metric,
                "first_completed_epoch_is_best": True,
            },
            "timing": {
                "model_prepare_seconds": model_prepare_seconds,
                "total_execution_seconds": (
                    time.perf_counter() - execution_started
                ),
            },
            "artifacts": {
                "one_epoch_pilot": _relative_or_absolute(
                    artifact_path,
                    project_root,
                ),
                "latest_checkpoint": _relative_or_absolute(
                    saved_latest,
                    project_root,
                ),
                "best_checkpoint": (
                    None
                    if saved_best is None
                    else _relative_or_absolute(saved_best, project_root)
                ),
            },
            "validation_passed": bool(
                train_result.all_losses_finite
                and train_result.all_inputs_unchanged
                and validation_result.all_inputs_unchanged
                and saved_latest.is_file()
                and saved_best is not None
                and saved_best.is_file()
            ),
        }
        _write_json(artifact_path, payload)
    except KeyboardInterrupt:
        interrupted = True
        interruption_payload = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "day12_full_split_one_epoch_pilot",
            "interrupted": True,
            "completed_checkpoint_written": False,
            "test_split_used": False,
            "message": (
                "Execution was interrupted before a complete epoch checkpoint. "
                "Existing checkpoints were not deleted."
            ),
        }
        _write_json(artifact_path, interruption_payload)
        raise
    finally:
        torch.set_num_threads(previous_threads)

    print("-" * 100)
    print(f"Average total loss       : {payload['train_epoch']['average_losses']['total_loss']:.6f}")
    print(f"Training time            : {payload['train_epoch']['elapsed_seconds'] / 60.0:.2f} min")
    print(f"Validation time          : {payload['validation']['elapsed_seconds']:.2f}s")
    print(f"Validation Precision     : {payload['validation']['metrics']['overall']['precision']:.6f}")
    print(f"Validation Recall        : {payload['validation']['metrics']['overall']['recall']:.6f}")
    print(f"Validation F1            : {payload['validation']['metrics']['overall']['f1']:.6f}")
    print(f"Validation mAP@0.50      : {payload['validation']['metrics']['overall']['map_50']}")
    print(f"Latest checkpoint        : {latest_path}")
    print(f"Best checkpoint          : {best_path}")
    print(f"Artifact                 : {artifact_path}")
    print("[RESULT]                 : " + ("PASS" if payload["validation_passed"] else "FAIL"))
    print("[ONE EPOCH PILOT ONLY] [FULL TRAINING NOT COMPLETED] [TEST SPLIT NOT USED]")

    if interrupted or not payload["validation_passed"]:
        raise RuntimeError("Day 12 one-epoch full-split pilot failed.")
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one full frozen-backbone Detection epoch and Validation."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    run_day12_detection_one_epoch_pilot(
        project_root=args.project_root,
        torch_num_threads=args.torch_num_threads,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
