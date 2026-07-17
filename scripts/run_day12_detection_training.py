"""Day 12 Detection Checkpoint를 재개해 Unfreeze Fine-tuning을 진행한다.

Stage 3의 Frozen-Backbone Epoch Checkpoint를 읽고 Backbone을 연 뒤 낮은
Learning Rate로 다음 Epoch를 학습한다. ``--target-total-epochs``를 늘려
동일 Script로 이후 Epoch를 안전하게 재개할 수 있다.

실행 예:
    python -m scripts.run_day12_detection_training `
        --project-root . `
        --target-total-epochs 2 `
        --unfreeze-learning-rate 0.001 `
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
    restore_detection_checkpoint,
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
from src.detection.training_resume import (
    extract_best_metric_value,
    is_better_metric,
    set_optimizer_learning_rate,
    validate_resume_epoch_range,
)
from src.detection.transforms import create_detection_transform


DEFAULT_MANIFEST = Path("data/processed/neu_det/splits.json")
DEFAULT_ARTIFACT = Path(
    "reports/artifacts/day12_detection_training_history.json"
)
DEFAULT_LATEST_CHECKPOINT = Path(
    "models/detection/day12_detection_latest.pt"
)
DEFAULT_BEST_CHECKPOINT = Path(
    "models/detection/day12_detection_best.pt"
)
MINIMUM_FREE_GIB = 2.0


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
    epoch_number = int(payload.get("epoch_index", 0)) + 1
    if event == "train_progress":
        print(
            f"[TRAIN EPOCH {epoch_number}] "
            f"batch={payload['batch_count']} "
            f"samples={payload['sample_count']} "
            f"latest_loss={payload['latest_total_loss']:.6f} "
            f"avg_loss={payload['average_total_loss']:.6f} "
            f"elapsed={payload['elapsed_seconds'] / 60.0:.2f}m",
            flush=True,
        )
    elif event == "evaluation_progress":
        print(
            f"[VALIDATION EPOCH {epoch_number}] "
            f"batch={payload['batch_count']} "
            f"samples={payload['sample_count']} "
            f"boxes={payload['prediction_box_count']} "
            f"elapsed={payload['elapsed_seconds']:.2f}s",
            flush=True,
        )


def run_day12_detection_training(
    *,
    project_root: Path,
    target_total_epochs: int,
    unfreeze_learning_rate: float = 0.001,
    torch_num_threads: int = 2,
    log_interval: int = 100,
) -> dict[str, Any]:
    """latest Checkpoint 다음 Epoch부터 목표 Epoch까지 학습·검증한다."""
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path.")
    for name, value in (
        ("target_total_epochs", target_total_epochs),
        ("torch_num_threads", torch_num_threads),
        ("log_interval", log_interval),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive int.")
    if (
        not isinstance(unfreeze_learning_rate, (int, float))
        or isinstance(unfreeze_learning_rate, bool)
        or float(unfreeze_learning_rate) <= 0.0
    ):
        raise ValueError(
            "unfreeze_learning_rate must be a positive number."
        )

    project_root = project_root.resolve()
    manifest_path = project_root / DEFAULT_MANIFEST
    artifact_path = project_root / DEFAULT_ARTIFACT
    latest_path = project_root / DEFAULT_LATEST_CHECKPOINT
    best_path = project_root / DEFAULT_BEST_CHECKPOINT
    for required_path, description in (
        (manifest_path, "Split manifest"),
        (latest_path, "Latest checkpoint"),
        (best_path, "Best checkpoint"),
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"{description} does not exist: {required_path}."
            )

    free_before = shutil.disk_usage(project_root).free / (1024**3)
    if free_before < MINIMUM_FREE_GIB:
        raise OSError(
            f"At least {MINIMUM_FREE_GIB:.1f} GiB free space is required. "
            f"Current: {free_before:.3f} GiB."
        )
    expected_weight = _expected_weight_path()
    if not expected_weight.is_file():
        raise FileNotFoundError(
            "COCO pretrained Detection weight is not cached."
        )

    continuation_config = DetectionTrainingConfig(
        batch_size=1,
        epochs=target_total_epochs,
        learning_rate=float(unfreeze_learning_rate),
        scheduler_name="none",
        scheduler_step_size=1,
        scheduler_gamma=1.0,
        freeze_backbone_epochs=1,
        horizontal_flip_probability=0.5,
        torch_num_threads=torch_num_threads,
    )
    model_config = DetectionModelConfig(
        min_size=continuation_config.min_size,
        max_size=continuation_config.max_size,
        use_pretrained_weights=True,
        use_pretrained_backbone=False,
        progress=False,
    )
    class_mapping = build_detection_checkpoint_class_mapping(
        model_config.index_to_class
    )

    torch.manual_seed(continuation_config.random_seed)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(torch_num_threads)
    execution_started = time.perf_counter()
    completed_epochs: list[int] = []
    interrupted = False

    try:
        train_dataset = NeuDetDetectionDataset.from_manifest(
            manifest_path=manifest_path,
            project_root=project_root,
            split="train",
            transform=create_detection_transform(
                training=True,
                horizontal_flip_probability=(
                    continuation_config.horizontal_flip_probability
                ),
            ),
            duplicate_box_policy=(
                continuation_config.duplicate_box_policy
            ),
        )
        validation_dataset = NeuDetDetectionDataset.from_manifest(
            manifest_path=manifest_path,
            project_root=project_root,
            split="validation",
            transform=create_detection_transform(training=False),
            duplicate_box_policy=(
                continuation_config.duplicate_box_policy
            ),
        )
        train_loader = _create_loader(
            dataset=train_dataset,
            batch_size=continuation_config.batch_size,
            shuffle=True,
            random_seed=continuation_config.random_seed,
        )
        validation_loader = _create_loader(
            dataset=validation_dataset,
            batch_size=continuation_config.batch_size,
            shuffle=False,
            random_seed=continuation_config.random_seed,
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
        unfreeze_metadata = set_detection_backbone_trainable(
            model,
            trainable=True,
        )
        optimization = build_detection_optimization(
            model=model,
            config=continuation_config,
        )
        checkpoint_state = restore_detection_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimization.optimizer,
            scheduler=None,
            expected_class_mapping=class_mapping,
            map_location="cpu",
        )
        epoch_range = validate_resume_epoch_range(
            checkpoint_epoch=checkpoint_state.epoch,
            total_epochs=target_total_epochs,
        )
        learning_rates = set_optimizer_learning_rate(
            optimization.optimizer,
            learning_rate=float(unfreeze_learning_rate),
        )
        history = list(checkpoint_state.history)
        best_metric = float(checkpoint_state.best_metric)
        starting_best_metric = best_metric

        print("=" * 100)
        print("DAY 12 - DETECTION RESUMED UNFREEZE FINE-TUNING")
        print("=" * 100)
        print(f"Project root             : {project_root}")
        print(f"Resume checkpoint epoch  : {checkpoint_state.epoch}")
        print(f"Next epoch index         : {checkpoint_state.next_epoch}")
        print(f"Target total epochs      : {target_total_epochs}")
        print(f"Train images             : {len(train_dataset)}")
        print(f"Validation images        : {len(validation_dataset)}")
        print(f"Batch size               : {continuation_config.batch_size}")
        print(f"Backbone trainable       : {unfreeze_metadata['backbone_trainable']}")
        print(f"Trainable parameters     : {unfreeze_metadata['trainable_parameters']:,}")
        print(f"Learning rate            : {learning_rates}")
        print(f"Starting best mAP@0.50   : {starting_best_metric:.6f}")
        print("[TEST SPLIT NOT USED] [RESUME FROM LATEST]")

        epoch_artifacts: list[dict[str, Any]] = []
        for epoch_index in epoch_range:
            train_result = run_detection_training_epoch(
                model=model,
                optimizer=optimization.optimizer,
                data_loader=train_loader,
                epoch_index=epoch_index,
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
                score_threshold=continuation_config.score_threshold,
                iou_threshold=continuation_config.iou_threshold,
                device="cpu",
                log_interval=max(1, min(50, log_interval)),
                progress_callback=_progress_printer,
            )
            validation_summary = validation_result.summary()
            candidate_metric = extract_best_metric_value(
                metric_name=continuation_config.best_metric_name,
                validation_summary=validation_summary,
            )
            improved = is_better_metric(
                metric_name=continuation_config.best_metric_name,
                candidate=candidate_metric,
                current_best=best_metric,
            )
            if improved:
                best_metric = candidate_metric

            history_entry = {
                "epoch": epoch_index,
                "stage": "unfrozen_backbone_fine_tuning",
                "backbone_trainable": True,
                "train": train_result.to_dict(),
                "validation": validation_summary,
                "learning_rates": list(learning_rates),
                "candidate_best_metric": candidate_metric,
                "best_metric_after_epoch": best_metric,
                "best_checkpoint_improved": improved,
            }
            history.append(history_entry)
            checkpoint_payload = build_detection_checkpoint_payload(
                epoch=epoch_index,
                model=model,
                optimizer=optimization.optimizer,
                scheduler=None,
                training_config=continuation_config.to_dict(),
                class_mapping=class_mapping,
                best_metric=best_metric,
                history=history,
            )
            saved_latest, saved_best = save_detection_checkpoint(
                payload=checkpoint_payload,
                latest_path=latest_path,
                best_path=best_path,
                is_best=improved,
            )
            completed_epochs.append(epoch_index)
            epoch_artifacts.append(
                {
                    **history_entry,
                    "latest_checkpoint": _relative_or_absolute(
                        saved_latest,
                        project_root,
                    ),
                    "best_checkpoint_written": saved_best is not None,
                }
            )

            overall = validation_summary["metrics"]["overall"]
            print("-" * 100)
            print(f"Completed epoch          : {epoch_index + 1}/{target_total_epochs}")
            print(f"Average total loss       : {train_result.average_losses['total_loss']:.6f}")
            print(f"Training time            : {train_result.elapsed_seconds / 60.0:.2f} min")
            print(f"Validation Precision     : {overall['precision']:.6f}")
            print(f"Validation Recall        : {overall['recall']:.6f}")
            print(f"Validation F1            : {overall['f1']:.6f}")
            print(f"Validation mAP@0.50      : {overall['map_50']}")
            print(f"Best improved            : {improved}")

        final_epoch = completed_epochs[-1]
        payload: dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "day12_detection_resumed_unfreeze_fine_tuning",
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
                "device": "cpu",
                "torch_num_threads": torch_num_threads,
            },
            "execution_policy": {
                "resumed_from_latest": True,
                "resume_checkpoint_epoch": checkpoint_state.epoch,
                "completed_epoch_indexes": completed_epochs,
                "target_total_epochs": target_total_epochs,
                "configured_training_completed": (
                    final_epoch == target_total_epochs - 1
                ),
                "backbone_trainable": True,
                "test_split_used": False,
                "proposal_limits_applied": False,
                "duplicate_box_policy": (
                    continuation_config.duplicate_box_policy
                ),
            },
            "storage": {
                "disk_free_gib_before": round(free_before, 3),
                "disk_free_gib_after": round(
                    shutil.disk_usage(project_root).free / (1024**3),
                    3,
                ),
                "pretrained_weight_path": str(expected_weight),
                "latest_checkpoint": _relative_or_absolute(
                    latest_path,
                    project_root,
                ),
                "latest_checkpoint_size_mib": round(
                    latest_path.stat().st_size / (1024**2),
                    3,
                ),
                "best_checkpoint": _relative_or_absolute(
                    best_path,
                    project_root,
                ),
                "best_checkpoint_size_mib": round(
                    best_path.stat().st_size / (1024**2),
                    3,
                ),
            },
            "data": {
                "train_images": len(train_dataset),
                "validation_images": len(validation_dataset),
                "test_images_used": 0,
            },
            "continuation_config": continuation_config.to_dict(),
            "checkpoint_training_config_before_resume": (
                checkpoint_state.training_config
            ),
            "model": model_result.metadata,
            "unfreeze": unfreeze_metadata,
            "optimization": {
                **optimization.metadata,
                "learning_rates_after_restore_override": list(
                    learning_rates
                ),
                "scheduler_restored": False,
                "reason": (
                    "Backbone Unfreeze uses a lower constant Learning Rate; "
                    "the Stage 3 Scheduler state is intentionally not resumed."
                ),
            },
            "starting_best_metric": starting_best_metric,
            "final_best_metric": best_metric,
            "epochs": epoch_artifacts,
            "history_length": len(history),
            "timing": {
                "model_prepare_seconds": model_prepare_seconds,
                "total_execution_seconds": (
                    time.perf_counter() - execution_started
                ),
            },
            "artifacts": {
                "training_history": _relative_or_absolute(
                    artifact_path,
                    project_root,
                ),
                "latest_checkpoint": _relative_or_absolute(
                    latest_path,
                    project_root,
                ),
                "best_checkpoint": _relative_or_absolute(
                    best_path,
                    project_root,
                ),
            },
            "validation_passed": bool(
                completed_epochs
                and latest_path.is_file()
                and best_path.is_file()
            ),
        }
        _write_json(artifact_path, payload)
    except KeyboardInterrupt:
        interrupted = True
        _write_json(
            artifact_path,
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "stage": "day12_detection_resumed_unfreeze_fine_tuning",
                "interrupted": True,
                "completed_epoch_indexes": completed_epochs,
                "existing_latest_checkpoint_preserved": latest_path.is_file(),
                "existing_best_checkpoint_preserved": best_path.is_file(),
                "test_split_used": False,
                "message": (
                    "The interrupted partial epoch was not saved. "
                    "The last fully completed checkpoint remains available."
                ),
            },
        )
        raise
    finally:
        torch.set_num_threads(previous_threads)

    print("=" * 100)
    print(f"Final completed epoch    : {payload['execution_policy']['completed_epoch_indexes'][-1] + 1}")
    print(f"Final best mAP@0.50      : {payload['final_best_metric']:.6f}")
    print(f"Latest checkpoint        : {latest_path}")
    print(f"Best checkpoint          : {best_path}")
    print(f"Artifact                 : {artifact_path}")
    print("[RESULT]                 : " + ("PASS" if payload["validation_passed"] else "FAIL"))
    print("[TEST SPLIT NOT USED] [BEST MODEL SELECTED ON VALIDATION]")

    if interrupted or not payload["validation_passed"]:
        raise RuntimeError("Day 12 resumed Detection training failed.")
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume Day 12 Detection training with an unfrozen backbone."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--target-total-epochs", type=int, required=True)
    parser.add_argument(
        "--unfreeze-learning-rate",
        type=float,
        default=0.001,
    )
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    run_day12_detection_training(
        project_root=args.project_root,
        target_total_epochs=args.target_total_epochs,
        unfreeze_learning_rate=args.unfreeze_learning_rate,
        torch_num_threads=args.torch_num_threads,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
