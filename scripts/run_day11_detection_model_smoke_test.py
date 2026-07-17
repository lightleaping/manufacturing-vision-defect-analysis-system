"""Weight 다운로드 없이 실제 NEU-DET 한 장으로 Detection Model을 검증한다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any

import torch
import torchvision

from src.detection.detection_dataset import NeuDetDetectionDataset
from src.detection.model_config import DetectionModelConfig
from src.detection.model_factory import (
    DetectionProposalLimits,
    create_detection_model,
)
from src.detection.model_smoke_visualization import (
    capture_model_smoke_prediction,
    save_model_smoke_prediction_figure,
)
from src.detection.model_validation import run_detection_model_smoke_validation
from src.detection.transforms import create_detection_transform


DEFAULT_MANIFEST = Path("data/processed/neu_det/splits.json")
DEFAULT_ARTIFACT = Path(
    "reports/artifacts/day11_detection_model_smoke_test.json"
)
DEFAULT_PREDICTION_FIGURE = Path(
    "reports/figures/day11_detection_model_predictions_smoke_test.png"
)


def _relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def write_model_smoke_artifact(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_day11_detection_model_smoke_test(
    *,
    project_root: Path,
    split: str = "train",
    sample_index: int = 0,
    smoke_size: int = 64,
    random_seed: int = 42,
    torch_num_threads: int = 2,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    manifest_path = project_root / DEFAULT_MANIFEST
    artifact_path = project_root / DEFAULT_ARTIFACT
    prediction_figure_path = project_root / DEFAULT_PREDICTION_FIGURE

    if not isinstance(sample_index, int) or isinstance(sample_index, bool):
        raise TypeError("sample_index must be int.")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")
    if not isinstance(smoke_size, int) or isinstance(smoke_size, bool):
        raise TypeError("smoke_size must be int.")
    if smoke_size < 32:
        raise ValueError("smoke_size must be at least 32.")
    if not isinstance(torch_num_threads, int) or torch_num_threads <= 0:
        raise ValueError("torch_num_threads must be a positive int.")

    torch.manual_seed(random_seed)
    previous_num_threads = torch.get_num_threads()
    torch.set_num_threads(torch_num_threads)

    try:
        dataset = NeuDetDetectionDataset.from_manifest(
            manifest_path=manifest_path,
            project_root=project_root,
            split=split,
            transform=create_detection_transform(training=False),
            duplicate_box_policy="preserve",
        )
        if sample_index >= len(dataset):
            raise IndexError(
                f"sample_index {sample_index} is outside dataset length {len(dataset)}."
            )
        image, target = dataset[sample_index]
        sample = dataset.samples[sample_index]

        config = DetectionModelConfig(
            min_size=smoke_size,
            max_size=smoke_size,
            use_pretrained_weights=False,
            use_pretrained_backbone=False,
            progress=False,
        )
        proposal_limits = DetectionProposalLimits()
        build_result = create_detection_model(
            config=config,
            device="cpu",
            training=False,
            proposal_limits=proposal_limits,
        )
        validation = run_detection_model_smoke_validation(
            model=build_result.model,
            images=[image],
            targets=[target],
            num_classes=config.num_classes,
            device="cpu",
        )
        prediction = capture_model_smoke_prediction(
            model=build_result.model,
            image=image,
            device="cpu",
        )
        prediction_figure = save_model_smoke_prediction_figure(
            image=image,
            target=target,
            prediction=prediction,
            index_to_class=dataset.index_to_class,
            output_path=prediction_figure_path,
            max_predictions=proposal_limits.box_detections_per_img,
        )
    finally:
        torch.set_num_threads(previous_num_threads)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "day": 11,
        "title": "Detection Model Forward and Loss Smoke Test",
        "project_name": "Manufacturing Vision Defect Analysis System",
        "project_name_ko": "제조 비전 결함 분석 시스템",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cpu",
            "torch_num_threads_for_smoke_test": torch_num_threads,
        },
        "execution_policy": {
            "full_training_executed": False,
            "optimizer_step_executed": False,
            "backward_executed": False,
            "pretrained_weight_download_requested": False,
            "pretrained_weight_download_executed": False,
            "purpose": (
                "Training/Evaluation forward contract validation before Day 12."
            ),
            "smoke_only_proposal_limits": proposal_limits.to_torchvision_kwargs(),
            "smoke_input_resize": [smoke_size, smoke_size],
            "day12_note": (
                "Day 12 pretrained fine-tuning uses the same model factory without "
                "smoke-only proposal limits."
            ),
        },
        "source_sample": {
            "split": split,
            "sample_index": sample_index,
            "record_id": sample.record_id,
            "image_path": _relative_or_absolute(sample.image_path, project_root),
            "annotation_path": _relative_or_absolute(
                sample.annotation_path,
                project_root,
            ),
            "image_shape": list(image.shape),
            "box_count": int(target["boxes"].shape[0]),
            "labels": [int(value) for value in target["labels"].tolist()],
        },
        "model": build_result.metadata,
        "smoke_test": validation.payload,
        "artifacts": {
            "analysis_json": _relative_or_absolute(artifact_path, project_root),
            "prediction_figure": prediction_figure,
        },
    }
    payload["validation_passed"] = bool(
        validation.payload["validation_passed"]
        and not build_result.metadata["network_download_requested"]
        and build_result.metadata["predictor_output_classes"] == config.num_classes
    )
    write_model_smoke_artifact(payload, artifact_path)

    print("=" * 100)
    print("DAY 11 - DETECTION MODEL FORWARD AND LOSS SMOKE TEST")
    print("=" * 100)
    print(f"Project root             : {project_root}")
    print(f"Source sample            : {sample.record_id}")
    print(f"Image shape              : {list(image.shape)}")
    print(f"Ground-truth boxes       : {int(target['boxes'].shape[0])}")
    print(f"Architecture             : {build_result.metadata['architecture']}")
    print(f"Device                   : {build_result.metadata['device']}")
    print(f"Classes with background  : {config.num_classes}")
    print(f"Smoke resize             : {smoke_size} x {smoke_size}")
    print("Pretrained download      : False")
    print()
    training_payload = validation.payload["training_forward"]
    for loss_name, loss_value in training_payload["losses"].items():
        print(f"[LOSS] {loss_name:<20} {loss_value:.6f}")
    print(f"[LOSS] {'total':<20} {training_payload['total_loss']:.6f}")
    print(
        "[TIME] training forward     "
        f"{training_payload['elapsed_seconds']:.3f}s"
    )
    print(
        "[TIME] evaluation forward   "
        f"{validation.payload['evaluation_forward']['elapsed_seconds']:.3f}s"
    )
    prediction = validation.payload["evaluation_forward"]["predictions"][0]
    print(f"[PRED] output boxes         {prediction['box_count']}")
    print(f"[ARTIFACT] {artifact_path}")
    print(f"[FIGURE]   {prediction_figure_path}")
    print(
        "[RESULT]   "
        + ("PASS" if payload["validation_passed"] else "FAIL")
    )

    if not payload["validation_passed"]:
        raise RuntimeError(
            "Day 11 Detection Model Smoke Test failed. Inspect the artifact."
        )
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a weight-free Faster R-CNN training/evaluation forward smoke test."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--smoke-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    run_day11_detection_model_smoke_test(
        project_root=args.project_root,
        split=args.split,
        sample_index=args.sample_index,
        smoke_size=args.smoke_size,
        random_seed=args.random_seed,
        torch_num_threads=args.torch_num_threads,
    )


if __name__ == "__main__":
    main()
