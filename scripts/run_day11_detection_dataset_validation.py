"""실제 NEU-DET Dataset·DataLoader를 검증하고 Day 11 Artifact를 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.detection.data_loader import DetectionDataLoaderConfig
from src.detection.dataset_runtime_visualization import (
    save_detection_dataset_batch_figure,
    save_duplicate_target_overlay_figure,
)
from src.detection.dataset_validation import (
    build_day11_detection_dataset_validation,
    write_validation_artifact,
)


DEFAULT_MANIFEST = Path("data/processed/neu_det/splits.json")
DEFAULT_ARTIFACT = Path(
    "reports/artifacts/day11_detection_dataset_validation.json"
)
DEFAULT_BATCH_FIGURE = Path(
    "reports/figures/day11_detection_dataset_batch.png"
)
DEFAULT_TARGET_FIGURE = Path(
    "reports/figures/day11_detection_target_overlay.png"
)


def run_day11_detection_dataset_validation(
    *,
    project_root: Path,
    batch_size: int = 2,
    duplicate_box_policy: str = "preserve",
) -> dict[str, Any]:
    project_root = project_root.resolve()
    manifest_path = project_root / DEFAULT_MANIFEST
    artifact_path = project_root / DEFAULT_ARTIFACT
    batch_figure_path = project_root / DEFAULT_BATCH_FIGURE
    target_figure_path = project_root / DEFAULT_TARGET_FIGURE

    payload, loaders = build_day11_detection_dataset_validation(
        project_root=project_root,
        manifest_path=manifest_path,
        loader_config=DetectionDataLoaderConfig(batch_size=batch_size),
        duplicate_box_policy=duplicate_box_policy,
    )

    figures = {
        "dataset_batch": save_detection_dataset_batch_figure(
            dataset=loaders.train_dataset,
            output_path=batch_figure_path,
        ),
        "duplicate_target_overlay": save_duplicate_target_overlay_figure(
            loaders=loaders,
            duplicate_records=payload["duplicate_box_records"],
            output_path=target_figure_path,
        ),
    }
    for metadata in figures.values():
        try:
            metadata["path"] = str(
                Path(metadata["path"]).resolve().relative_to(project_root)
            ).replace("\\", "/")
        except ValueError:
            metadata["path"] = str(Path(metadata["path"]).resolve())

    payload["figures"] = figures
    payload["checks"]["all_figures_decodable"] = all(
        metadata["decode_valid"] and metadata["size_bytes"] > 0
        for metadata in figures.values()
    )
    payload["validation_passed"] = all(payload["checks"].values())
    write_validation_artifact(payload, artifact_path)

    print("=" * 100)
    print("DAY 11 - DETECTION DATASET RUNTIME VALIDATION")
    print("=" * 100)
    print(f"Project root             : {project_root}")
    print(f"Manifest                 : {manifest_path}")
    print(f"Duplicate policy         : {duplicate_box_policy}")
    print(f"Total samples            : {payload['totals']['sample_count']}")
    print(f"Total boxes              : {payload['totals']['box_count']}")
    print(
        "Exact duplicates         : "
        f"{payload['totals']['raw_exact_duplicate_count']}"
    )
    print()
    for split_name, result in payload["splits"].items():
        status = "PASS" if result["validation_passed"] else "FAIL"
        print(
            f"[{status}] {split_name:<10} "
            f"images={result['sample_count']:<4} "
            f"boxes={result['dataset_box_count']:<4} "
            f"loaded={result['compared_record_count']}"
        )
    print()
    print(f"[ARTIFACT] {artifact_path}")
    print(f"[FIGURE]   {batch_figure_path}")
    print(f"[FIGURE]   {target_figure_path}")
    print(
        "[RESULT]   "
        + ("PASS" if payload["validation_passed"] else "FAIL")
    )

    if not payload["validation_passed"]:
        raise RuntimeError(
            "Day 11 Detection Dataset validation failed. "
            "Inspect the generated JSON artifact."
        )
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the Day 11 NEU-DET Detection Dataset runtime contract."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--duplicate-box-policy",
        choices=("preserve", "remove_exact"),
        default="preserve",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    run_day11_detection_dataset_validation(
        project_root=args.project_root,
        batch_size=args.batch_size,
        duplicate_box_policy=args.duplicate_box_policy,
    )


if __name__ == "__main__":
    main()
