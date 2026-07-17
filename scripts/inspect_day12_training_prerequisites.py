"""Day 12 Detection 학습 전 실제 저장소·디스크·Cache를 읽기 전용 점검한다.

이 Script는 다음 작업을 하지 않는다.
- pretrained weight 다운로드
- Model backward/optimizer step
- Checkpoint 삭제
- pytest 임시 폴더 삭제
- 기존 프로젝트 파일 수정

실행:
    python -m scripts.inspect_day12_training_prerequisites
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Iterable
from urllib.parse import urlparse

import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)

from src.detection.model_config import DetectionModelConfig
from src.detection.model_factory import create_detection_model


DEFAULT_OUTPUT_PATH = Path(
    "reports/artifacts/day12_detection_training_prerequisites.json"
)
DAY11_README_START = "<!-- DAY11_DETECTION_DATASET_MODEL_START -->"
DAY11_README_END = "<!-- DAY11_DETECTION_DATASET_MODEL_END -->"
MINIMUM_FREE_GIB = 3.0
RECOMMENDED_FREE_GIB = 5.0

PRIMARY_PATHS = (
    Path("src/detection/model_config.py"),
    Path("src/detection/model_factory.py"),
    Path("src/detection/detection_dataset.py"),
    Path("src/detection/data_loader.py"),
    Path("data/processed/neu_det/splits.json"),
    Path("reports/artifacts/day11_detection_dataset_validation.json"),
    Path("reports/artifacts/day11_detection_model_smoke_test.json"),
    Path("reports/day11_detection_dataset_and_model_implementation_summary.md"),
    Path("requirements.txt"),
    Path("README.md"),
)

SEARCH_TERMS = (
    "checkpoint",
    "optimizer",
    "scheduler",
    "box_iou",
    "average_precision",
    "detection_metrics",
    "train_detection",
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _path_metadata(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_directory": path.is_dir(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _python_symbols(path: Path) -> list[str]:
    if not path.is_file() or path.suffix.lower() != ".py":
        return []
    try:
        tree = ast.parse(_read_text(path))
    except (OSError, SyntaxError):
        return []
    return sorted(
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _torch_cache_root() -> Path:
    torch_home = os.getenv("TORCH_HOME")
    if torch_home:
        return Path(torch_home).expanduser().resolve()
    return (Path.home() / ".cache" / "torch").resolve()


def _cache_files(checkpoint_directory: Path) -> list[dict[str, Any]]:
    if not checkpoint_directory.is_dir():
        return []
    return [
        {
            "name": path.name,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "size_mib": round(path.stat().st_size / (1024**2), 3),
        }
        for path in sorted(checkpoint_directory.iterdir())
        if path.is_file()
    ]


def _find_existing_references(project_root: Path) -> dict[str, list[str]]:
    matches = {term: [] for term in SEARCH_TERMS}
    candidates: list[Path] = []
    for relative_root in (Path("src"), Path("scripts"), Path("tests")):
        root = project_root / relative_root
        if root.is_dir():
            candidates.extend(root.rglob("*.py"))

    for path in sorted(candidates):
        text = _read_text(path).lower()
        for term in SEARCH_TERMS:
            if term.lower() in text:
                matches[term].append(_relative(path, project_root))
    return matches


def _list_detection_files(project_root: Path) -> list[dict[str, Any]]:
    detection_root = project_root / "src" / "detection"
    if not detection_root.is_dir():
        return []
    return [
        {
            **_path_metadata(path, project_root),
            "symbols": _python_symbols(path),
        }
        for path in sorted(detection_root.glob("*.py"))
    ]


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def _build_weight_free_model_metadata() -> dict[str, Any]:
    config = DetectionModelConfig(
        min_size=320,
        max_size=320,
        use_pretrained_weights=False,
        use_pretrained_backbone=False,
        progress=False,
    )
    result = create_detection_model(
        config=config,
        device="cpu",
        training=False,
        proposal_limits=None,
    )
    total_parameters = sum(
        parameter.numel() for parameter in result.model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in result.model.parameters()
        if parameter.requires_grad
    )
    return {
        **result.metadata,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "full_torchvision_proposal_defaults_preserved": True,
    }


def inspect_day12_training_prerequisites(
    *,
    project_root: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    build_weight_free_model: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    disk = shutil.disk_usage(root)
    free_gib = disk.free / (1024**3)

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    weight_url = weights.url
    weight_filename = Path(urlparse(weight_url).path).name
    cache_root = _torch_cache_root()
    checkpoint_cache = cache_root / "hub" / "checkpoints"
    expected_weight_path = checkpoint_cache / weight_filename

    primary_paths = {
        item.as_posix(): _path_metadata(root / item, root)
        for item in PRIMARY_PATHS
    }
    readme_path = root / "README.md"
    readme_text = _read_text(readme_path) if readme_path.is_file() else ""
    day11_markers_ok = (
        readme_text.count(DAY11_README_START) == 1
        and readme_text.count(DAY11_README_END) == 1
    )

    model_metadata: dict[str, Any]
    if build_weight_free_model:
        try:
            model_metadata = {
                "status": "PASS",
                **_build_weight_free_model_metadata(),
            }
        except Exception as error:  # 실제 환경 진단 결과를 Artifact에 남긴다.
            model_metadata = {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "message": str(error),
            }
    else:
        model_metadata = {
            "status": "SKIPPED",
            "reason": "--skip-model-build",
        }

    existing_model_paths = []
    for relative in (Path("models"), Path("checkpoints")):
        path = root / relative
        existing_model_paths.append(_path_metadata(path, root))
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in {
                    ".pt",
                    ".pth",
                    ".ckpt",
                }:
                    existing_model_paths.append(_path_metadata(child, root))

    requirements_text = (
        _read_text(root / "requirements.txt")
        if (root / "requirements.txt").is_file()
        else ""
    )

    checks = {
        "project_root_exists": root.is_dir(),
        "all_primary_paths_exist": all(
            metadata["exists"] for metadata in primary_paths.values()
        ),
        "day11_readme_markers_valid": day11_markers_ok,
        "minimum_free_space_ready": free_gib >= MINIMUM_FREE_GIB,
        "recommended_free_space_ready": free_gib >= RECOMMENDED_FREE_GIB,
        "pretrained_weight_cached": expected_weight_path.is_file(),
        "weight_free_model_factory_compatible": (
            model_metadata["status"] == "PASS"
        ),
        "torch_cpu_build": "+cpu" in torch.__version__
        or not torch.cuda.is_available(),
        "no_new_dependency_required_for_stage1": True,
    }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "day": 12,
        "title": "Detection Training Prerequisites",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "pytest": _package_version("pytest"),
        },
        "execution_policy": {
            "pretrained_download_requested": False,
            "pretrained_download_executed": False,
            "training_executed": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "files_deleted": False,
        },
        "disk": {
            "free_bytes": disk.free,
            "free_gib": round(free_gib, 3),
            "minimum_required_gib": MINIMUM_FREE_GIB,
            "recommended_gib": RECOMMENDED_FREE_GIB,
        },
        "pretrained_weight": {
            "enum": str(weights),
            "url": weight_url,
            "expected_filename": weight_filename,
            "expected_cache_path": str(expected_weight_path),
            "cached": expected_weight_path.is_file(),
            "cached_size_bytes": (
                expected_weight_path.stat().st_size
                if expected_weight_path.is_file()
                else None
            ),
            "download_size_bytes": (
                expected_weight_path.stat().st_size
                if expected_weight_path.is_file()
                else None
            ),
            "download_size_note": (
                "Exact byte size is recorded only when the file already exists. "
                "This inspection performs no network request."
            ),
        },
        "torch_cache": {
            "root": str(cache_root),
            "checkpoint_directory": str(checkpoint_cache),
            "files": _cache_files(checkpoint_cache),
        },
        "primary_paths": primary_paths,
        "src_detection_files": _list_detection_files(root),
        "model_and_checkpoint_paths": existing_model_paths,
        "existing_training_metric_references": _find_existing_references(root),
        "requirements": {
            "path": "requirements.txt",
            "contains_torch": "torch" in requirements_text.lower(),
            "contains_torchvision": "torchvision" in requirements_text.lower(),
            "contains_pycocotools": "pycocotools" in requirements_text.lower(),
            "contains_torchmetrics": "torchmetrics" in requirements_text.lower(),
        },
        "weight_free_model_factory": model_metadata,
        "checks": checks,
        "ready_for_pretrained_download": (
            checks["minimum_free_space_ready"]
            and checks["all_primary_paths_exist"]
            and checks["day11_readme_markers_valid"]
            and checks["weight_free_model_factory_compatible"]
        ),
    }

    resolved_output = (
        output_path
        if output_path.is_absolute()
        else root / output_path
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_name(f".{resolved_output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(resolved_output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print("=" * 100)
    print("DAY 12 - DETECTION TRAINING PREREQUISITES")
    print("=" * 100)
    print(f"Project root             : {root}")
    print(f"Disk free                : {free_gib:.3f} GiB")
    print(f"Minimum / recommended    : {MINIMUM_FREE_GIB:.1f} / {RECOMMENDED_FREE_GIB:.1f} GiB")
    print(f"Detection weight cached  : {expected_weight_path.is_file()}")
    print(f"Expected weight file     : {weight_filename}")
    print(f"Cache directory          : {checkpoint_cache}")
    print(f"Day 11 README markers    : {'PASS' if day11_markers_ok else 'FAIL'}")
    print(f"Weight-free model build  : {model_metadata['status']}")
    if model_metadata["status"] == "PASS":
        print(
            "Model parameters          : "
            f"{model_metadata['total_parameters']:,}"
        )
        print(
            "Trainable parameters      : "
            f"{model_metadata['trainable_parameters']:,}"
        )
    print(f"Artifact                  : {resolved_output}")
    print(
        "Ready for download        : "
        f"{payload['ready_for_pretrained_download']}"
    )
    print()
    print("[NO DOWNLOAD] [NO TRAINING] [NO FILE DELETION]")
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Day 12 training prerequisites without downloading weights."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--skip-model-build",
        action="store_true",
        help="Skip weight-free model construction and parameter counting.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    inspect_day12_training_prerequisites(
        project_root=args.project_root,
        output_path=args.output_path,
        build_weight_free_model=not args.skip_model_build,
    )


if __name__ == "__main__":
    main()
