"""Day 8 Streamlit Dashboard 육안 검증과 Screenshot Artifact를 확정한다.

실행:
    python -m scripts.finalize_day8_visual_validation

이 Script는 NORMAL·DEFECT Dashboard Screenshot이 실제 PNG인지 검사한 뒤,
기존 Day 8 통합 검증 JSON에 육안 검증 완료 상태와 Screenshot 경로를 기록한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "Manufacturing Vision Defect Analysis System"
RUN_NAME = "day8_streamlit_dashboard_validation"

VALIDATION_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "artifacts"
    / "day8_streamlit_dashboard_validation.json"
)
NORMAL_SCREENSHOT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "day8_streamlit_dashboard_normal.png"
)
DEFECT_SCREENSHOT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "day8_streamlit_dashboard_defect.png"
)
DEFAULT_SCREENSHOT_PATHS = (
    NORMAL_SCREENSHOT_PATH,
    DEFECT_SCREENSHOT_PATH,
)


def read_json_object(path: Path) -> dict[str, Any]:
    """JSON 파일을 UTF-8로 읽고 최상위 Object 형식을 검증한다."""

    if not path.is_file():
        raise FileNotFoundError(f"Validation artifact does not exist: {path}")

    with path.open(mode="r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise TypeError("Validation artifact top-level value must be an object.")
    return payload


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a Mapping.")
    return value


def _validate_existing_integration_payload(payload: Mapping[str, Any]) -> None:
    """육안 검증 전 실제 FastAPI 통합 검증 결과가 정상인지 확인한다."""

    if payload.get("project") != PROJECT_NAME:
        raise ValueError(f"project must be {PROJECT_NAME}.")
    if payload.get("run_name") != RUN_NAME:
        raise ValueError(f"run_name must be {RUN_NAME}.")

    health = _require_mapping(payload.get("health"), name="health")
    if health.get("status") != "ok":
        raise ValueError("health.status must be ok.")
    if health.get("model_loaded") is not True:
        raise ValueError("health.model_loaded must be true.")

    normal_response = _require_mapping(
        _require_mapping(payload.get("normal_image"), name="normal_image").get(
            "response"
        ),
        name="normal_image.response",
    )
    defect_response = _require_mapping(
        _require_mapping(payload.get("defect_image"), name="defect_image").get(
            "response"
        ),
        name="defect_image.response",
    )

    if normal_response.get("prediction") != 0:
        raise ValueError("NORMAL integration prediction must be 0.")
    if normal_response.get("prediction_class_name") != "NORMAL":
        raise ValueError("NORMAL integration class must be NORMAL.")
    if defect_response.get("prediction") != 1:
        raise ValueError("DEFECT integration prediction must be 1.")
    if defect_response.get("prediction_class_name") != "DEFECT":
        raise ValueError("DEFECT integration class must be DEFECT.")


def validate_png_screenshot(path: Path) -> dict[str, int | str]:
    """Screenshot가 실제로 열 수 있는 비어 있지 않은 PNG인지 검증한다."""

    if not path.is_file():
        raise FileNotFoundError(f"Screenshot does not exist: {path}")
    if path.suffix.lower() != ".png":
        raise ValueError(f"Screenshot extension must be .png: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Screenshot must not be empty: {path}")

    try:
        with Image.open(path) as image:
            image.load()
            image_format = image.format
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Screenshot is not a readable image: {path}") from exc

    if image_format != "PNG":
        raise ValueError(f"Screenshot decoded format must be PNG: {path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Screenshot dimensions must be positive: {path}")

    return {
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": path.stat().st_size,
    }


def _relative_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def finalize_visual_validation_payload(
    payload: Mapping[str, Any],
    *,
    screenshot_paths: Sequence[Path],
) -> dict[str, Any]:
    """검증 결과를 복사하여 육안 검증 완료 Metadata를 추가한다."""

    _validate_existing_integration_payload(payload)
    if len(screenshot_paths) != 2:
        raise ValueError("Exactly two screenshots are required.")

    screenshot_artifacts: list[dict[str, Any]] = []
    for label, path in zip(("NORMAL", "DEFECT"), screenshot_paths, strict=True):
        metadata = validate_png_screenshot(path)
        screenshot_artifacts.append(
            {
                "label": label,
                "path": _relative_project_path(path),
                **metadata,
            }
        )

    finalized_payload = dict(payload)
    finalized_payload["ui_visual_validation_completed"] = True
    finalized_payload["ui_visual_validation_result"] = "PASS"
    finalized_payload["screenshot_artifacts"] = screenshot_artifacts
    return finalized_payload


def write_json_atomically(*, path: Path, payload: Mapping[str, Any]) -> None:
    """임시 파일에 쓴 뒤 교체하여 중간 상태 JSON 생성을 방지한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    payload = read_json_object(VALIDATION_ARTIFACT_PATH)
    finalized_payload = finalize_visual_validation_payload(
        payload,
        screenshot_paths=DEFAULT_SCREENSHOT_PATHS,
    )
    write_json_atomically(
        path=VALIDATION_ARTIFACT_PATH,
        payload=finalized_payload,
    )

    print("=" * 92)
    print("DAY 8 - STREAMLIT DASHBOARD VISUAL VALIDATION FINALIZATION")
    print("=" * 92)
    print("[PASS] Existing FastAPI integration validation confirmed")
    for artifact in finalized_payload["screenshot_artifacts"]:
        print(
            f"[PASS] {artifact['label']} screenshot: {artifact['path']} "
            f"({artifact['width']} x {artifact['height']}, "
            f"{artifact['size_bytes']} bytes)"
        )
    print("[PASS] UI visual validation status updated")
    print(f"[ARTIFACT] {VALIDATION_ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
