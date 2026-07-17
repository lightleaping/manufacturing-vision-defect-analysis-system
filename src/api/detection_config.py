"""Day 13 Detection API의 모델·Checkpoint·Threshold 설정."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Final

from src.api.config import PROJECT_ROOT
from src.detection.model_config import SUPPORTED_ARCHITECTURE


DEFAULT_DETECTION_CHECKPOINT_PATH: Final[Path] = (
    PROJECT_ROOT
    / "models"
    / "detection"
    / "day12_detection_best.pt"
)


@dataclass(frozen=True, slots=True)
class DetectionApiSettings:
    """Detection 추론 서비스가 공유하는 고정 설정.

    Classification 설정과 책임을 섞지 않기 위해 별도 설정 객체로 유지한다.
    """

    model_name: str = "FasterRCNNMobileNetV3Large320FPN"
    model_version: str = "day12_detection_best"
    architecture: str = SUPPORTED_ARCHITECTURE
    device: str = "cpu"
    checkpoint_path: Path = DEFAULT_DETECTION_CHECKPOINT_PATH

    default_score_threshold: float = 0.5
    minimum_score_threshold: float = 0.05
    maximum_score_threshold: float = 0.95
    iou_threshold: float = 0.5

    checkpoint_metric_name: str = "map_50"

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty str.")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty str.")
        if self.architecture != SUPPORTED_ARCHITECTURE:
            raise ValueError(
                f"Unsupported detection architecture: {self.architecture!r}."
            )
        if self.device != "cpu":
            raise ValueError("The verified Day 13 Detection API path is CPU only.")
        if not isinstance(self.checkpoint_path, Path):
            raise TypeError("checkpoint_path must be pathlib.Path.")
        if self.checkpoint_path.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("checkpoint_path must use .pt or .pth extension.")

        minimum = _finite_float(
            self.minimum_score_threshold,
            "minimum_score_threshold",
        )
        maximum = _finite_float(
            self.maximum_score_threshold,
            "maximum_score_threshold",
        )
        default = _finite_float(
            self.default_score_threshold,
            "default_score_threshold",
        )
        iou = _finite_float(self.iou_threshold, "iou_threshold")

        if not 0.0 <= minimum <= 1.0:
            raise ValueError("minimum_score_threshold must be in [0, 1].")
        if not 0.0 <= maximum <= 1.0:
            raise ValueError("maximum_score_threshold must be in [0, 1].")
        if minimum > maximum:
            raise ValueError(
                "minimum_score_threshold must not exceed maximum_score_threshold."
            )
        if not minimum <= default <= maximum:
            raise ValueError(
                "default_score_threshold must be within the configured range."
            )
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1].")
        if (
            not isinstance(self.checkpoint_metric_name, str)
            or not self.checkpoint_metric_name.strip()
        ):
            raise ValueError("checkpoint_metric_name must be a non-empty str.")


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def resolve_score_threshold(
    value: str | int | float | None,
    *,
    settings: DetectionApiSettings,
) -> float:
    """Query String 또는 내부 숫자를 검증된 Score Threshold로 변환한다."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return float(settings.default_score_threshold)
    if isinstance(value, bool):
        raise ValueError("score_threshold must be a finite number.")

    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("score_threshold must be a finite number.") from error

    if not math.isfinite(threshold):
        raise ValueError("score_threshold must be a finite number.")
    if not (
        settings.minimum_score_threshold
        <= threshold
        <= settings.maximum_score_threshold
    ):
        raise ValueError(
            "score_threshold must be between "
            f"{settings.minimum_score_threshold:.2f} and "
            f"{settings.maximum_score_threshold:.2f}."
        )
    return threshold


DEFAULT_DETECTION_API_SETTINGS = DetectionApiSettings()
