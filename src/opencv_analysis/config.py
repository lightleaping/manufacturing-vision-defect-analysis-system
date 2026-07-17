"""OpenCV 분석 파이프라인의 결정론적 파라미터 설정."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _validate_positive_odd_pair(name: str, value: tuple[int, int]) -> None:
    """Kernel처럼 양수 홀수 두 개가 필요한 설정을 검증한다."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a tuple of two integers")

    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} values must be integers")
        if item <= 0 or item % 2 == 0:
            raise ValueError(f"{name} values must be positive odd integers")


def _validate_positive_pair(name: str, value: tuple[int, int]) -> None:
    """CLAHE tile grid처럼 양수 두 개가 필요한 설정을 검증한다."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a tuple of two integers")

    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} values must be integers")
        if item <= 0:
            raise ValueError(f"{name} values must be positive integers")


@dataclass(frozen=True, slots=True)
class OpenCVAnalysisConfig:
    """OpenCV 처리 단계의 파라미터를 한곳에서 관리한다.

    기본 설정은 작은 표면 결함 이미지와 Casting 이미지 모두에서 구조 검증이
    가능하도록 보수적으로 선택했다. 실제 결함 탐지 성능을 보장하는 값이 아니다.
    """

    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    gaussian_kernel_size: tuple[int, int] = (5, 5)
    gaussian_sigma_x: float = 0.0

    canny_low_threshold: int = 50
    canny_high_threshold: int = 150

    adaptive_threshold_block_size: int = 11
    adaptive_threshold_c: float = 2.0
    adaptive_threshold_invert: bool = True

    morphology_kernel_size: tuple[int, int] = (3, 3)
    morphology_open_iterations: int = 1
    morphology_close_iterations: int = 1

    min_contour_area_ratio: float = 0.0005
    max_contours: int = 500
    contour_line_thickness: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.clahe_clip_limit, (int, float)) or isinstance(
            self.clahe_clip_limit, bool
        ):
            raise TypeError("clahe_clip_limit must be a number")
        if self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be greater than 0")
        _validate_positive_pair("clahe_tile_grid_size", self.clahe_tile_grid_size)

        _validate_positive_odd_pair("gaussian_kernel_size", self.gaussian_kernel_size)
        if not isinstance(self.gaussian_sigma_x, (int, float)) or isinstance(
            self.gaussian_sigma_x, bool
        ):
            raise TypeError("gaussian_sigma_x must be a number")
        if self.gaussian_sigma_x < 0:
            raise ValueError("gaussian_sigma_x must be greater than or equal to 0")

        for name, value in (
            ("canny_low_threshold", self.canny_low_threshold),
            ("canny_high_threshold", self.canny_high_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be between 0 and 255")
        if self.canny_low_threshold >= self.canny_high_threshold:
            raise ValueError("canny_low_threshold must be lower than canny_high_threshold")

        if isinstance(self.adaptive_threshold_block_size, bool) or not isinstance(
            self.adaptive_threshold_block_size, int
        ):
            raise TypeError("adaptive_threshold_block_size must be an integer")
        if (
            self.adaptive_threshold_block_size <= 1
            or self.adaptive_threshold_block_size % 2 == 0
        ):
            raise ValueError(
                "adaptive_threshold_block_size must be an odd integer greater than 1"
            )
        if not isinstance(self.adaptive_threshold_c, (int, float)) or isinstance(
            self.adaptive_threshold_c, bool
        ):
            raise TypeError("adaptive_threshold_c must be a number")
        if not isinstance(self.adaptive_threshold_invert, bool):
            raise TypeError("adaptive_threshold_invert must be a boolean")

        _validate_positive_odd_pair(
            "morphology_kernel_size", self.morphology_kernel_size
        )
        for name, value in (
            ("morphology_open_iterations", self.morphology_open_iterations),
            ("morphology_close_iterations", self.morphology_close_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be greater than or equal to 0")

        if not isinstance(self.min_contour_area_ratio, (int, float)) or isinstance(
            self.min_contour_area_ratio, bool
        ):
            raise TypeError("min_contour_area_ratio must be a number")
        if not 0 <= self.min_contour_area_ratio <= 1:
            raise ValueError("min_contour_area_ratio must be between 0 and 1")

        if isinstance(self.max_contours, bool) or not isinstance(self.max_contours, int):
            raise TypeError("max_contours must be an integer")
        if self.max_contours <= 0:
            raise ValueError("max_contours must be greater than 0")

        if isinstance(self.contour_line_thickness, bool) or not isinstance(
            self.contour_line_thickness, int
        ):
            raise TypeError("contour_line_thickness must be an integer")
        if self.contour_line_thickness <= 0:
            raise ValueError("contour_line_thickness must be greater than 0")

    def to_dict(self) -> dict[str, Any]:
        """Artifact JSON에 기록할 수 있도록 기본 Python 타입으로 변환한다."""
        result = asdict(self)
        # JSON에서 tuple은 list로 직렬화되지만 명시적으로 바꿔 구조를 고정한다.
        result["clahe_tile_grid_size"] = list(self.clahe_tile_grid_size)
        result["gaussian_kernel_size"] = list(self.gaussian_kernel_size)
        result["morphology_kernel_size"] = list(self.morphology_kernel_size)
        return result
