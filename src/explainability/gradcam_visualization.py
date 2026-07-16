from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError


class GradCAMVisualizationError(RuntimeError):
    """Grad-CAM 이미지 로딩, 변환, 저장 과정의 공통 예외입니다."""


@dataclass(frozen=True)
class GradCAMVisualizationRecord:
    """Overview Figure 한 행을 구성하는 시각화 입력입니다."""

    sample_index: int
    image_path: str
    selection_type: str
    ground_truth_class_name: str
    prediction_class_name: str
    defect_probability: float
    target_class: str
    target_layer_name: str
    cam: np.ndarray


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """이미지 파일을 검증한 뒤 RGB uint8 배열로 반환합니다."""

    path = Path(image_path)
    if not path.is_file():
        raise GradCAMVisualizationError(f"이미지 파일이 존재하지 않습니다: {path}")

    try:
        # verify()는 파일 구조만 검사하고 실제 Pixel을 읽지는 않는다.
        with Image.open(path) as image:
            image.verify()

        # verify() 뒤에는 파일을 다시 열어 실제 RGB Pixel을 읽는다.
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            array = np.asarray(rgb_image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise GradCAMVisualizationError(
            f"이미지를 열거나 RGB로 변환할 수 없습니다: {path}"
        ) from exc

    if array.ndim != 3 or array.shape[2] != 3:
        raise GradCAMVisualizationError(
            f"RGB 이미지 Shape가 올바르지 않습니다: {array.shape}"
        )

    return array


def _validate_cam(cam: np.ndarray | torch.Tensor) -> np.ndarray:
    """CAM을 유한한 2차원 float32 배열로 정규화합니다."""

    if isinstance(cam, torch.Tensor):
        cam_array = cam.detach().cpu().numpy()
    else:
        cam_array = np.asarray(cam)

    if cam_array.ndim != 2:
        raise GradCAMVisualizationError(
            f"CAM은 [H, W] 2차원이어야 합니다. 실제 Shape={cam_array.shape}"
        )

    # float64 확장을 막기 위해 모든 CAM 계산은 float32로 고정한다.
    cam_array = cam_array.astype(np.float32, copy=False)

    if not np.isfinite(cam_array).all():
        raise GradCAMVisualizationError("CAM에 NaN 또는 Infinity가 포함되어 있습니다.")

    cam_min = float(cam_array.min())
    cam_max = float(cam_array.max())

    if cam_min < 0.0 or cam_max > 1.0:
        value_range = cam_max - cam_min
        if value_range <= 1e-8:
            raise GradCAMVisualizationError("CAM 값 범위가 0이어서 정규화할 수 없습니다.")
        cam_array = (cam_array - cam_min) / value_range

    if float(cam_array.max()) <= 1e-8:
        raise GradCAMVisualizationError("CAM이 모두 0이어서 시각화할 수 없습니다.")

    return np.clip(cam_array, 0.0, 1.0).astype(np.float32, copy=False)


def resize_cam(
    cam: np.ndarray | torch.Tensor,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """CAM을 원본 이미지 크기로 Bilinear Resize합니다."""

    if height <= 0 or width <= 0:
        raise GradCAMVisualizationError("Resize 대상 height와 width는 0보다 커야 합니다.")

    cam_array = _validate_cam(cam)
    cam_tensor = torch.from_numpy(cam_array).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        cam_tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    resized_array = resized.squeeze(0).squeeze(0).numpy()

    if not np.isfinite(resized_array).all():
        raise GradCAMVisualizationError("Resize된 CAM에 NaN 또는 Infinity가 있습니다.")

    return np.clip(resized_array, 0.0, 1.0).astype(np.float32, copy=False)


def colorize_cam(
    cam: np.ndarray | torch.Tensor,
    *,
    colormap_name: str = "jet",
) -> np.ndarray:
    """0~1 CAM을 RGB Heatmap으로 변환합니다.

    ``jet``은 Grad-CAM 문헌과 사례에서 널리 사용되어 고활성 영역을 빨간색,
    저활성 영역을 파란색으로 빠르게 구분할 수 있어 기본값으로 사용합니다.
    """

    cam_array = _validate_cam(cam)

    try:
        colormap = matplotlib.colormaps.get_cmap(colormap_name)
    except ValueError as exc:
        raise GradCAMVisualizationError(
            f"지원하지 않는 Matplotlib Color Map입니다: {colormap_name}"
        ) from exc

    # Matplotlib Color Map은 RGBA를 반환하므로 Alpha Channel을 제외한다.
    rgba = colormap(cam_array)
    heatmap = np.asarray(rgba[..., :3], dtype=np.float32)

    if heatmap.ndim != 3 or heatmap.shape[2] != 3:
        raise GradCAMVisualizationError(
            f"Heatmap RGB Shape가 올바르지 않습니다: {heatmap.shape}"
        )

    return np.clip(heatmap, 0.0, 1.0).astype(np.float32, copy=False)


def create_gradcam_visuals(
    *,
    original_rgb: np.ndarray,
    cam: np.ndarray | torch.Tensor,
    alpha: float = 0.40,
    colormap_name: str = "jet",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """원본, RGB Heatmap, Overlay를 모두 0~1 float32 RGB 배열로 반환합니다."""

    original = np.asarray(original_rgb)
    if original.ndim != 3 or original.shape[2] != 3:
        raise GradCAMVisualizationError(
            f"original_rgb는 [H, W, 3]이어야 합니다. 실제 Shape={original.shape}"
        )

    if not 0.0 <= alpha <= 1.0:
        raise GradCAMVisualizationError("alpha는 0과 1 사이여야 합니다.")

    if np.issubdtype(original.dtype, np.integer):
        original_float = original.astype(np.float32) / np.float32(255.0)
    else:
        original_float = original.astype(np.float32, copy=False)
        if not np.isfinite(original_float).all():
            raise GradCAMVisualizationError(
                "original_rgb에 NaN 또는 Infinity가 포함되어 있습니다."
            )
        if float(original_float.min()) < 0.0 or float(original_float.max()) > 1.0:
            raise GradCAMVisualizationError(
                "실수형 original_rgb는 0~1 범위여야 합니다."
            )

    height, width = original_float.shape[:2]
    resized_cam = resize_cam(cam, height=height, width=width)
    heatmap = colorize_cam(resized_cam, colormap_name=colormap_name)

    # Python float 연산으로 float64가 되지 않도록 float32 Scalar를 사용한다.
    alpha_value = np.float32(alpha)
    overlay = (
        (np.float32(1.0) - alpha_value) * original_float
        + alpha_value * heatmap
    )

    if not np.isfinite(overlay).all():
        raise GradCAMVisualizationError("Overlay에 NaN 또는 Infinity가 있습니다.")

    return (
        np.clip(original_float, 0.0, 1.0).astype(np.float32, copy=False),
        np.clip(heatmap, 0.0, 1.0).astype(np.float32, copy=False),
        np.clip(overlay, 0.0, 1.0).astype(np.float32, copy=False),
    )


def _to_display_uint8(rgb_image: np.ndarray) -> np.ndarray:
    """Matplotlib 표시용 RGB를 메모리 효율적인 uint8 배열로 변환합니다.

    float RGB를 ``imshow``에 직접 전달하면 Matplotlib이 확대 과정에서 큰
    float64 RGBA 배열을 만들 수 있다. 저장 직전에 uint8로 변환하면 동일한
    시각적 결과를 유지하면서 중간 배열 메모리 사용량을 크게 줄일 수 있다.
    """

    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise GradCAMVisualizationError(
            f"표시용 RGB Shape가 올바르지 않습니다: {image.shape}"
        )

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    image_float = image.astype(np.float32, copy=False)
    if not np.isfinite(image_float).all():
        raise GradCAMVisualizationError(
            "표시용 RGB에 NaN 또는 Infinity가 포함되어 있습니다."
        )
    if float(image_float.min()) < 0.0 or float(image_float.max()) > 1.0:
        raise GradCAMVisualizationError("표시용 실수 RGB는 0~1 범위여야 합니다.")

    display_image = np.rint(image_float * np.float32(255.0)).astype(np.uint8)
    return np.ascontiguousarray(display_image)


def _atomic_save_figure(
    *,
    figure: plt.Figure,
    output_path: str | Path,
    dpi: int,
) -> Path:
    """Figure를 임시 PNG에 저장한 뒤 os.replace로 최종 경로에 교체합니다."""

    destination = Path(output_path)
    if destination.suffix.lower() != ".png":
        raise GradCAMVisualizationError("Grad-CAM Figure 출력은 .png 파일이어야 합니다.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".png",
            prefix=f".{destination.stem}_",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        # bbox_inches='tight'는 저장 시 Figure를 한 번 더 크게 렌더링할 수 있어
        # 저메모리 Windows 환경에서는 사용하지 않는다.
        figure.savefig(
            temporary_path,
            format="png",
            dpi=dpi,
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )
        os.replace(temporary_path, destination)
    except (OSError, MemoryError) as exc:
        raise GradCAMVisualizationError(
            "Grad-CAM Figure 저장에 실패했습니다. "
            "메모리가 부족한 경우 다른 프로그램을 종료하거나 dpi를 낮춰주세요: "
            f"{destination}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    if not destination.is_file() or destination.stat().st_size <= 0:
        raise GradCAMVisualizationError(
            f"저장된 Grad-CAM Figure가 비어 있습니다: {destination}"
        )

    return destination


def save_gradcam_overview(
    *,
    records: Sequence[GradCAMVisualizationRecord],
    output_path: str | Path,
    alpha: float = 0.40,
    colormap_name: str = "jet",
    dpi: int = 120,
    figure_title: str = "Day 6 - ResNet18 Grad-CAM Explainability",
) -> Path:
    """각 표본을 Original / Heatmap / Overlay 3열 Overview로 저장합니다.

    실제 Overview는 최대 7개 표본을 세로로 배치하므로, 포트폴리오 가독성을
    유지하면서도 CPU·메모리 부담을 줄이기 위해 기본 DPI를 120으로 사용합니다.
    """

    if not records:
        raise GradCAMVisualizationError("Grad-CAM Overview에 표시할 표본이 없습니다.")

    if dpi <= 0:
        raise GradCAMVisualizationError("dpi는 0보다 커야 합니다.")

    row_count = len(records)
    figure, axes = plt.subplots(
        nrows=row_count,
        ncols=3,
        figsize=(12.0, max(3.4 * row_count, 4.0)),
        squeeze=False,
    )

    try:
        for row_index, record in enumerate(records):
            original_uint8 = load_rgb_image(record.image_path)
            original, heatmap, overlay = create_gradcam_visuals(
                original_rgb=original_uint8,
                cam=record.cam,
                alpha=alpha,
                colormap_name=colormap_name,
            )

            # Figure 저장 시 발생하는 float64 RGBA 확장을 방지하기 위해
            # 표시용 배열만 uint8 RGB로 변환한다.
            display_original = _to_display_uint8(original)
            display_heatmap = _to_display_uint8(heatmap)
            display_overlay = _to_display_uint8(overlay)

            axes[row_index, 0].imshow(
                display_original,
                interpolation="nearest",
                resample=False,
            )
            axes[row_index, 1].imshow(
                display_heatmap,
                interpolation="nearest",
                resample=False,
            )
            axes[row_index, 2].imshow(
                display_overlay,
                interpolation="nearest",
                resample=False,
            )

            metadata_title = (
                f"{record.selection_type}\n"
                f"sample={record.sample_index} | {Path(record.image_path).name}\n"
                f"GT={record.ground_truth_class_name} | "
                f"Pred={record.prediction_class_name} | "
                f"P(DEFECT)={record.defect_probability:.6f}\n"
                f"Target={record.target_class} | Layer={record.target_layer_name}"
            )

            axes[row_index, 0].set_title(metadata_title, fontsize=8, pad=8)
            axes[row_index, 1].set_title("Grad-CAM Heatmap", fontsize=9, pad=7)
            axes[row_index, 2].set_title(
                f"Overlay (alpha={alpha:.2f})",
                fontsize=9,
                pad=7,
            )

            for column_index in range(3):
                axes[row_index, column_index].axis("off")

            # 다음 행으로 넘어가기 전에 Python 참조를 명시적으로 제거한다.
            del original_uint8
            del original
            del heatmap
            del overlay
            del display_original
            del display_heatmap
            del display_overlay

        figure.suptitle(figure_title, fontsize=14, y=0.995)
        figure.subplots_adjust(
            left=0.02,
            right=0.98,
            bottom=0.02,
            top=0.96,
            hspace=0.62,
            wspace=0.08,
        )

        return _atomic_save_figure(
            figure=figure,
            output_path=output_path,
            dpi=dpi,
        )
    finally:
        # 성공·실패 여부와 관계없이 Axes가 가진 이미지 배열 참조를 해제한다.
        figure.clear()
        plt.close(figure)
        gc.collect()
