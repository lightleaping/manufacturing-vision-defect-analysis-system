"""Day 7 FastAPI 이미지 추론 API의 고정 설정."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "resnet18_transfer_best.pt"
)


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """API, 업로드 검증, 모델 추론에서 함께 사용하는 설정."""

    service_name: str = "Manufacturing Vision Defect Analysis System"
    api_version: str = "v1"

    model_name: str = "ResNet18Transfer"
    model_version: str = "resnet18_transfer_best"
    positive_class: str = "DEFECT"
    device: str = "cpu"
    classification_threshold: float = 0.5
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH

    # Dataset 원본은 300×300이므로 정상 업로드에는 충분한 여유가 있다.
    max_upload_bytes: int = 10 * 1024 * 1024

    # 압축 해제 후 지나치게 큰 이미지가 메모리를 점유하는 것을 방지한다.
    max_image_pixels: int = 25_000_000

    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    allowed_content_types: tuple[str, ...] = ("image/jpeg", "image/png")
    allowed_decoded_formats: tuple[str, ...] = ("JPEG", "PNG")


DEFAULT_API_SETTINGS = ApiSettings()
