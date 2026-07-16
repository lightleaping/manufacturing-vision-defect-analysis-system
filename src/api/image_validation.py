"""업로드 이미지 제한 읽기, 실제 Decode 검증, RGB 변환."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.config import ApiSettings, DEFAULT_API_SETTINGS


_EXTENSION_TO_CONTENT_TYPE = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

_EXTENSION_TO_DECODED_FORMAT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
}

_CONTENT_TYPE_TO_DECODED_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
}


class ImageValidationError(ValueError):
    """클라이언트가 보낸 이미지가 프로젝트 정책을 만족하지 못한 경우."""

    def __init__(self, *, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(slots=True)
class ValidatedImage:
    """검증과 RGB 변환을 마친 이미지 및 원본 Metadata."""

    original_filename: str
    content_type: str
    original_width: int
    original_height: int
    original_mode: str
    decoded_format: str
    rgb_image: Image.Image


def _normalize_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""

    # 부가 Parameter가 있더라도 기본 MIME Type만 비교한다.
    return content_type.split(";", maxsplit=1)[0].strip().lower()


def _extract_suffix(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def validate_upload_metadata(
    *,
    filename: str | None,
    content_type: str | None,
    settings: ApiSettings = DEFAULT_API_SETTINGS,
) -> tuple[str, str]:
    """파일명 확장자와 Content-Type을 교차 검증한다."""

    suffix = _extract_suffix(filename)
    normalized_content_type = _normalize_content_type(content_type)

    if suffix not in settings.allowed_extensions:
        raise ImageValidationError(
            code="UNSUPPORTED_FILE_TYPE",
            message="JPEG, JPG, PNG 형식의 이미지만 업로드할 수 있습니다.",
            status_code=415,
        )

    if normalized_content_type not in settings.allowed_content_types:
        raise ImageValidationError(
            code="UNSUPPORTED_FILE_TYPE",
            message="지원하지 않는 이미지 Content-Type입니다.",
            status_code=415,
        )

    expected_content_type = _EXTENSION_TO_CONTENT_TYPE[suffix]
    if normalized_content_type != expected_content_type:
        raise ImageValidationError(
            code="UNSUPPORTED_FILE_TYPE",
            message="파일 확장자와 Content-Type이 일치하지 않습니다.",
            status_code=415,
        )

    return suffix, normalized_content_type


async def read_upload_file_limited(
    upload_file: UploadFile,
    *,
    max_upload_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> bytes:
    """허용 크기보다 1 byte까지만 읽어 과대 업로드를 차단한다."""

    chunks: list[bytes] = []
    total_size = 0

    while True:
        remaining_to_limit = max_upload_bytes + 1 - total_size
        if remaining_to_limit <= 0:
            break

        current_chunk_size = min(chunk_size, remaining_to_limit)
        chunk = await upload_file.read(current_chunk_size)

        if not chunk:
            break

        chunks.append(chunk)
        total_size += len(chunk)

        if total_size > max_upload_bytes:
            raise ImageValidationError(
                code="FILE_TOO_LARGE",
                message="업로드 파일이 최대 허용 크기를 초과했습니다.",
                status_code=413,
            )

    data = b"".join(chunks)

    if not data:
        raise ImageValidationError(
            code="EMPTY_FILE",
            message="업로드한 파일이 비어 있습니다.",
            status_code=400,
        )

    return data


def decode_and_validate_image(
    *,
    data: bytes,
    filename: str,
    content_type: str,
    settings: ApiSettings = DEFAULT_API_SETTINGS,
) -> ValidatedImage:
    """실제 이미지 형식, 무결성, Pixel 수를 확인하고 RGB로 변환한다."""

    if not data:
        raise ImageValidationError(
            code="EMPTY_FILE",
            message="업로드한 파일이 비어 있습니다.",
            status_code=400,
        )

    suffix, normalized_content_type = validate_upload_metadata(
        filename=filename,
        content_type=content_type,
        settings=settings,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            # verify()로 Container 구조와 손상 여부를 먼저 검사한다.
            with Image.open(BytesIO(data)) as verification_image:
                verification_image.verify()

            # verify() 이후 실제 Pixel을 읽기 위해 새 Image 객체를 연다.
            with Image.open(BytesIO(data)) as decoded_image:
                decoded_format = (decoded_image.format or "").upper()

                if decoded_format not in settings.allowed_decoded_formats:
                    raise ImageValidationError(
                        code="UNSUPPORTED_FILE_TYPE",
                        message="실제 Decode 결과가 지원 이미지 형식이 아닙니다.",
                        status_code=415,
                    )

                expected_from_suffix = _EXTENSION_TO_DECODED_FORMAT[suffix]
                expected_from_content_type = _CONTENT_TYPE_TO_DECODED_FORMAT[
                    normalized_content_type
                ]

                if (
                    decoded_format != expected_from_suffix
                    or decoded_format != expected_from_content_type
                ):
                    raise ImageValidationError(
                        code="UNSUPPORTED_FILE_TYPE",
                        message="파일 Metadata와 실제 이미지 형식이 일치하지 않습니다.",
                        status_code=415,
                    )

                width, height = decoded_image.size
                if width <= 0 or height <= 0:
                    raise ImageValidationError(
                        code="INVALID_IMAGE",
                        message="이미지 크기 정보가 올바르지 않습니다.",
                        status_code=400,
                    )

                if width * height > settings.max_image_pixels:
                    raise ImageValidationError(
                        code="IMAGE_TOO_LARGE",
                        message="이미지 Pixel 수가 최대 허용 범위를 초과했습니다.",
                        status_code=413,
                    )

                original_mode = decoded_image.mode
                decoded_image.load()

                # 원본 객체와 수명 및 메모리를 분리한 RGB 복사본을 만든다.
                rgb_image = decoded_image.convert("RGB").copy()

    except ImageValidationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageValidationError(
            code="INVALID_IMAGE",
            message="업로드한 파일을 정상적인 이미지로 읽을 수 없습니다.",
            status_code=400,
        ) from exc

    return ValidatedImage(
        original_filename=filename,
        content_type=normalized_content_type,
        original_width=width,
        original_height=height,
        original_mode=original_mode,
        decoded_format=decoded_format,
        rgb_image=rgb_image,
    )


async def validate_uploaded_image(
    upload_file: UploadFile,
    *,
    settings: ApiSettings = DEFAULT_API_SETTINGS,
) -> ValidatedImage:
    """FastAPI UploadFile을 제한 읽기하고 검증된 RGB 이미지로 만든다."""

    filename = upload_file.filename or ""
    content_type = upload_file.content_type or ""

    # 허용하지 않는 Metadata는 파일 본문을 읽기 전에 거절한다.
    validate_upload_metadata(
        filename=filename,
        content_type=content_type,
        settings=settings,
    )

    data = await read_upload_file_limited(
        upload_file,
        max_upload_bytes=settings.max_upload_bytes,
    )

    return decode_and_validate_image(
        data=data,
        filename=filename,
        content_type=content_type,
        settings=settings,
    )
