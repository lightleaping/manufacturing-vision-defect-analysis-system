from __future__ import annotations

import asyncio
from io import BytesIO

import pytest
from PIL import Image

from src.api.config import ApiSettings
from src.api.image_validation import (
    ImageValidationError,
    decode_and_validate_image,
    validate_uploaded_image,
)


def _make_image_bytes(
    *,
    image_format: str,
    mode: str = "RGB",
    size: tuple[int, int] = (32, 24),
) -> bytes:
    if mode == "RGBA":
        color = (10, 20, 30, 120)
    elif mode == "L":
        color = 128
    else:
        color = (10, 20, 30)

    image = Image.new(mode, size, color=color)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    image.close()
    return buffer.getvalue()


class _AsyncUpload:
    def __init__(self, *, filename: str, content_type: str, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._buffer = BytesIO(data)

    async def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


@pytest.mark.parametrize(
    ("image_format", "filename", "content_type"),
    [
        ("JPEG", "sample.jpg", "image/jpeg"),
        ("JPEG", "sample.jpeg", "image/jpeg"),
        ("PNG", "sample.png", "image/png"),
    ],
)
def test_decode_supported_image(
    image_format: str,
    filename: str,
    content_type: str,
) -> None:
    result = decode_and_validate_image(
        data=_make_image_bytes(image_format=image_format),
        filename=filename,
        content_type=content_type,
    )

    assert result.decoded_format == image_format
    assert result.original_width == 32
    assert result.original_height == 24
    assert result.rgb_image.mode == "RGB"
    result.rgb_image.close()


@pytest.mark.parametrize("mode", ["RGBA", "L"])
def test_non_rgb_image_is_converted_to_rgb(mode: str) -> None:
    result = decode_and_validate_image(
        data=_make_image_bytes(image_format="PNG", mode=mode),
        filename="sample.png",
        content_type="image/png",
    )

    assert result.original_mode == mode
    assert result.rgb_image.mode == "RGB"
    result.rgb_image.close()


def test_empty_file_is_rejected() -> None:
    with pytest.raises(ImageValidationError) as exc_info:
        decode_and_validate_image(
            data=b"",
            filename="empty.jpg",
            content_type="image/jpeg",
        )

    assert exc_info.value.code == "EMPTY_FILE"
    assert exc_info.value.status_code == 400


def test_corrupted_image_is_rejected() -> None:
    with pytest.raises(ImageValidationError) as exc_info:
        decode_and_validate_image(
            data=b"not-a-real-image",
            filename="broken.jpg",
            content_type="image/jpeg",
        )

    assert exc_info.value.code == "INVALID_IMAGE"


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(ImageValidationError) as exc_info:
        decode_and_validate_image(
            data=_make_image_bytes(image_format="PNG"),
            filename="sample.gif",
            content_type="image/gif",
        )

    assert exc_info.value.code == "UNSUPPORTED_FILE_TYPE"
    assert exc_info.value.status_code == 415


def test_extension_and_content_type_mismatch_is_rejected() -> None:
    with pytest.raises(ImageValidationError) as exc_info:
        decode_and_validate_image(
            data=_make_image_bytes(image_format="JPEG"),
            filename="sample.jpg",
            content_type="image/png",
        )

    assert exc_info.value.code == "UNSUPPORTED_FILE_TYPE"


def test_metadata_and_actual_format_mismatch_is_rejected() -> None:
    with pytest.raises(ImageValidationError) as exc_info:
        decode_and_validate_image(
            data=_make_image_bytes(image_format="PNG"),
            filename="sample.jpg",
            content_type="image/jpeg",
        )

    assert exc_info.value.code == "UNSUPPORTED_FILE_TYPE"


def test_upload_size_limit_is_enforced() -> None:
    upload = _AsyncUpload(
        filename="large.jpg",
        content_type="image/jpeg",
        data=b"x" * 101,
    )
    settings = ApiSettings(max_upload_bytes=100)

    with pytest.raises(ImageValidationError) as exc_info:
        asyncio.run(
            validate_uploaded_image(
                upload,  # type: ignore[arg-type]
                settings=settings,
            )
        )

    assert exc_info.value.code == "FILE_TOO_LARGE"
    assert exc_info.value.status_code == 413


def test_pixel_limit_is_enforced() -> None:
    settings = ApiSettings(max_image_pixels=399)

    with pytest.raises(ImageValidationError) as exc_info:
        decode_and_validate_image(
            data=_make_image_bytes(image_format="PNG", size=(20, 20)),
            filename="large.png",
            content_type="image/png",
            settings=settings,
        )

    assert exc_info.value.code == "IMAGE_TOO_LARGE"
    assert exc_info.value.status_code == 413
