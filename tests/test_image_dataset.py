"""
CastingDefectDataset 테스트.

이 테스트 파일의 책임
---------------------
1. Dataset 생성 입력을 올바르게 검증하는지 확인한다.
2. 이미지 Path와 Label 오류를 확인한다.
3. Dataset의 __len__() 동작을 확인한다.
4. Dataset의 __getitem__()이 실제 이미지를 읽는지 확인한다.
5. 모든 이미지를 RGB 3채널로 변환하는지 확인한다.
6. Transform 결과가 올바른 PyTorch 이미지 Tensor인지 확인한다.
7. 이미지 읽기 오류를 프로젝트 수준 RuntimeError로 변환하는지 확인한다.
8. 실제 Casting 이미지가 Dataset을 정상적으로 통과하는지 확인한다.

현재 단계
---------
정식 Train·Validation·Test Transform은 아직 구현하지 않았다.

따라서 정상 이미지 변환 테스트에서는 torchvision의 ToTensor()를
임시로 사용한다.

정식 Resize·Normalize·Data Augmentation 검증은 다음 Transform
구현 단계에서 별도로 작성한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToTensor

from src.data.dataset_config import (
    TRAIN_ROOT,
)
from src.data.dataset_split import (
    ImageSample,
    collect_image_samples,
)
from src.data.image_dataset import (
    CastingDefectDataset,
)


def create_test_image(
    image_path: Path,
    mode: str = "RGB",
    size: tuple[int, int] = (
        20,
        10,
    ),
) -> None:
    """
    Dataset 테스트에 사용할 실제 이미지 파일을 생성한다.

    Parameters
    ----------
    image_path:
        생성할 이미지 파일 Path다.

    mode:
        Pillow 이미지 Mode다.

        기본값:

            RGB

        Grayscale → RGB 변환 테스트에서는:

            L

        을 전달한다.

    size:
        Pillow 이미지 크기다.

        Pillow의 크기 순서는 다음과 같다.

            Width × Height

        기본값:

            Width  = 20

            Height = 10

    처리
    ----
    1. 부모 디렉터리를 생성한다.
    2. 이미지 Mode에 맞는 색상 값을 결정한다.
    3. Pillow Image를 생성한다.
    4. 실제 파일로 저장한다.

    필요한 이유
    -----------
    Dataset의 __getitem__()은 실제로 Image.open()을 실행한다.

    따라서 정상 이미지 읽기 테스트에서는 빈 파일이 아니라
    Pillow가 읽을 수 있는 실제 이미지 파일이 필요하다.
    """

    image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if mode == "L":
        # Grayscale은 하나의 밝기 값만 사용한다.
        color: int | tuple[int, ...] = 128

    elif mode == "RGBA":
        # RGBA는 Red·Green·Blue·Alpha 네 값을 사용한다.
        color = (
            10,
            20,
            30,
            255,
        )

    else:
        # RGB는 Red·Green·Blue 세 값을 사용한다.
        color = (
            10,
            20,
            30,
        )

    image = Image.new(
        mode=mode,
        size=size,
        color=color,
    )

    image.save(
        image_path
    )


def create_image_sample(
    image_path: Path,
    label: int = 0,
) -> ImageSample:
    """
    Dataset 테스트용 ImageSample을 생성한다.

    Parameters
    ----------
    image_path:
        ImageSample에 저장할 이미지 Path다.

    label:
        이미지 Label이다.

        기본값:

            0 → NORMAL

    Returns
    -------
    ImageSample
        이미지 Path와 Label이 연결된 Sample이다.
    """

    return ImageSample(
        image_path=image_path,
        label=label,
    )


def test_dataset_rejects_empty_samples() -> None:
    """
    Sample이 하나도 없으면 ValueError가 발생하는지 검증한다.

    빈 Dataset은 모델 학습·평가에 사용할 수 없으므로
    Dataset 생성 단계에서 거부해야 한다.
    """

    with pytest.raises(
        ValueError,
        match="Dataset Sample이 비어 있습니다",
    ):
        CastingDefectDataset(
            samples=[],
            transform=ToTensor(),
        )


def test_dataset_rejects_non_callable_transform(
    tmp_path: Path,
) -> None:
    """
    Transform이 호출 가능한 객체가 아니면 TypeError가 발생하는지 검증한다.
    """

    image_path = (
        tmp_path
        / "normal.png"
    )

    create_test_image(
        image_path=image_path,
    )

    with pytest.raises(
        TypeError,
        match=(
            "transform은 호출 가능한 "
            "객체여야 합니다"
        ),
    ):
        CastingDefectDataset(
            samples=[
                create_image_sample(
                    image_path=image_path,
                ),
            ],
            # 런타임 입력 검증을 확인하기 위해
            # 의도적으로 None을 전달한다.
            transform=None,  # type: ignore[arg-type]
        )


def test_dataset_rejects_empty_allowed_labels(
    tmp_path: Path,
) -> None:
    """
    허용 Label 집합이 비어 있으면 ValueError가 발생하는지 검증한다.
    """

    image_path = (
        tmp_path
        / "normal.png"
    )

    create_test_image(
        image_path=image_path,
    )

    with pytest.raises(
        ValueError,
        match="allowed_labels가 비어 있습니다",
    ):
        CastingDefectDataset(
            samples=[
                create_image_sample(
                    image_path=image_path,
                ),
            ],
            transform=ToTensor(),
            allowed_labels=set(),
        )


def test_dataset_rejects_non_image_sample(
    tmp_path: Path,
) -> None:
    """
    Dataset 항목이 ImageSample이 아니면 TypeError가 발생하는지 검증한다.
    """

    with pytest.raises(
        TypeError,
        match=(
            "Dataset의 모든 항목은 "
            "ImageSample이어야 합니다"
        ),
    ):
        CastingDefectDataset(
            # Dataset 런타임 검증을 확인하기 위해
            # 의도적으로 object를 전달한다.
            samples=[
                object(),
            ],  # type: ignore[list-item]
            transform=ToTensor(),
        )


def test_dataset_rejects_boolean_label(
    tmp_path: Path,
) -> None:
    """
    Boolean Label을 정수 Label로 허용하지 않는지 검증한다.

    Python에서 bool은 int의 하위 타입이다.

        isinstance(True, int)

        → True

    그러나 현재 프로젝트 Label은 실제 정수 0 또는 1이어야 한다.

    따라서 Dataset은 type(label) is int 기준으로 검증한다.
    """

    image_path = (
        tmp_path
        / "normal.png"
    )

    create_test_image(
        image_path=image_path,
    )

    boolean_label_sample = ImageSample(
        image_path=image_path,
        # 런타임 검증을 확인하기 위해
        # 의도적으로 bool을 전달한다.
        label=True,  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match=(
            "ImageSample label은 "
            "int여야 합니다"
        ),
    ):
        CastingDefectDataset(
            samples=[
                boolean_label_sample,
            ],
            transform=ToTensor(),
        )


def test_dataset_rejects_unknown_label(
    tmp_path: Path,
) -> None:
    """
    현재 프로젝트에 정의되지 않은 Label을 거부하는지 검증한다.

    허용 Label:

        0 → NORMAL

        1 → DEFECT

    테스트 Label:

        2 → 정의되지 않음
    """

    image_path = (
        tmp_path
        / "unknown.png"
    )

    create_test_image(
        image_path=image_path,
    )

    with pytest.raises(
        ValueError,
        match="허용되지 않은 Label입니다",
    ):
        CastingDefectDataset(
            samples=[
                create_image_sample(
                    image_path=image_path,
                    label=2,
                ),
            ],
            transform=ToTensor(),
        )


def test_dataset_rejects_missing_image_file(
    tmp_path: Path,
) -> None:
    """
    존재하지 않는 이미지 Path에서 FileNotFoundError가 발생하는지 검증한다.
    """

    missing_image_path = (
        tmp_path
        / "missing.png"
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "이미지 파일을 "
            "찾을 수 없습니다"
        ),
    ):
        CastingDefectDataset(
            samples=[
                create_image_sample(
                    image_path=(
                        missing_image_path
                    ),
                ),
            ],
            transform=ToTensor(),
        )


def test_dataset_rejects_directory_as_image_path(
    tmp_path: Path,
) -> None:
    """
    이미지 Path가 파일이 아니라 디렉터리이면 ValueError가 발생하는지 검증한다.
    """

    image_directory = (
        tmp_path
        / "image_directory"
    )

    image_directory.mkdir()

    with pytest.raises(
        ValueError,
        match=(
            "이미지 Path는 "
            "파일이어야 합니다"
        ),
    ):
        CastingDefectDataset(
            samples=[
                create_image_sample(
                    image_path=(
                        image_directory
                    ),
                ),
            ],
            transform=ToTensor(),
        )


def test_dataset_rejects_duplicate_image_path(
    tmp_path: Path,
) -> None:
    """
    같은 이미지 Path가 Dataset에 두 번 포함되면 ValueError가 발생하는지 검증한다.

    중복 이미지는 클래스 분포와 모델 학습 결과를 왜곡할 수 있다.
    """

    image_path = (
        tmp_path
        / "duplicate.png"
    )

    create_test_image(
        image_path=image_path,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Dataset에 중복 이미지 "
            "Path가 있습니다"
        ),
    ):
        CastingDefectDataset(
            samples=[
                create_image_sample(
                    image_path=image_path,
                    label=0,
                ),
                create_image_sample(
                    image_path=image_path,
                    label=1,
                ),
            ],
            transform=ToTensor(),
        )


def test_dataset_length_and_read_only_properties(
    tmp_path: Path,
) -> None:
    """
    __len__(), samples, allowed_labels Property가 올바른 값을 반환하는지 검증한다.
    """

    normal_path = (
        tmp_path
        / "normal.png"
    )

    defect_path = (
        tmp_path
        / "defect.png"
    )

    create_test_image(
        image_path=normal_path,
    )

    create_test_image(
        image_path=defect_path,
    )

    samples = [
        create_image_sample(
            image_path=normal_path,
            label=0,
        ),
        create_image_sample(
            image_path=defect_path,
            label=1,
        ),
    ]

    dataset = CastingDefectDataset(
        samples=samples,
        transform=ToTensor(),
    )

    assert len(dataset) == 2

    # Dataset 내부 Sample은 외부 list와 분리된 Tuple로 저장된다.
    assert isinstance(
        dataset.samples,
        tuple,
    )

    assert dataset.samples == tuple(
        samples
    )

    # 현재 기본 허용 Label은 0과 1이다.
    assert dataset.allowed_labels == (
        frozenset(
            {
                0,
                1,
            }
        )
    )


def test_getitem_converts_grayscale_to_rgb_tensor(
    tmp_path: Path,
) -> None:
    """
    Grayscale 이미지가 RGB 3채널 Tensor로 변환되는지 검증한다.

    원본:
        Pillow Mode L

        Width  = 20

        Height = 10

    기대 Tensor:
        [Channel, Height, Width]

        [3, 10, 20]
    """

    grayscale_path = (
        tmp_path
        / "grayscale.png"
    )

    create_test_image(
        image_path=grayscale_path,
        mode="L",
        size=(
            20,
            10,
        ),
    )

    dataset = CastingDefectDataset(
        samples=[
            create_image_sample(
                image_path=(
                    grayscale_path
                ),
                label=1,
            ),
        ],
        transform=ToTensor(),
    )

    image_tensor, label = dataset[0]

    assert isinstance(
        image_tensor,
        Tensor,
    )

    assert image_tensor.shape == (
        3,
        10,
        20,
    )

    assert image_tensor.dtype == (
        torch.float32
    )

    assert label == 1

    # 원본 Grayscale의 하나의 밝기 Channel이
    # RGB 세 Channel에 동일하게 복제되었는지 확인한다.
    assert torch.equal(
        image_tensor[0],
        image_tensor[1],
    )

    assert torch.equal(
        image_tensor[1],
        image_tensor[2],
    )


def test_getitem_rejects_non_integer_index(
    tmp_path: Path,
) -> None:
    """
    int가 아닌 Index를 전달하면 TypeError가 발생하는지 검증한다.
    """

    image_path = (
        tmp_path
        / "normal.png"
    )

    create_test_image(
        image_path=image_path,
    )

    dataset = CastingDefectDataset(
        samples=[
            create_image_sample(
                image_path=image_path,
            ),
        ],
        transform=ToTensor(),
    )

    with pytest.raises(
        TypeError,
        match=(
            "Dataset index는 "
            "int여야 합니다"
        ),
    ):
        _ = dataset[
            "0"  # type: ignore[index]
        ]


@pytest.mark.parametrize(
    "invalid_index",
    [
        -1,
        1,
    ],
)
def test_getitem_rejects_out_of_range_index(
    tmp_path: Path,
    invalid_index: int,
) -> None:
    """
    Dataset 범위를 벗어난 Index에서 IndexError가 발생하는지 검증한다.

    Dataset Size:

        1

    유효 Index:

        0

    잘못된 Index:

        -1

        1
    """

    image_path = (
        tmp_path
        / "normal.png"
    )

    create_test_image(
        image_path=image_path,
    )

    dataset = CastingDefectDataset(
        samples=[
            create_image_sample(
                image_path=image_path,
            ),
        ],
        transform=ToTensor(),
    )

    with pytest.raises(
        IndexError,
        match=(
            "Dataset index가 "
            "범위를 벗어났습니다"
        ),
    ):
        _ = dataset[
            invalid_index
        ]


def test_getitem_wraps_unreadable_image_error(
    tmp_path: Path,
) -> None:
    """
    파일은 존재하지만 이미지가 아니면 RuntimeError가 발생하는지 검증한다.

    Dataset 생성 시:
        Path 존재

        파일 여부 정상

    __getitem__() 호출 시:
        Pillow가 이미지로 식별하지 못함

        RuntimeError로 변환
    """

    broken_image_path = (
        tmp_path
        / "broken.jpeg"
    )

    broken_image_path.write_text(
        "this is not a valid image",
        encoding="utf-8",
    )

    dataset = CastingDefectDataset(
        samples=[
            create_image_sample(
                image_path=(
                    broken_image_path
                ),
            ),
        ],
        transform=ToTensor(),
    )

    with pytest.raises(
        RuntimeError,
        match="이미지를 읽을 수 없습니다",
    ):
        _ = dataset[0]


def test_getitem_rejects_non_tensor_transform_result(
    tmp_path: Path,
) -> None:
    """
    Transform이 Tensor가 아닌 값을 반환하면 TypeError가 발생하는지 검증한다.
    """

    image_path = (
        tmp_path
        / "normal.png"
    )

    create_test_image(
        image_path=image_path,
    )

    def return_pillow_image(
        image: Image.Image,
    ) -> Image.Image:
        """
        오류 검증을 위해 Pillow Image를 그대로 반환한다.
        """

        return image

    dataset = CastingDefectDataset(
        samples=[
            create_image_sample(
                image_path=image_path,
            ),
        ],
        transform=return_pillow_image,  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match=(
            "Transform 결과는 "
            "torch.Tensor여야 합니다"
        ),
    ):
        _ = dataset[0]


def test_getitem_rejects_non_three_dimensional_tensor(
    tmp_path: Path,
) -> None:
    """
    Transform 결과가 [C, H, W] 3차원이 아니면 ValueError가 발생하는지 검증한다.
    """

    image_path = (
        tmp_path
        / "normal.png"
    )

    create_test_image(
        image_path=image_path,
    )

    def return_two_dimensional_tensor(
        image: Image.Image,
    ) -> Tensor:
        """
        오류 검증을 위해 [H, W] 2차원 Tensor를 반환한다.
        """

        return torch.zeros(
            (
                image.height,
                image.width,
            ),
            dtype=torch.float32,
        )

    dataset = CastingDefectDataset(
        samples=[
            create_image_sample(
                image_path=image_path,
            ),
        ],
        transform=(
            return_two_dimensional_tensor
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "이미지 Tensor는 3차원"
        ),
    ):
        _ = dataset[0]


def test_getitem_rejects_non_rgb_channel_tensor(
    tmp_path: Path,
) -> None:
    """
    Transform 결과의 Channel 수가 3이 아니면 ValueError가 발생하는지 검증한다.
    """

    image_path = (
        tmp_path
        / "normal.png"
    )

    create_test_image(
        image_path=image_path,
    )

    def return_single_channel_tensor(
        image: Image.Image,
    ) -> Tensor:
        """
        오류 검증을 위해 [1, H, W] Tensor를 반환한다.
        """

        return torch.zeros(
            (
                1,
                image.height,
                image.width,
            ),
            dtype=torch.float32,
        )

    dataset = CastingDefectDataset(
        samples=[
            create_image_sample(
                image_path=image_path,
            ),
        ],
        transform=(
            return_single_channel_tensor
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "이미지 Tensor의 Channel은 "
            "3이어야 합니다"
        ),
    ):
        _ = dataset[0]


def test_getitem_rejects_non_floating_tensor(
    tmp_path: Path,
) -> None:
    """
    Transform 결과가 Floating Point Tensor가 아니면 ValueError가 발생하는지 검증한다.
    """

    image_path = (
        tmp_path
        / "normal.png"
    )

    create_test_image(
        image_path=image_path,
    )

    def return_integer_tensor(
        image: Image.Image,
    ) -> Tensor:
        """
        오류 검증을 위해 uint8 Tensor를 반환한다.
        """

        return torch.zeros(
            (
                3,
                image.height,
                image.width,
            ),
            dtype=torch.uint8,
        )

    dataset = CastingDefectDataset(
        samples=[
            create_image_sample(
                image_path=image_path,
            ),
        ],
        transform=return_integer_tensor,
    )

    with pytest.raises(
        ValueError,
        match=(
            "이미지 Tensor는 "
            "Floating Point"
        ),
    ):
        _ = dataset[0]


def test_real_dataset_first_sample_returns_expected_tensor() -> None:
    """
    실제 Casting Train 이미지 한 장이 Dataset을 정상 통과하는지 검증한다.

    Day 1 확인 결과
    ----------------
    원본 이미지:

        300 × 300

        RGB

        3 Channel

    현재 Transform:

        ToTensor()

    기대 결과:

        Tensor Shape

            [3, 300, 300]

        dtype

            torch.float32

        Pixel Range

            0.0 ~ 1.0

        첫 Sample Label

            0 → NORMAL
    """

    all_samples = collect_image_samples(
        data_root=TRAIN_ROOT,
    )

    first_sample = (
        all_samples[0]
    )

    dataset = CastingDefectDataset(
        # 실제 전체 이미지 Path는 collect 단계에서 확인했다.
        #
        # 이번 테스트의 목적은 실제 이미지 한 장의
        # 로드·RGB·Tensor 흐름을 확인하는 것이므로
        # 첫 Sample만 Dataset에 전달한다.
        samples=[
            first_sample,
        ],
        transform=ToTensor(),
    )

    image_tensor, label = dataset[0]

    assert len(dataset) == 1

    assert image_tensor.shape == (
        3,
        300,
        300,
    )

    assert image_tensor.dtype == (
        torch.float32
    )

    assert (
        image_tensor.min().item()
        >= 0.0
    )

    assert (
        image_tensor.max().item()
        <= 1.0
    )

    assert label == 0

    assert (
        dataset.samples[0].image_path
        == first_sample.image_path
    )