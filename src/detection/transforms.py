"""Detection Image와 Bounding Box를 항상 함께 처리하는 Transform."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F


DetectionTarget = dict[str, Tensor]
DetectionImage = Image.Image | Tensor
DetectionTransform = Callable[
    [DetectionImage, DetectionTarget],
    tuple[DetectionImage, DetectionTarget],
]


def clone_detection_target(target: DetectionTarget) -> DetectionTarget:
    """호출자가 전달한 Target Tensor를 직접 수정하지 않도록 복사한다."""
    if not isinstance(target, dict):
        raise TypeError("target must be dict[str, Tensor].")

    cloned: DetectionTarget = {}
    for key, value in target.items():
        if not isinstance(key, str):
            raise TypeError("target keys must be str.")
        if not isinstance(value, Tensor):
            raise TypeError(f"target[{key!r}] must be torch.Tensor.")
        cloned[key] = value.clone()
    return cloned


class Compose:
    """Image·Target Transform 여러 개를 정의된 순서대로 실행한다."""

    def __init__(self, transforms: Sequence[DetectionTransform]) -> None:
        if isinstance(transforms, (str, bytes)):
            raise TypeError("transforms must be a sequence of callables.")
        self.transforms = tuple(transforms)
        if any(not callable(transform) for transform in self.transforms):
            raise TypeError("Every transform must be callable.")

    def __call__(
        self,
        image: DetectionImage,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        current_image: DetectionImage = image.copy() if isinstance(
            image, Image.Image
        ) else image.clone()
        current_target = clone_detection_target(target)

        for transform in self.transforms:
            current_image, current_target = transform(
                current_image,
                current_target,
            )

        if not isinstance(current_image, Tensor):
            raise TypeError("Final detection image must be torch.Tensor.")
        return current_image, current_target


class ToFloatTensor:
    """Pillow RGB 또는 uint8 Tensor를 float32 [0, 1]로 변환한다."""

    def __call__(
        self,
        image: DetectionImage,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        cloned_target = clone_detection_target(target)

        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            tensor = F.pil_to_tensor(image)
        elif isinstance(image, Tensor):
            tensor = image.clone()
        else:
            raise TypeError("image must be PIL.Image.Image or torch.Tensor.")

        if tensor.ndim != 3:
            raise ValueError("image tensor must have shape [C, H, W].")
        if int(tensor.shape[0]) != 3:
            raise ValueError("image tensor must have exactly 3 RGB channels.")
        if tensor.numel() == 0:
            raise ValueError("image tensor must not be empty.")

        if tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
            minimum = float(tensor.min().item())
            maximum = float(tensor.max().item())
            if minimum < 0.0 or maximum > 1.0:
                raise ValueError(
                    "Floating image tensor must already use range [0, 1]."
                )
        else:
            tensor = tensor.to(dtype=torch.float32).div(255.0)

        return tensor.contiguous(), cloned_target


class RandomHorizontalFlip:
    """Image와 0-based exclusive XYXY Box를 같은 방향으로 뒤집는다."""

    def __init__(self, probability: float = 0.5) -> None:
        if not isinstance(probability, (int, float)) or isinstance(
            probability, bool
        ):
            raise TypeError("probability must be numeric.")
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1].")
        self.probability = probability

    def __call__(
        self,
        image: DetectionImage,
        target: DetectionTarget,
    ) -> tuple[DetectionImage, DetectionTarget]:
        cloned_target = clone_detection_target(target)

        if isinstance(image, Image.Image):
            width = image.width
        elif isinstance(image, Tensor):
            if image.ndim != 3:
                raise ValueError("image tensor must have shape [C, H, W].")
            width = int(image.shape[-1])
        else:
            raise TypeError("image must be PIL.Image.Image or torch.Tensor.")

        should_flip = self.probability == 1.0 or (
            self.probability > 0.0
            and float(torch.rand(1).item()) < self.probability
        )
        if not should_flip:
            unchanged = image.copy() if isinstance(image, Image.Image) else image.clone()
            return unchanged, cloned_target

        boxes = cloned_target.get("boxes")
        if boxes is None:
            raise KeyError("target must contain 'boxes'.")
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("target['boxes'] must have shape [N, 4].")

        flipped_image = F.hflip(image)
        flipped_boxes = boxes.clone()
        if boxes.numel() > 0:
            # exclusive 좌표: [xmin, xmax] -> [W-xmax, W-xmin]
            flipped_boxes[:, 0] = width - boxes[:, 2]
            flipped_boxes[:, 2] = width - boxes[:, 0]
        cloned_target["boxes"] = flipped_boxes
        return flipped_image, cloned_target


def create_detection_transform(
    *,
    training: bool,
    horizontal_flip_probability: float = 0.0,
) -> Compose:
    """Day 11의 결정론적 기본 Transform Pipeline을 만든다.

    ``training=True``여도 기본 Flip 확률은 0이다. Day 12 학습 설정에서만
    명시적으로 확률을 올린다.
    """
    if not isinstance(training, bool):
        raise TypeError("training must be bool.")
    if not isinstance(horizontal_flip_probability, (int, float)) or isinstance(
        horizontal_flip_probability, bool
    ):
        raise TypeError("horizontal_flip_probability must be numeric.")
    if not 0.0 <= float(horizontal_flip_probability) <= 1.0:
        raise ValueError("horizontal_flip_probability must be in [0, 1].")

    transforms: list[DetectionTransform] = []
    if training and horizontal_flip_probability > 0.0:
        transforms.append(
            RandomHorizontalFlip(float(horizontal_flip_probability))
        )
    transforms.append(ToFloatTensor())
    return Compose(transforms)
