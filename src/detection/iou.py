"""Dependency 추가 없이 사용하는 Detection IoU 계산.

[신규 구현]
- Torchvision 내부 연산과 분리된 순수 PyTorch IoU Matrix
- 빈 Tensor, 경계 접촉, 잘못된 XYXY, NaN·inf 방어
- Prediction Matching과 Failure Analysis가 같은 IoU 정의를 재사용
"""

from __future__ import annotations

import torch
from torch import Tensor


def validate_xyxy_boxes(
    boxes: Tensor,
    *,
    name: str = "boxes",
    allow_empty: bool = True,
) -> Tensor:
    """유효한 ``FloatTensor[N, 4]`` XYXY Box인지 검증한다."""
    if not isinstance(boxes, Tensor):
        raise TypeError(f"{name} must be torch.Tensor.")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N, 4].")
    if not allow_empty and boxes.shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not boxes.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating-point dtype.")
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError(f"{name} contains NaN or infinity.")
    if boxes.numel() > 0:
        if not bool((boxes[:, 2] > boxes[:, 0]).all()):
            raise ValueError(f"Every {name} box must satisfy xmax > xmin.")
        if not bool((boxes[:, 3] > boxes[:, 1]).all()):
            raise ValueError(f"Every {name} box must satisfy ymax > ymin.")
    return boxes


def box_iou_matrix(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """두 Box 집합의 모든 Pair IoU를 ``[N, M]`` Matrix로 반환한다."""
    first = validate_xyxy_boxes(boxes1, name="boxes1")
    second = validate_xyxy_boxes(boxes2, name="boxes2")

    output_device = first.device
    dtype = torch.promote_types(first.dtype, second.dtype)
    first = first.to(dtype=dtype)
    second = second.to(device=output_device, dtype=dtype)

    if first.shape[0] == 0 or second.shape[0] == 0:
        return torch.zeros(
            (first.shape[0], second.shape[0]),
            dtype=dtype,
            device=output_device,
        )

    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]

    first_area = (
        (first[:, 2] - first[:, 0])
        * (first[:, 3] - first[:, 1])
    )
    second_area = (
        (second[:, 2] - second[:, 0])
        * (second[:, 3] - second[:, 1])
    )
    union = first_area[:, None] + second_area[None, :] - intersection
    if not bool((union > 0).all()):
        raise RuntimeError("IoU union area must be positive.")
    return intersection / union


def box_iou(box1: Tensor, box2: Tensor) -> float:
    """Box 한 쌍의 IoU를 Python float로 반환한다."""
    if not isinstance(box1, Tensor) or box1.shape != (4,):
        raise ValueError("box1 must have shape [4].")
    if not isinstance(box2, Tensor) or box2.shape != (4,):
        raise ValueError("box2 must have shape [4].")
    value = box_iou_matrix(box1.reshape(1, 4), box2.reshape(1, 4))
    return float(value.item())
