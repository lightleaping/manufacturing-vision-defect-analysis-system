"""Day 12 Detection Checkpoint 재개와 Best Metric 선택 보조 함수.

[기존 코드 참고]
- ``src.detection.checkpoint``의 latest·best 저장 및 CPU 복원 계약을 유지한다.
- ``src.detection.training_config``의 Best Metric 이름을 그대로 사용한다.

[신규 구현]
- Checkpoint 재개 뒤 Optimizer Learning Rate를 Unfreeze 단계 값으로 낮춘다.
- Validation Metric에서 Best Checkpoint 비교 값을 안전하게 추출한다.
- mAP·F1·Recall은 큰 값, Validation Loss는 작은 값을 Best로 판단한다.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from torch.optim import Optimizer


_HIGHER_IS_BETTER = frozenset({"map_50", "f1", "recall"})
_LOWER_IS_BETTER = frozenset({"validation_loss"})


def set_optimizer_learning_rate(
    optimizer: Optimizer,
    *,
    learning_rate: float,
) -> tuple[float, ...]:
    """모든 Optimizer Parameter Group의 Learning Rate를 같은 값으로 맞춘다."""
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer.")
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise ValueError("learning_rate must be a positive finite number.")

    value = float(learning_rate)
    for group in optimizer.param_groups:
        group["lr"] = value
        # PyTorch Optimizer가 initial_lr를 가진 경우 Scheduler 없이도
        # Artifact와 재개 상태가 현재 단계의 Learning Rate를 반영하게 한다.
        if "initial_lr" in group:
            group["initial_lr"] = value
    return tuple(float(group["lr"]) for group in optimizer.param_groups)


def extract_best_metric_value(
    *,
    metric_name: str,
    validation_summary: Mapping[str, Any],
) -> float:
    """Validation Summary에서 Best 비교에 사용할 유한한 Scalar를 꺼낸다."""
    if metric_name not in _HIGHER_IS_BETTER | _LOWER_IS_BETTER:
        raise ValueError(f"Unsupported best metric: {metric_name!r}.")
    if not isinstance(validation_summary, Mapping):
        raise TypeError("validation_summary must be a mapping.")

    if metric_name == "validation_loss":
        raw_value = validation_summary.get("validation_loss")
    else:
        metrics = validation_summary.get("metrics")
        if not isinstance(metrics, Mapping):
            raise KeyError("validation_summary must contain metrics mapping.")
        overall = metrics.get("overall")
        if not isinstance(overall, Mapping):
            raise KeyError("validation metrics must contain overall mapping.")
        raw_value = overall.get(metric_name)

    if raw_value is None:
        raise ValueError(f"Validation metric {metric_name!r} is None.")
    if (
        not isinstance(raw_value, (int, float))
        or isinstance(raw_value, bool)
        or not math.isfinite(float(raw_value))
    ):
        raise ValueError(
            f"Validation metric {metric_name!r} must be a finite number."
        )
    return float(raw_value)


def is_better_metric(
    *,
    metric_name: str,
    candidate: float,
    current_best: float,
) -> bool:
    """Metric 방향에 맞춰 Candidate가 현재 Best보다 엄격히 나은지 판단한다."""
    if metric_name not in _HIGHER_IS_BETTER | _LOWER_IS_BETTER:
        raise ValueError(f"Unsupported best metric: {metric_name!r}.")
    for name, value in (
        ("candidate", candidate),
        ("current_best", current_best),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be a finite number.")

    if metric_name in _HIGHER_IS_BETTER:
        return float(candidate) > float(current_best)
    return float(candidate) < float(current_best)


def validate_resume_epoch_range(
    *,
    checkpoint_epoch: int,
    total_epochs: int,
) -> range:
    """Checkpoint 다음 Epoch부터 목표 Epoch 직전까지의 실행 범위를 만든다."""
    if (
        not isinstance(checkpoint_epoch, int)
        or isinstance(checkpoint_epoch, bool)
        or checkpoint_epoch < 0
    ):
        raise ValueError("checkpoint_epoch must be a non-negative int.")
    if (
        not isinstance(total_epochs, int)
        or isinstance(total_epochs, bool)
        or total_epochs <= 0
    ):
        raise ValueError("total_epochs must be a positive int.")
    next_epoch = checkpoint_epoch + 1
    if next_epoch >= total_epochs:
        raise ValueError(
            "Checkpoint already reached the requested total_epochs. "
            f"checkpoint_epoch={checkpoint_epoch}, total_epochs={total_epochs}."
        )
    return range(next_epoch, total_epochs)
