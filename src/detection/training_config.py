"""Day 12 Detection 학습·평가 설정.

[기존 코드 참고]
- ``src.detection.model_config.DetectionModelConfig``의 320×320 입력과
  Background 포함 7-Class 정책을 이어서 사용한다.
- ``src.detection.data_loader.DetectionDataLoaderConfig``의 Windows CPU
  안전 설정(num_workers=0, pin_memory=False)을 유지한다.

[신규 구현]
- 학습·평가·Checkpoint 선택에 필요한 값을 하나의 불변 Config로 관리한다.
- JSON Artifact와 Checkpoint에 바로 저장할 수 있도록 ``to_dict``를 제공한다.
- 잘못된 CPU 학습 설정은 실제 학습 전에 즉시 거부한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


OptimizerName = Literal["sgd", "adamw"]
SchedulerName = Literal["none", "step_lr"]
BestMetricName = Literal["map_50", "f1", "recall", "validation_loss"]
DuplicateBoxPolicy = Literal["preserve", "remove_exact"]


@dataclass(frozen=True, slots=True)
class DetectionTrainingConfig:
    """CPU Fine-tuning과 IoU 기반 평가의 단일 설정 원본.

    기본값은 전체 학습을 완료했다는 의미가 아니다. 실제 Epoch 수는 Pilot의
    Batch당 시간·Epoch 예상 시간·Checkpoint 크기를 확인한 뒤 확정한다.
    """

    batch_size: int = 1
    epochs: int = 3
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    optimizer_name: OptimizerName = "sgd"
    scheduler_name: SchedulerName = "step_lr"
    scheduler_step_size: int = 2
    scheduler_gamma: float = 0.1

    score_threshold: float = 0.5
    iou_threshold: float = 0.5
    best_metric_name: BestMetricName = "map_50"

    freeze_backbone_epochs: int = 1
    horizontal_flip_probability: float = 0.5
    duplicate_box_policy: DuplicateBoxPolicy = "preserve"

    min_size: int = 320
    max_size: int = 320
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False
    torch_num_threads: int = 2
    random_seed: int = 42

    def __post_init__(self) -> None:
        self._validate_positive_int("batch_size", self.batch_size)
        self._validate_positive_int("epochs", self.epochs)
        self._validate_positive_float("learning_rate", self.learning_rate)
        self._validate_probability("momentum", self.momentum, include_one=False)
        self._validate_non_negative_float("weight_decay", self.weight_decay)

        if self.optimizer_name not in {"sgd", "adamw"}:
            raise ValueError("optimizer_name must be 'sgd' or 'adamw'.")
        if self.scheduler_name not in {"none", "step_lr"}:
            raise ValueError("scheduler_name must be 'none' or 'step_lr'.")

        self._validate_positive_int(
            "scheduler_step_size",
            self.scheduler_step_size,
        )
        self._validate_probability(
            "scheduler_gamma",
            self.scheduler_gamma,
            include_zero=False,
            include_one=True,
        )
        self._validate_probability(
            "score_threshold",
            self.score_threshold,
            include_zero=True,
            include_one=True,
        )
        self._validate_probability(
            "iou_threshold",
            self.iou_threshold,
            include_zero=False,
            include_one=True,
        )

        if self.best_metric_name not in {
            "map_50",
            "f1",
            "recall",
            "validation_loss",
        }:
            raise ValueError(
                "best_metric_name must be one of "
                "'map_50', 'f1', 'recall', 'validation_loss'."
            )

        if (
            not isinstance(self.freeze_backbone_epochs, int)
            or isinstance(self.freeze_backbone_epochs, bool)
            or self.freeze_backbone_epochs < 0
        ):
            raise ValueError("freeze_backbone_epochs must be a non-negative int.")
        if self.freeze_backbone_epochs > self.epochs:
            raise ValueError(
                "freeze_backbone_epochs must not exceed total epochs."
            )

        self._validate_probability(
            "horizontal_flip_probability",
            self.horizontal_flip_probability,
            include_zero=True,
            include_one=True,
        )
        if self.duplicate_box_policy not in {"preserve", "remove_exact"}:
            raise ValueError(
                "duplicate_box_policy must be 'preserve' or 'remove_exact'."
            )

        self._validate_positive_int("min_size", self.min_size)
        self._validate_positive_int("max_size", self.max_size)
        if self.max_size < self.min_size:
            raise ValueError("max_size must be greater than or equal to min_size.")

        if (
            not isinstance(self.num_workers, int)
            or isinstance(self.num_workers, bool)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative int.")
        for name in (
            "pin_memory",
            "persistent_workers",
            "drop_last",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool.")

        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "persistent_workers requires num_workers greater than zero."
            )
        if self.pin_memory:
            raise ValueError(
                "The verified Day 12 CPU path requires pin_memory=False."
            )
        if self.drop_last:
            raise ValueError(
                "Detection training keeps every NEU-DET sample; drop_last=False."
            )

        self._validate_positive_int("torch_num_threads", self.torch_num_threads)
        if not isinstance(self.random_seed, int) or isinstance(
            self.random_seed,
            bool,
        ):
            raise TypeError("random_seed must be int.")

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive int.")

    @staticmethod
    def _validate_positive_float(name: str, value: float) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math_is_finite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be a positive finite number.")

    @staticmethod
    def _validate_non_negative_float(name: str, value: float) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math_is_finite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be a non-negative finite number.")

    @staticmethod
    def _validate_probability(
        name: str,
        value: float,
        *,
        include_zero: bool = True,
        include_one: bool = True,
    ) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric.")
        numeric = float(value)
        if not math_is_finite(numeric):
            raise ValueError(f"{name} must be finite.")

        lower_ok = numeric >= 0.0 if include_zero else numeric > 0.0
        upper_ok = numeric <= 1.0 if include_one else numeric < 1.0
        if not lower_ok or not upper_ok:
            left = "[" if include_zero else "("
            right = "]" if include_one else ")"
            raise ValueError(f"{name} must be in {left}0, 1{right}.")

    def to_dict(self) -> dict[str, Any]:
        """JSON과 Checkpoint에 저장 가능한 평범한 Dictionary를 반환한다."""
        return asdict(self)


def math_is_finite(value: float) -> bool:
    """``math.isfinite``를 작은 함수로 분리해 테스트와 오류 메시지를 단순화한다."""
    import math

    return math.isfinite(value)
