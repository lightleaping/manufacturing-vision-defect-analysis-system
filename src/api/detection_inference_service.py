"""Day 13 Faster R-CNN Detection 추론과 안전한 Prediction 정규화."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from threading import Lock
import time
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.transforms.functional import pil_to_tensor

from src.api.detection_config import (
    DEFAULT_DETECTION_API_SETTINGS,
    DetectionApiSettings,
    resolve_score_threshold,
)
from src.api.image_validation import ValidatedImage
from src.api.schemas import (
    DetectionBox,
    DetectionItem,
    DetectionPredictionResponse,
)


class DetectionInferenceServiceError(RuntimeError):
    """외부에 안전한 Code·Message로 변환 가능한 Detection 오류."""

    def __init__(self, *, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class DetectionInferenceService:
    """앱 Lifespan 동안 한 번 로딩한 Faster R-CNN 추론 서비스.

    모델은 요청마다 다시 만들지 않는다. 같은 Process의 동시 CPU Forward가
    겹치지 않도록 Lock으로 보호하고, 모든 추론은 ``torch.inference_mode``에서
    실행한다.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        checkpoint_epoch_index: int,
        checkpoint_metric_value: float,
        class_mapping: Mapping[str, int],
        settings: DetectionApiSettings = DEFAULT_DETECTION_API_SETTINGS,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module.")
        if (
            not isinstance(checkpoint_epoch_index, int)
            or isinstance(checkpoint_epoch_index, bool)
            or checkpoint_epoch_index < 0
        ):
            raise ValueError("checkpoint_epoch_index must be a non-negative int.")
        if isinstance(checkpoint_metric_value, bool):
            raise TypeError("checkpoint_metric_value must be numeric.")
        metric_value = float(checkpoint_metric_value)
        if not math.isfinite(metric_value) or not 0.0 <= metric_value <= 1.0:
            raise ValueError("checkpoint_metric_value must be finite and in [0, 1].")
        if not isinstance(settings, DetectionApiSettings):
            raise TypeError("settings must be DetectionApiSettings.")

        normalized_mapping = _validate_class_mapping(class_mapping)
        self._index_to_class = {
            class_index: class_name
            for class_name, class_index in normalized_mapping.items()
        }
        self._model = model.to(torch.device(settings.device))
        self._model.eval()
        self._checkpoint_epoch_index = checkpoint_epoch_index
        self._checkpoint_metric_value = metric_value
        self._class_mapping = normalized_mapping
        self._settings = settings
        self._forward_lock = Lock()

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def device_name(self) -> str:
        return self._settings.device

    @property
    def checkpoint_epoch(self) -> int:
        """사람이 읽는 1-based Epoch 번호."""

        return self._checkpoint_epoch_index + 1

    def predict(
        self,
        validated_image: ValidatedImage,
        *,
        score_threshold: str | int | float | None = None,
    ) -> DetectionPredictionResponse:
        """검증된 RGB 이미지에서 Class·Score·원본 좌표 Box를 반환한다."""

        if not isinstance(validated_image, ValidatedImage):
            raise TypeError("validated_image must be ValidatedImage.")

        try:
            threshold = resolve_score_threshold(
                score_threshold,
                settings=self._settings,
            )
        except ValueError as error:
            raise DetectionInferenceServiceError(
                code="INVALID_SCORE_THRESHOLD",
                message=str(error),
                status_code=400,
            ) from error

        image_tensor = _validated_image_to_tensor(validated_image)
        model_input = image_tensor.to(torch.device(self._settings.device))

        try:
            started = time.perf_counter()
            with self._forward_lock:
                with torch.inference_mode():
                    raw_output = self._model([model_input])
            inference_time_ms = max(
                0.0,
                (time.perf_counter() - started) * 1000.0,
            )
            prediction = _extract_single_prediction(raw_output)
            detections = self._normalize_detections(
                prediction=prediction,
                image_width=validated_image.original_width,
                image_height=validated_image.original_height,
                score_threshold=threshold,
            )
        except DetectionInferenceServiceError:
            raise
        except Exception as error:
            raise DetectionInferenceServiceError(
                code="DETECTION_INFERENCE_FAILED",
                message="Detection 모델 추론 중 내부 오류가 발생했습니다.",
                status_code=500,
            ) from error

        return DetectionPredictionResponse(
            detections=detections,
            detection_count=len(detections),
            score_threshold=threshold,
            iou_threshold=float(self._settings.iou_threshold),
            model_name=self._settings.model_name,
            model_version=self._settings.model_version,
            architecture=self._settings.architecture,
            device=self._settings.device,
            checkpoint_epoch=self.checkpoint_epoch,
            checkpoint_metric_name=self._settings.checkpoint_metric_name,
            checkpoint_metric_value=self._checkpoint_metric_value,
            original_filename=validated_image.original_filename,
            content_type=validated_image.content_type,
            image_width=validated_image.original_width,
            image_height=validated_image.original_height,
            image_mode=validated_image.original_mode,
            model_input_mode="RGB",
            inference_time_ms=inference_time_ms,
        )

    def _normalize_detections(
        self,
        *,
        prediction: Mapping[str, Tensor],
        image_width: int,
        image_height: int,
        score_threshold: float,
    ) -> list[DetectionItem]:
        boxes = prediction["boxes"].detach().cpu()
        labels = prediction["labels"].detach().cpu()
        scores = prediction["scores"].detach().cpu()

        _validate_prediction_tensors(boxes=boxes, labels=labels, scores=scores)

        normalized: list[DetectionItem] = []
        for index in range(int(scores.shape[0])):
            score = float(scores[index].item())
            label_id = int(labels[index].item())
            raw_box = boxes[index].tolist()

            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise _invalid_model_output("Prediction score must be finite and in [0, 1].")
            if label_id == 0 or label_id not in self._index_to_class:
                raise _invalid_model_output(
                    f"Prediction label_id is outside the configured defect range: {label_id}."
                )
            if len(raw_box) != 4 or any(not math.isfinite(float(v)) for v in raw_box):
                raise _invalid_model_output("Prediction box must contain four finite values.")

            xmin = min(max(float(raw_box[0]), 0.0), float(image_width))
            ymin = min(max(float(raw_box[1]), 0.0), float(image_height))
            xmax = min(max(float(raw_box[2]), 0.0), float(image_width))
            ymax = min(max(float(raw_box[3]), 0.0), float(image_height))

            if xmax <= xmin or ymax <= ymin:
                raise _invalid_model_output(
                    "Prediction box is invalid after clamping to the original image."
                )

            if score < score_threshold:
                continue

            normalized.append(
                DetectionItem(
                    label_id=label_id,
                    label_name=self._index_to_class[label_id],
                    score=score,
                    box=DetectionBox(
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                    ),
                )
            )

        normalized.sort(
            key=lambda item: (
                -item.score,
                item.label_id,
                item.box.ymin,
                item.box.xmin,
            )
        )
        return normalized


def _validate_class_mapping(
    class_mapping: Mapping[str, int],
) -> dict[str, int]:
    if not isinstance(class_mapping, Mapping):
        raise TypeError("class_mapping must be a mapping.")
    normalized = dict(class_mapping)
    if normalized.get("BACKGROUND") != 0:
        raise ValueError("class_mapping must reserve BACKGROUND=0.")
    if any(not isinstance(name, str) or not name for name in normalized):
        raise ValueError("Every class name must be a non-empty str.")
    if any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in normalized.values()
    ):
        raise ValueError("Every class index must be a non-negative int.")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("class_mapping indexes must be unique.")
    if set(normalized.values()) != set(range(len(normalized))):
        raise ValueError("class_mapping indexes must be contiguous from 0.")
    if len(normalized) < 2:
        raise ValueError("class_mapping must include at least one defect class.")
    return normalized


def _validated_image_to_tensor(validated_image: ValidatedImage) -> Tensor:
    image = validated_image.rgb_image.copy()
    if image.mode != "RGB":
        raise DetectionInferenceServiceError(
            code="INVALID_MODEL_INPUT",
            message="Detection 모델 입력은 RGB 이미지여야 합니다.",
            status_code=500,
        )

    tensor = pil_to_tensor(image).to(dtype=torch.float32).div(255.0)
    if tensor.ndim != 3 or int(tensor.shape[0]) != 3:
        raise DetectionInferenceServiceError(
            code="INVALID_MODEL_INPUT",
            message="Detection 모델 입력 Tensor 형식이 올바르지 않습니다.",
            status_code=500,
        )
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise DetectionInferenceServiceError(
            code="INVALID_MODEL_INPUT",
            message="Detection 모델 입력 Tensor가 비어 있거나 유효하지 않습니다.",
            status_code=500,
        )
    if float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
        raise DetectionInferenceServiceError(
            code="INVALID_MODEL_INPUT",
            message="Detection 모델 입력 범위가 올바르지 않습니다.",
            status_code=500,
        )
    return tensor.contiguous()


def _extract_single_prediction(raw_output: Any) -> Mapping[str, Tensor]:
    if isinstance(raw_output, Tensor) or not isinstance(raw_output, Sequence):
        raise _invalid_model_output("Model output must be a sequence.")
    if len(raw_output) != 1:
        raise _invalid_model_output("Single-image inference must return one prediction.")
    prediction = raw_output[0]
    if not isinstance(prediction, Mapping):
        raise _invalid_model_output("Prediction must be a mapping.")
    required = {"boxes", "labels", "scores"}
    missing = required - set(prediction)
    if missing:
        raise _invalid_model_output(
            f"Prediction is missing keys: {sorted(missing)}."
        )
    for key in required:
        if not isinstance(prediction[key], Tensor):
            raise _invalid_model_output(f"Prediction {key} must be torch.Tensor.")
    return prediction


def _validate_prediction_tensors(
    *,
    boxes: Tensor,
    labels: Tensor,
    scores: Tensor,
) -> None:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise _invalid_model_output("Prediction boxes must have shape [N, 4].")
    if labels.ndim != 1:
        raise _invalid_model_output("Prediction labels must have shape [N].")
    if scores.ndim != 1:
        raise _invalid_model_output("Prediction scores must have shape [N].")
    if not (boxes.shape[0] == labels.shape[0] == scores.shape[0]):
        raise _invalid_model_output(
            "Prediction boxes, labels, and scores counts must match."
        )
    if labels.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise _invalid_model_output("Prediction labels must use an integer dtype.")


def _invalid_model_output(message: str) -> DetectionInferenceServiceError:
    return DetectionInferenceServiceError(
        code="INVALID_DETECTION_MODEL_OUTPUT",
        message=message,
        status_code=500,
    )
