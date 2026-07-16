"""검증된 RGB 이미지에 Test Transform과 이진 분류 모델을 적용한다."""

from __future__ import annotations

import math
from collections.abc import Callable
from time import perf_counter

import torch
from PIL import Image
from torch import Tensor, nn

from src.api.config import ApiSettings, DEFAULT_API_SETTINGS
from src.api.image_validation import ValidatedImage
from src.api.schemas import PredictionResponse


class InferenceServiceError(RuntimeError):
    """모델 입력 준비, 추론, 출력 검증 실패."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int = 500,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ImageInferenceService:
    """단일 제조 이미지의 NORMAL·DEFECT 추론 Service."""

    def __init__(
        self,
        *,
        model: nn.Module,
        transform: Callable[[Image.Image], Tensor],
        device: str | torch.device = "cpu",
        settings: ApiSettings = DEFAULT_API_SETTINGS,
    ) -> None:
        threshold = float(settings.classification_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "classification_threshold must be finite and between 0 and 1"
            )

        if not callable(transform):
            raise TypeError("transform must be callable")

        self.settings = settings
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.transform = transform

        # Service 생성 시 한 번만 Evaluation Mode로 전환한다.
        self.model.eval()
        self.is_ready = True

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    @property
    def model_version(self) -> str:
        return self.settings.model_version

    @property
    def device_name(self) -> str:
        return str(self.device)

    def _prepare_batch(self, image: Image.Image) -> Tensor:
        transformed = self.transform(image)

        if not isinstance(transformed, Tensor):
            raise InferenceServiceError(
                code="INVALID_MODEL_INPUT",
                message="이미지 Transform 결과가 Tensor가 아닙니다.",
            )

        # Day 2 create_test_transform() 결과는 [C, H, W]다.
        if transformed.ndim != 3:
            raise InferenceServiceError(
                code="INVALID_MODEL_INPUT",
                message="이미지 Transform 출력 Shape가 올바르지 않습니다.",
            )

        if not torch.isfinite(transformed).all().item():
            raise InferenceServiceError(
                code="INVALID_MODEL_INPUT",
                message="이미지 Transform 결과에 NaN 또는 Infinity가 있습니다.",
            )

        batch = transformed.unsqueeze(0)
        if batch.ndim != 4 or batch.shape[0] != 1:
            raise InferenceServiceError(
                code="INVALID_MODEL_INPUT",
                message="모델 입력 Batch Shape가 올바르지 않습니다.",
            )

        return batch.to(self.device)

    @staticmethod
    def _validate_model_output(output: object) -> Tensor:
        if not isinstance(output, Tensor):
            raise InferenceServiceError(
                code="INVALID_MODEL_OUTPUT",
                message="모델 출력이 Tensor가 아닙니다.",
            )

        # ResNet18Transfer의 Batch Size 1 출력 [1]을 기본으로 사용한다.
        # 일반적인 Binary Head의 [1, 1]도 같은 의미이므로 허용한다.
        if tuple(output.shape) not in {(1,), (1, 1)}:
            raise InferenceServiceError(
                code="INVALID_MODEL_OUTPUT",
                message="모델 출력 Shape가 올바르지 않습니다.",
            )

        return output.reshape(-1)

    def predict(self, image: ValidatedImage) -> PredictionResponse:
        """Raw Logit → Sigmoid → Threshold 순서로 추론한다."""

        started_at = perf_counter()
        batch = self._prepare_batch(image.rgb_image)

        try:
            # Prediction API에서는 Gradient와 Backward가 필요하지 않다.
            with torch.inference_mode():
                raw_output = self.model(batch)
        except Exception as exc:
            raise InferenceServiceError(
                code="INFERENCE_FAILED",
                message="모델 추론 중 오류가 발생했습니다.",
            ) from exc

        output = self._validate_model_output(raw_output)

        raw_logit = float(output[0].item())
        if not math.isfinite(raw_logit):
            raise InferenceServiceError(
                code="INVALID_MODEL_OUTPUT",
                message="모델 Raw Logit이 유한한 값이 아닙니다.",
            )

        defect_probability = float(torch.sigmoid(output[0]).item())
        if not math.isfinite(defect_probability):
            raise InferenceServiceError(
                code="INVALID_MODEL_OUTPUT",
                message="모델 확률 출력이 유한한 값이 아닙니다.",
            )

        if not 0.0 <= defect_probability <= 1.0:
            raise InferenceServiceError(
                code="INVALID_MODEL_OUTPUT",
                message="모델 확률 출력 범위가 올바르지 않습니다.",
            )

        threshold = float(self.settings.classification_threshold)
        prediction = 1 if defect_probability >= threshold else 0
        prediction_class_name = "DEFECT" if prediction == 1 else "NORMAL"
        normal_probability = 1.0 - defect_probability
        inference_time_ms = (perf_counter() - started_at) * 1000.0

        return PredictionResponse(
            prediction=prediction,
            prediction_class_name=prediction_class_name,
            defect_probability=defect_probability,
            normal_probability=normal_probability,
            raw_logit=raw_logit,
            classification_threshold=threshold,
            model_name=self.settings.model_name,
            model_version=self.settings.model_version,
            positive_class=self.settings.positive_class,
            original_filename=image.original_filename,
            content_type=image.content_type,
            image_width=image.original_width,
            image_height=image.original_height,
            image_mode=image.original_mode,
            inference_time_ms=inference_time_ms,
        )
