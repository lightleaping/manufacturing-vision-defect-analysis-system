"""FastAPI 정상 응답과 오류 응답 Schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """API Process와 Classification Model 준비 상태."""

    status: str
    service: str
    model_loaded: bool
    model_name: str
    device: str


class PredictionResponse(BaseModel):
    """정상·불량 이진 분류 결과."""

    prediction: int = Field(ge=0, le=1)
    prediction_class_name: str

    defect_probability: float = Field(ge=0.0, le=1.0)
    normal_probability: float = Field(ge=0.0, le=1.0)
    raw_logit: float
    classification_threshold: float = Field(ge=0.0, le=1.0)

    model_name: str
    model_version: str
    positive_class: str

    original_filename: str
    content_type: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    image_mode: str
    inference_time_ms: float = Field(ge=0.0)


class DetectionBox(BaseModel):
    """원본 업로드 이미지 좌표계의 XYXY Bounding Box."""

    xmin: float = Field(ge=0.0)
    ymin: float = Field(ge=0.0)
    xmax: float = Field(gt=0.0)
    ymax: float = Field(gt=0.0)


class DetectionItem(BaseModel):
    """한 개의 Faster R-CNN 결함 Prediction."""

    label_id: int = Field(ge=1, le=6)
    label_name: str
    score: float = Field(ge=0.0, le=1.0)
    box: DetectionBox


class DetectionPredictionResponse(BaseModel):
    """Detection 모델 Metadata·입력 이미지 정보·Prediction 목록."""

    detections: list[DetectionItem] = Field(default_factory=list)
    detection_count: int = Field(ge=0)

    score_threshold: float = Field(ge=0.05, le=0.95)
    iou_threshold: float = Field(ge=0.0, le=1.0)

    model_name: str
    model_version: str
    architecture: str
    device: str
    checkpoint_epoch: int = Field(gt=0)
    checkpoint_metric_name: str
    checkpoint_metric_value: float = Field(ge=0.0, le=1.0)

    original_filename: str
    content_type: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    image_mode: str
    model_input_mode: str
    inference_time_ms: float = Field(ge=0.0)


class ErrorDetail(BaseModel):
    """외부에 공개해도 안전한 오류 정보."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """프로젝트 공통 오류 응답."""

    detail: ErrorDetail
