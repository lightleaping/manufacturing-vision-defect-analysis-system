"""FastAPI 정상 응답과 오류 응답 Schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """API Process와 Model 준비 상태."""

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


class ErrorDetail(BaseModel):
    """외부에 공개해도 안전한 오류 정보."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """프로젝트 공통 오류 응답."""

    detail: ErrorDetail
