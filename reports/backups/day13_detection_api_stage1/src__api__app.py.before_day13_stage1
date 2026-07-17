"""FastAPI Application Factory, Lifespan, Health·Prediction Endpoint."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.config import ApiSettings, DEFAULT_API_SETTINGS
from src.api.image_validation import ImageValidationError, validate_uploaded_image
from src.api.inference_service import InferenceServiceError
from src.api.schemas import ErrorResponse, HealthResponse, PredictionResponse


ServiceFactory = Callable[[], Any]
_DEFAULT_FACTORY = object()


def _create_default_service() -> Any:
    """순환 Import와 불필요한 Module 초기화를 피하기 위한 지연 Import."""

    from src.api.model_loader import create_production_inference_service

    return create_production_inference_service()


def _error_content(*, code: str, message: str) -> dict[str, dict[str, str]]:
    return {
        "detail": {
            "code": code,
            "message": message,
        }
    }


def create_app(
    *,
    service_factory: ServiceFactory | None | object = _DEFAULT_FACTORY,
    settings: ApiSettings = DEFAULT_API_SETTINGS,
) -> FastAPI:
    """Production과 Test가 같은 API 구조를 사용하도록 App을 생성한다.

    service_factory 생략:
        실제 ResNet18 Production Service를 Lifespan에서 한 번 로딩한다.

    service_factory=None:
        Model Not Ready 상태를 의도적으로 만든다.

    Dummy Factory 전달:
        실제 Checkpoint 없이 API 통합 테스트를 수행한다.
    """

    if service_factory is _DEFAULT_FACTORY:
        resolved_service_factory: ServiceFactory | None = _create_default_service
    elif service_factory is None or callable(service_factory):
        resolved_service_factory = service_factory
    else:
        raise TypeError("service_factory must be callable, None, or omitted")

    @asynccontextmanager
    async def lifespan(current_app: FastAPI):
        current_app.state.inference_service = None
        current_app.state.model_startup_failed = False

        if resolved_service_factory is not None:
            try:
                # Process 시작 시 한 번만 Model·Checkpoint·Transform을 로딩한다.
                current_app.state.inference_service = resolved_service_factory()
            except Exception:
                # 내부 경로, Checkpoint 정보, Stack Trace는 HTTP 응답에 노출하지 않는다.
                current_app.state.inference_service = None
                current_app.state.model_startup_failed = True

        yield

        current_app.state.inference_service = None

    application = FastAPI(
        title=settings.service_name,
        version=settings.api_version,
        description=(
            "ResNet18 Best Checkpoint를 사용하여 제조 이미지를 "
            "NORMAL 또는 DEFECT로 분류하는 API"
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request, exc
        return JSONResponse(
            status_code=400,
            content=_error_content(
                code="MISSING_FILE",
                message="multipart/form-data의 file 필드가 필요합니다.",
            ),
        )

    @application.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["Health"],
    )
    async def health(request: Request) -> HealthResponse:
        service = getattr(request.app.state, "inference_service", None)
        model_loaded = bool(
            service is not None
            and getattr(service, "is_ready", False)
        )

        return HealthResponse(
            status="ok",
            service=settings.service_name,
            model_loaded=model_loaded,
            model_name=getattr(service, "model_name", settings.model_name),
            device=getattr(service, "device_name", settings.device),
        )

    @application.post(
        "/api/v1/predictions",
        response_model=PredictionResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["Predictions"],
    )
    async def create_prediction(
        request: Request,
        file: UploadFile = File(...),
    ) -> PredictionResponse:
        service = getattr(request.app.state, "inference_service", None)

        if service is None or not getattr(service, "is_ready", False):
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "MODEL_NOT_READY",
                    "message": "추론 모델이 준비되지 않았습니다.",
                },
            )

        try:
            validated_image = await validate_uploaded_image(
                file,
                settings=settings,
            )
            return service.predict(validated_image)

        except ImageValidationError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": exc.message},
            ) from exc

        except InferenceServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": exc.message},
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "INFERENCE_FAILED",
                    "message": "이미지 추론 중 내부 오류가 발생했습니다.",
                },
            ) from exc

        finally:
            await file.close()

    return application


# Uvicorn에서 import할 실제 Production App.
# Model은 이 줄에서 로딩되지 않고 Lifespan 시작 시 한 번만 로딩된다.
app = create_app()
