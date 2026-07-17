"""Day 8 Streamlit Dashboard 설정.

Streamlit은 모델을 직접 로딩하지 않고 Day 7 FastAPI에 HTTP 요청만 보낸다.
환경변수는 로컬 개발 URL과 Timeout을 코드에서 분리하기 위해 사용한다.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 2.0
DEFAULT_READ_TIMEOUT_SECONDS = 30.0
DEFAULT_HEALTH_CACHE_TTL_SECONDS = 5.0
DEFAULT_HEALTH_CHECK_ENABLED = True


def _parse_positive_float(*, name: str, raw_value: str, default: float) -> float:
    """환경변수의 양수 실수 값을 안전하게 해석한다."""

    candidate = raw_value.strip()
    if not candidate:
        return default

    try:
        value = float(candidate)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc

    if value <= 0.0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _parse_boolean(*, name: str, raw_value: str, default: bool) -> bool:
    """true/false 계열 환경변수를 명확한 Boolean으로 변환한다."""

    candidate = raw_value.strip().lower()
    if not candidate:
        return default
    if candidate in {"1", "true", "yes", "on"}:
        return True
    if candidate in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, 1/0, yes/no, on/off")


@dataclass(frozen=True, slots=True)
class DashboardSettings:
    """Dashboard와 FastAPI Client가 함께 사용하는 고정 설정."""

    project_name: str = "Manufacturing Vision Defect Analysis System"
    project_name_ko: str = "제조 비전 결함 분석 시스템"
    api_base_url: str = DEFAULT_API_BASE_URL
    health_endpoint: str = "/api/v1/health"
    prediction_endpoint: str = "/api/v1/predictions"
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS
    health_check_enabled: bool = DEFAULT_HEALTH_CHECK_ENABLED
    accepted_upload_extensions: tuple[str, ...] = ("jpg", "jpeg", "png")
    accepted_content_types: tuple[str, ...] = ("image/jpeg", "image/png")

    def __post_init__(self) -> None:
        base_url = self.api_base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("api_base_url must start with http:// or https://")
        object.__setattr__(self, "api_base_url", base_url)

        for field_name in (
            "connect_timeout_seconds",
            "read_timeout_seconds",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be greater than 0")

        for field_name in ("health_endpoint", "prediction_endpoint"):
            value = getattr(self, field_name)
            if not value.startswith("/"):
                raise ValueError(f"{field_name} must start with /")


def load_dashboard_settings(
    environ: Mapping[str, str] | None = None,
) -> DashboardSettings:
    """환경변수에서 Dashboard 설정을 읽는다.

    지원 환경변수:
    - MVDA_API_BASE_URL
    - MVDA_API_CONNECT_TIMEOUT_SECONDS
    - MVDA_API_READ_TIMEOUT_SECONDS
    - MVDA_DASHBOARD_HEALTH_CHECK_ENABLED
    """

    source = os.environ if environ is None else environ

    return DashboardSettings(
        api_base_url=source.get(
            "MVDA_API_BASE_URL",
            DEFAULT_API_BASE_URL,
        ),
        connect_timeout_seconds=_parse_positive_float(
            name="MVDA_API_CONNECT_TIMEOUT_SECONDS",
            raw_value=source.get("MVDA_API_CONNECT_TIMEOUT_SECONDS", ""),
            default=DEFAULT_CONNECT_TIMEOUT_SECONDS,
        ),
        read_timeout_seconds=_parse_positive_float(
            name="MVDA_API_READ_TIMEOUT_SECONDS",
            raw_value=source.get("MVDA_API_READ_TIMEOUT_SECONDS", ""),
            default=DEFAULT_READ_TIMEOUT_SECONDS,
        ),
        health_check_enabled=_parse_boolean(
            name="MVDA_DASHBOARD_HEALTH_CHECK_ENABLED",
            raw_value=source.get("MVDA_DASHBOARD_HEALTH_CHECK_ENABLED", ""),
            default=DEFAULT_HEALTH_CHECK_ENABLED,
        ),
    )
