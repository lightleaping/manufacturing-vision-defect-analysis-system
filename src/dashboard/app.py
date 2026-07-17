"""Day 8 — Streamlit Image Inference Dashboard.

실행:
    python -m streamlit run .\\src\\dashboard\\app.py

고정 흐름:
    Browser -> Streamlit -> Day 7 FastAPI -> ResNet18 -> JSON -> Streamlit
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.dashboard.api_client import (
    DashboardApiClient,
    DashboardApiError,
    DashboardHealth,
    DashboardPrediction,
)
from src.dashboard.config import (
    DEFAULT_HEALTH_CACHE_TTL_SECONDS,
    DashboardSettings,
    load_dashboard_settings,
)
from src.dashboard.session_state import (
    ANALYSIS_EXECUTED_KEY,
    HEALTH_ERROR_KEY,
    HEALTH_KEY,
    LAST_UPLOAD_FILENAME_KEY,
    PREDICTION_ERROR_KEY,
    PREDICTION_KEY,
    begin_analysis,
    initialize_dashboard_state,
    save_health,
    save_prediction,
    save_prediction_error,
)
from src.dashboard.styles import inject_dashboard_styles
from src.dashboard.ui_helpers import (
    build_error_message,
    build_image_metadata_caption,
    build_prediction_message,
    format_inference_time,
    format_probability,
    inspect_uploaded_image,
    resolve_content_type,
)


@st.cache_data(ttl=DEFAULT_HEALTH_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_cached_health(settings: DashboardSettings) -> DashboardHealth:
    """짧은 TTL로 Health 상태를 Cache하여 매 rerun의 불필요한 요청을 줄인다."""

    with DashboardApiClient(settings) as client:
        return client.get_health()


def _render_header(settings: DashboardSettings) -> None:
    st.markdown(
        f"""
        <div class="mvda-hero">
            <h1>{settings.project_name}</h1>
            <p><strong>{settings.project_name_ko}</strong></p>
            <p>제조 이미지를 업로드하면 Day 7 FastAPI가 ResNet18 모델로 NORMAL 또는 DEFECT를 추론합니다.</p>
            <p class="mvda-note">Positive Class: DEFECT · Threshold: 0.5 · Streamlit은 모델을 직접 로딩하지 않습니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_health_panel(
    *,
    settings: DashboardSettings,
    health: DashboardHealth | None,
    error: DashboardApiError | None,
) -> None:
    st.subheader("FastAPI 상태")

    if not settings.health_check_enabled:
        st.info("Dashboard Health Check가 환경설정으로 비활성화되어 있습니다.")
        st.caption(f"API Base URL: {settings.api_base_url}")
        return

    if error is not None:
        st.error(build_error_message(error))
        st.caption(f"API Base URL: {settings.api_base_url}")
        return

    if health is None:
        st.warning("FastAPI 상태를 아직 확인하지 못했습니다.")
        return

    if health.model_loaded:
        st.success("FastAPI 연결 정상 · 추론 모델 준비 완료")
    else:
        st.warning("FastAPI는 연결됐지만 추론 모델이 준비되지 않았습니다.")

    columns = st.columns(3)
    columns[0].metric("Model", health.model_name)
    columns[1].metric("Device", health.device)
    columns[2].metric("Model Loaded", "YES" if health.model_loaded else "NO")
    st.caption(f"Service: {health.service} · API Base URL: {settings.api_base_url}")


def _render_upload_preview(uploaded_file: Any) -> tuple[bytes, str]:
    image_bytes = uploaded_file.getvalue()
    content_type = resolve_content_type(
        filename=uploaded_file.name,
        declared_content_type=getattr(uploaded_file, "type", None),
    )

    st.image(image_bytes, caption="업로드 이미지 Preview")
    try:
        metadata = inspect_uploaded_image(
            filename=uploaded_file.name,
            image_bytes=image_bytes,
        )
    except ValueError:
        st.warning(
            "Preview Metadata를 읽지 못했습니다. 최종 이미지 검증은 FastAPI가 수행합니다."
        )
    else:
        st.caption(build_image_metadata_caption(metadata))

    return image_bytes, content_type


def _run_prediction(
    *,
    settings: DashboardSettings,
    filename: str,
    content_type: str,
    image_bytes: bytes,
) -> None:
    begin_analysis(st.session_state, filename=filename)

    try:
        with st.spinner("FastAPI에서 이미지를 분석하고 있습니다..."):
            with DashboardApiClient(settings) as client:
                prediction = client.predict_image(
                    filename=filename,
                    content_type=content_type,
                    image_bytes=image_bytes,
                )
    except DashboardApiError as exc:
        save_prediction_error(st.session_state, error=exc)
    except Exception:
        # 예상하지 못한 내부 오류도 경로나 Stack Trace를 화면에 노출하지 않는다.
        save_prediction_error(
            st.session_state,
            error=DashboardApiError(code="API_REQUEST_ERROR"),
        )
    else:
        save_prediction(st.session_state, prediction=prediction)


def _render_prediction(prediction: DashboardPrediction) -> None:
    st.subheader("Prediction 결과")

    st.markdown(
        f"""
        <div class="mvda-card">
            <div class="mvda-label">Prediction</div>
            <div class="mvda-prediction">{prediction.prediction_class_name}</div>
            <div>{build_prediction_message(prediction)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "P(DEFECT)",
        format_probability(prediction.defect_probability),
    )
    metric_columns[1].metric(
        "P(NORMAL)",
        format_probability(prediction.normal_probability),
    )
    metric_columns[2].metric("Raw Logit", f"{prediction.raw_logit:.6f}")
    metric_columns[3].metric(
        "Inference Time",
        format_inference_time(prediction.inference_time_ms),
    )

    st.caption("DEFECT probability")
    st.progress(prediction.defect_probability)

    with st.expander("Model·Image Metadata", expanded=False):
        st.json(prediction.to_dict())


def _render_prediction_state() -> None:
    error = st.session_state[PREDICTION_ERROR_KEY]
    prediction = st.session_state[PREDICTION_KEY]
    analyzed = bool(st.session_state[ANALYSIS_EXECUTED_KEY])

    if error is not None:
        st.error(build_error_message(error))
        return
    if prediction is not None:
        _render_prediction(prediction)
        return
    if analyzed:
        st.warning("분석 요청은 실행됐지만 표시할 결과가 없습니다.")
    else:
        st.info("이미지를 업로드한 뒤 '이미지 분석 실행' 버튼을 누르세요.")


def _render_explanation() -> None:
    st.divider()
    st.subheader("결과 해석과 주의사항")
    st.markdown(
        """
        - **NORMAL**: 모델이 입력 이미지를 정상 제품 이미지로 분류했습니다.
        - **DEFECT**: 모델이 입력 이미지를 불량 제품 이미지로 분류했습니다.
        - 확률은 FastAPI 응답 값을 그대로 표시하며 Streamlit에서 다시 계산하지 않습니다.
        - Day 8 기본 Dashboard는 빠른 Prediction 결과를 우선 제공합니다.
        - Grad-CAM은 Day 6에서 수행한 별도의 설명 가능성 분석이며 기본 추론과 혼동하지 않습니다.
        """
    )
    st.warning(
        "이 결과는 이미지 분류 모델의 예측이며 실제 생산 공정의 최종 품질 판정을 대체하지 않습니다."
    )


def main() -> None:
    st.set_page_config(
        page_title="Manufacturing Vision Defect Analysis System",
        page_icon="🔎",
        layout="wide",
    )

    settings = load_dashboard_settings()
    initialize_dashboard_state(st.session_state)
    inject_dashboard_styles(st)
    _render_header(settings)

    health: DashboardHealth | None = None
    health_error: DashboardApiError | None = None

    if settings.health_check_enabled:
        try:
            health = fetch_cached_health(settings)
        except DashboardApiError as exc:
            health_error = exc
        except Exception:
            health_error = DashboardApiError(code="API_REQUEST_ERROR")

    save_health(
        st.session_state,
        health=health,
        error=health_error,
    )
    _render_health_panel(
        settings=settings,
        health=st.session_state[HEALTH_KEY],
        error=st.session_state[HEALTH_ERROR_KEY],
    )

    st.divider()
    st.subheader("제조 이미지 업로드")
    uploaded_file = st.file_uploader(
        "JPEG 또는 PNG 이미지를 선택하세요.",
        type=list(settings.accepted_upload_extensions),
        accept_multiple_files=False,
        key="manufacturing_image_uploader",
    )

    image_bytes = b""
    content_type = "application/octet-stream"
    if uploaded_file is not None:
        image_bytes, content_type = _render_upload_preview(uploaded_file)

    analyze_clicked = st.button(
        "이미지 분석 실행",
        type="primary",
        disabled=uploaded_file is None,
        key="run_image_prediction",
    )

    if analyze_clicked and uploaded_file is not None:
        _run_prediction(
            settings=settings,
            filename=uploaded_file.name,
            content_type=content_type,
            image_bytes=image_bytes,
        )

    last_filename = st.session_state[LAST_UPLOAD_FILENAME_KEY]
    if last_filename:
        st.caption(f"마지막 분석 파일: {last_filename}")

    _render_prediction_state()
    _render_explanation()


if __name__ == "__main__":
    main()
