"""Day 13 Detection 전용 Streamlit Page 구성 함수."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.dashboard.config import DashboardSettings
from src.dashboard.detection_api_client import (
    DashboardDetectionPrediction,
    DetectionDashboardApiClient,
    DetectionDashboardApiError,
)
from src.dashboard.detection_session_state import (
    DETECTION_ANALYSIS_EXECUTED_KEY,
    DETECTION_ERROR_KEY,
    DETECTION_LAST_SCORE_THRESHOLD_KEY,
    DETECTION_LAST_UPLOAD_FILENAME_KEY,
    DETECTION_RESULT_KEY,
    begin_detection_analysis,
    initialize_detection_state,
    save_detection_error,
    save_detection_result,
)
from src.dashboard.detection_ui_helpers import (
    build_detection_error_message,
    build_detection_summary_message,
    build_detection_table_rows,
    format_detection_score,
    render_detection_overlay,
)
from src.dashboard.ui_helpers import (
    build_image_metadata_caption,
    format_inference_time,
    inspect_uploaded_image,
    resolve_content_type,
)


def _render_detection_header(
    settings: DashboardSettings,
) -> None:
    st.markdown(
        f"""
        <div class="mvda-hero">
            <h1>{settings.project_name}</h1>
            <p><strong>{settings.project_name_ko}</strong></p>
            <p>Day 12 Faster R-CNN Best Checkpoint의 결함 Class·Score·Bounding Box를 표시합니다.</p>
            <p class="mvda-note">Streamlit은 Checkpoint를 직접 로딩하지 않고 FastAPI Detection Endpoint만 호출합니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_detection_scope_explanation() -> None:
    st.info(
        "Detection Prediction은 학습된 Faster R-CNN의 Class·Score·Bounding Box입니다. "
        "OpenCV의 Threshold·Morphology 기반 Contour 후보나 Ground Truth와 동일하지 않습니다."
    )


def _render_upload_preview(
    uploaded_file: Any,
) -> tuple[bytes, str]:
    image_bytes = uploaded_file.getvalue()
    content_type = resolve_content_type(
        filename=uploaded_file.name,
        declared_content_type=getattr(
            uploaded_file,
            "type",
            None,
        ),
    )

    st.image(
        image_bytes,
        caption="Detection 입력 원본 이미지",
    )
    try:
        metadata = inspect_uploaded_image(
            filename=uploaded_file.name,
            image_bytes=image_bytes,
        )
    except ValueError:
        st.warning(
            "Preview Metadata를 읽지 못했습니다. "
            "최종 이미지 검증은 FastAPI가 수행합니다."
        )
    else:
        st.caption(
            build_image_metadata_caption(metadata)
        )

    return image_bytes, content_type


def _run_detection(
    *,
    settings: DashboardSettings,
    filename: str,
    content_type: str,
    image_bytes: bytes,
    score_threshold: float,
) -> None:
    begin_detection_analysis(
        st.session_state,
        filename=filename,
        score_threshold=score_threshold,
    )

    try:
        with st.spinner(
            "FastAPI에서 Faster R-CNN Detection을 실행하고 있습니다..."
        ):
            with DetectionDashboardApiClient(
                settings
            ) as client:
                result = client.detect_image(
                    filename=filename,
                    content_type=content_type,
                    image_bytes=image_bytes,
                    score_threshold=score_threshold,
                )
    except DetectionDashboardApiError as error:
        save_detection_error(
            st.session_state,
            error=error,
        )
    except Exception:
        save_detection_error(
            st.session_state,
            error=DetectionDashboardApiError(
                code="API_REQUEST_ERROR",
            ),
        )
    else:
        save_detection_result(
            st.session_state,
            result=result,
        )


def _render_detection_result(
    *,
    image_bytes: bytes,
    prediction: DashboardDetectionPrediction,
) -> None:
    st.subheader("Detection 결과")
    st.success(
        build_detection_summary_message(prediction)
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Detection Count",
        prediction.detection_count,
    )
    metric_columns[1].metric(
        "Score Threshold",
        f"{prediction.score_threshold:.2f}",
    )
    metric_columns[2].metric(
        "Inference Time",
        format_inference_time(
            prediction.inference_time_ms
        ),
    )
    metric_columns[3].metric(
        "Checkpoint Epoch",
        prediction.checkpoint_epoch,
    )

    if prediction.detection_count == 0:
        st.image(
            image_bytes,
            caption=(
                "현재 Threshold 이상 Detection 없음 "
                "· 원본 이미지"
            ),
        )
        st.info(
            "Prediction이 없다는 것은 Ground Truth상 결함이 없다는 뜻이 아니라, "
            "현재 Score Threshold 이상으로 반환된 모델 Prediction이 없다는 뜻입니다."
        )
    else:
        original_column, overlay_column = st.columns(2)
        original_column.image(
            image_bytes,
            caption="원본 이미지",
        )
        overlay = render_detection_overlay(
            image_bytes=image_bytes,
            prediction=prediction,
            maximum_boxes=8,
        )
        overlay_column.image(
            overlay,
            caption=(
                "Faster R-CNN Prediction Overlay "
                "· 상위 8개까지 표시"
            ),
        )

        st.subheader("Prediction Table")
        st.dataframe(
            build_detection_table_rows(prediction),
            use_container_width=True,
            hide_index=True,
        )
        top_prediction = prediction.detections[0]
        st.caption(
            "최고 Score Prediction: "
            f"{top_prediction.label_name} "
            f"({format_detection_score(top_prediction.score)})"
        )

    with st.expander(
        "Detection Model·Image Metadata",
        expanded=False,
    ):
        st.json(prediction.to_dict())


def _render_detection_state(
    *,
    image_bytes: bytes,
) -> None:
    error = st.session_state[DETECTION_ERROR_KEY]
    result = st.session_state[DETECTION_RESULT_KEY]
    analyzed = bool(
        st.session_state[
            DETECTION_ANALYSIS_EXECUTED_KEY
        ]
    )

    if error is not None:
        st.error(
            build_detection_error_message(error)
        )
        return
    if result is not None:
        _render_detection_result(
            image_bytes=image_bytes,
            prediction=result,
        )
        return
    if analyzed:
        st.warning(
            "Detection 요청은 실행됐지만 표시할 결과가 없습니다."
        )
    else:
        st.info(
            "이미지를 업로드하고 Score Threshold를 확인한 뒤 "
            "'Detection 실행' 버튼을 누르세요."
        )


def _render_detection_notes() -> None:
    st.divider()
    st.subheader("결과 해석과 주의사항")
    st.markdown(
        """
        - 기본 Score Threshold는 Day 12 평가와 동일한 **0.5**입니다.
        - Slider 조절은 사용자 탐색용이며 Day 12 공식 Test 지표를 다시 계산하지 않습니다.
        - Bounding Box는 FastAPI가 반환한 **원본 업로드 이미지 좌표**를 사용합니다.
        - `crazing`은 Threshold 0.5에서 Recall이 낮으므로 Threshold를 내리면 후보가 늘 수 있지만 False Positive도 늘 수 있습니다.
        - OpenCV Contour 후보는 Threshold·Morphology 기반 보조 분석이며 Detection Prediction과 다릅니다.
        """
    )
    st.warning(
        "Detection 결과는 학습 모델의 Prediction이며 실제 생산 공정의 "
        "최종 품질 판정이나 Ground Truth를 대체하지 않습니다."
    )


def render_detection_page(
    *,
    settings: DashboardSettings,
) -> None:
    """Detection 전용 Page를 렌더링한다."""

    initialize_detection_state(st.session_state)
    _render_detection_header(settings)
    _render_detection_scope_explanation()

    st.divider()
    st.subheader("결함 Detection 이미지 업로드")

    score_threshold = st.slider(
        "Score Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help=(
            "Day 12 공식 평가 기본값은 0.5입니다. "
            "이 Slider는 Prediction 탐색용입니다."
        ),
        key="detection_score_threshold_slider",
    )

    uploaded_file = st.file_uploader(
        "JPEG 또는 PNG 이미지를 선택하세요.",
        type=list(
            settings.accepted_upload_extensions
        ),
        accept_multiple_files=False,
        key="detection_image_uploader",
    )

    image_bytes = b""
    content_type = "application/octet-stream"
    if uploaded_file is not None:
        image_bytes, content_type = (
            _render_upload_preview(uploaded_file)
        )

    detect_clicked = st.button(
        "Detection 실행",
        type="primary",
        disabled=uploaded_file is None,
        key="run_detection_prediction",
    )

    if detect_clicked and uploaded_file is not None:
        _run_detection(
            settings=settings,
            filename=uploaded_file.name,
            content_type=content_type,
            image_bytes=image_bytes,
            score_threshold=score_threshold,
        )

    last_filename = st.session_state[
        DETECTION_LAST_UPLOAD_FILENAME_KEY
    ]
    if last_filename:
        last_threshold = st.session_state[
            DETECTION_LAST_SCORE_THRESHOLD_KEY
        ]
        st.caption(
            f"마지막 Detection 파일: {last_filename} "
            f"· Score Threshold: {last_threshold:.2f}"
        )

    _render_detection_state(
        image_bytes=image_bytes,
    )
    _render_detection_notes()
