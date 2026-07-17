from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"


def test_dashboard_app_renders_core_single_page_widgets(monkeypatch) -> None:
    # UI 구조 테스트에서는 실제 FastAPI와 Checkpoint를 사용하지 않는다.
    monkeypatch.setenv("MVDA_DASHBOARD_HEALTH_CHECK_ENABLED", "false")

    app = AppTest.from_file(APP_PATH, default_timeout=10).run()

    assert len(app.exception) == 0
    assert len(app.file_uploader) == 1
    assert len(app.button) == 1
    assert app.button[0].label == "이미지 분석 실행"
    assert app.button[0].disabled is True

    markdown_values = [element.value for element in app.markdown]
    assert any(
        "Manufacturing Vision Defect Analysis System" in value
        for value in markdown_values
    )

    subheaders = [element.value for element in app.subheader]
    assert "FastAPI 상태" in subheaders
    assert "제조 이미지 업로드" in subheaders
    assert "결과 해석과 주의사항" in subheaders


def test_dashboard_does_not_show_prediction_before_button_click(monkeypatch) -> None:
    monkeypatch.setenv("MVDA_DASHBOARD_HEALTH_CHECK_ENABLED", "false")

    app = AppTest.from_file(APP_PATH, default_timeout=10).run()

    metric_labels = [element.label for element in app.metric]
    assert "P(DEFECT)" not in metric_labels
    assert "P(NORMAL)" not in metric_labels
    assert any(
        "이미지를 업로드" in element.value
        for element in app.info
    )
