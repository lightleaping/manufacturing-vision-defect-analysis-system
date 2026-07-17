"""Day 8 Streamlit Dashboard의 최소 CSS."""

from __future__ import annotations

from typing import Any

DASHBOARD_CSS = """
<style>
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.mvda-hero {
    padding: 1.25rem 1.4rem;
    border: 1px solid rgba(120, 120, 120, 0.25);
    border-radius: 14px;
    margin-bottom: 1.2rem;
}
.mvda-hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 2rem;
}
.mvda-hero p {
    margin: 0.25rem 0;
    line-height: 1.55;
}
.mvda-card {
    padding: 1rem 1.1rem;
    border: 1px solid rgba(120, 120, 120, 0.25);
    border-radius: 12px;
    margin: 0.5rem 0 1rem 0;
}
.mvda-label {
    font-size: 0.84rem;
    opacity: 0.75;
    margin-bottom: 0.25rem;
}
.mvda-prediction {
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0.15rem 0 0.4rem 0;
}
.mvda-note {
    font-size: 0.92rem;
    line-height: 1.55;
    opacity: 0.85;
}
</style>
"""


def inject_dashboard_styles(st_module: Any) -> None:
    """Streamlit Module을 인자로 받아 테스트 가능하게 CSS를 주입한다."""

    st_module.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
