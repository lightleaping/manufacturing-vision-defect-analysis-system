from __future__ import annotations

from src.dashboard.styles import DASHBOARD_CSS, inject_dashboard_styles


class _FakeStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def markdown(self, body: str, *, unsafe_allow_html: bool) -> None:
        self.calls.append((body, unsafe_allow_html))


def test_dashboard_css_is_scoped_and_injected() -> None:
    fake = _FakeStreamlit()

    inject_dashboard_styles(fake)

    assert ".mvda-hero" in DASHBOARD_CSS
    assert ".mvda-card" in DASHBOARD_CSS
    assert fake.calls == [(DASHBOARD_CSS, True)]
