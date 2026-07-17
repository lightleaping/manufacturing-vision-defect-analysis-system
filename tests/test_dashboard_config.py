from __future__ import annotations

import pytest

from src.dashboard.config import DashboardSettings, load_dashboard_settings


def test_default_settings_use_local_fastapi_url() -> None:
    settings = load_dashboard_settings({})

    assert settings.api_base_url == "http://127.0.0.1:8000"
    assert settings.health_endpoint == "/api/v1/health"
    assert settings.prediction_endpoint == "/api/v1/predictions"
    assert settings.health_check_enabled is True


def test_environment_overrides_are_parsed() -> None:
    settings = load_dashboard_settings(
        {
            "MVDA_API_BASE_URL": "http://localhost:9000/",
            "MVDA_API_CONNECT_TIMEOUT_SECONDS": "3.5",
            "MVDA_API_READ_TIMEOUT_SECONDS": "40",
            "MVDA_DASHBOARD_HEALTH_CHECK_ENABLED": "false",
        }
    )

    assert settings.api_base_url == "http://localhost:9000"
    assert settings.connect_timeout_seconds == 3.5
    assert settings.read_timeout_seconds == 40.0
    assert settings.health_check_enabled is False


def test_invalid_base_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="http"):
        DashboardSettings(api_base_url="127.0.0.1:8000")


def test_invalid_boolean_environment_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="true/false"):
        load_dashboard_settings(
            {"MVDA_DASHBOARD_HEALTH_CHECK_ENABLED": "sometimes"}
        )
