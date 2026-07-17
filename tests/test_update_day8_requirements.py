from __future__ import annotations

from scripts.update_day8_requirements import normalize_package_name


def test_normalize_package_name_handles_versions_and_hyphens() -> None:
    assert normalize_package_name("streamlit>=1.40") == "streamlit"
    assert normalize_package_name("python_multipart==0.0.32") == "python-multipart"
    assert normalize_package_name("Pillow") == "pillow"
    assert normalize_package_name("# comment") == ""
