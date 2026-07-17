from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "inspect_day13_detection_api_stage1.py"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location(
        "inspect_day13_detection_api_stage1",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_mapping_normalizes_background_only() -> None:
    script = load_script_module()

    result = script.canonical_expected_class_mapping(
        {
            0: "background",
            1: "crazing",
            2: "inclusion",
            3: "patches",
            4: "pitted_surface",
            5: "rolled_in_scale",
            6: "scratches",
        }
    )

    assert result == {
        "BACKGROUND": 0,
        "crazing": 1,
        "inclusion": 2,
        "patches": 3,
        "pitted_surface": 4,
        "rolled_in_scale": 5,
        "scratches": 6,
    }


def test_canonical_mapping_preserves_defect_names() -> None:
    script = load_script_module()

    result = script.canonical_expected_class_mapping(
        {
            0: "any_background_alias",
            1: "rolled_in_scale",
        }
    )

    assert result["BACKGROUND"] == 0
    assert result["rolled_in_scale"] == 1


def test_canonical_mapping_rejects_non_contiguous_indexes() -> None:
    script = load_script_module()

    with pytest.raises(
        ValueError,
        match="contiguous",
    ):
        script.canonical_expected_class_mapping(
            {
                0: "background",
                2: "crazing",
            }
        )
