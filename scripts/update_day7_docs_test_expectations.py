"""Day 7 문서 테스트의 전체 회귀 테스트 기대값을 1255로 갱신한다."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = PROJECT_ROOT / "tests" / "test_create_day7_docs.py"

REPLACEMENTS = {
    'assert "1244 passed" in report':
        'assert "1255 passed" in report',
    'assert "Full Regression Tests: 1244 passed" in section':
        'assert "Full Regression Tests: 1255 passed" in section',
    'assert result == "Full Regression Tests : 1244 passed"':
        'assert result == "Full Regression Tests : 1255 passed"',
}


def main() -> None:
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Test file does not exist: {TEST_PATH}")

    original = TEST_PATH.read_text(encoding="utf-8")
    updated = original

    for old_text, new_text in REPLACEMENTS.items():
        match_count = updated.count(old_text)
        if match_count != 1:
            raise ValueError(
                "Expected exactly one target assertion. "
                f"Target: {old_text!r}, found: {match_count}."
            )
        updated = updated.replace(old_text, new_text, 1)

    TEST_PATH.write_text(
        updated,
        encoding="utf-8",
        newline="\n",
    )

    print("[PASS] Day 7 docs test expectations updated to 1255")
    print(f"[FILE] {TEST_PATH}")


if __name__ == "__main__":
    main()
