"""Day 7 문서 테스트의 과도하게 넓은 문자열 단언을 안전하게 수정한다."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = PROJECT_ROOT / "tests" / "test_create_day7_docs.py"

OLD_TEXT = '    assert "old" not in second_result\n'
NEW_TEXT = '    assert "\\nold\\n" not in second_result\n'


def main() -> None:
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"Test file does not exist: {TEST_PATH}")

    original = TEST_PATH.read_text(encoding="utf-8")

    match_count = original.count(OLD_TEXT)
    if match_count != 1:
        raise ValueError(
            "Expected exactly one target assertion. "
            f"Received: {match_count}."
        )

    updated = original.replace(OLD_TEXT, NEW_TEXT, 1)
    TEST_PATH.write_text(updated, encoding="utf-8", newline="\n")

    print("[PASS] Day 7 docs test assertion updated")
    print(f"[FILE] {TEST_PATH}")


if __name__ == "__main__":
    main()
