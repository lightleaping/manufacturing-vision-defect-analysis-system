r"""Day 12 Stage 2 Pilot의 Background Mapping 불일치를 안전하게 수정한다.

실행 위치:
    manufacturing-vision-defect-analysis-system 프로젝트 Root

실행 명령:
    python .\patch_day12_stage2_background_mapping.py
"""

from __future__ import annotations

from pathlib import Path
import py_compile
import shutil


TARGET_RELATIVE_PATH = Path(
    "scripts/run_day12_detection_training_pilot.py"
)
BACKUP_RELATIVE_PATH = Path(
    "reports/backups/day12_stage2_background_mapping/"
    "run_day12_detection_training_pilot.py.before_hotfix"
)


def _replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"[STOP] {name} 패턴 개수가 1이 아닙니다: {count}. "
            "파일이 예상 버전과 다르므로 자동 수정하지 않습니다."
        )
    return text.replace(old, new, 1)


def main() -> None:
    project_root = Path.cwd().resolve()
    target_path = project_root / TARGET_RELATIVE_PATH
    backup_path = project_root / BACKUP_RELATIVE_PATH

    if not target_path.is_file():
        raise FileNotFoundError(
            f"[STOP] 수정 대상 파일이 없습니다: {target_path}"
        )

    original_text = target_path.read_text(encoding="utf-8")
    already_fixed = (
        "def _canonical_class_mapping(" in original_text
        and "class_mapping=class_mapping," in original_text
        and '"class_to_index": class_mapping,' in original_text
    )
    if already_fixed:
        print("[SKIP] Background Mapping Hotfix가 이미 적용돼 있습니다.")
        print(f"[FILE] {target_path}")
        return

    patched_text = original_text
    patched_text = _replace_once(
        patched_text,
        "import argparse\nfrom datetime import datetime, timezone\n",
        (
            "import argparse\n"
            "from collections.abc import Mapping\n"
            "from datetime import datetime, timezone\n"
        ),
        name="Mapping import",
    )

    old_validation_block = '''def _validate_positive_int(name: str, value: int) -> None:\n    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:\n        raise ValueError(f"{name} must be a positive int.")\n\n\n'''
    new_validation_block = '''def _validate_positive_int(name: str, value: int) -> None:\n    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:\n        raise ValueError(f"{name} must be a positive int.")\n\n\ndef _canonical_class_mapping(\n    index_to_class: Mapping[int, str],\n) -> dict[str, int]:\n    """Model Label Mapping을 Checkpoint의 고정 계약으로 정규화한다.\n\n    일부 기존 구성은 Background 이름을 ``background``로 반환할 수 있다.\n    Detection Label 0의 의미는 같으므로 저장 경계에서만 ``BACKGROUND``로\n    통일한다. 나머지 결함 Class 이름은 원본 Canonical 이름을 유지한다.\n    """\n    if not isinstance(index_to_class, Mapping):\n        raise TypeError("index_to_class must be a mapping.")\n\n    class_mapping: dict[str, int] = {}\n    for class_index, class_name in sorted(index_to_class.items()):\n        if (\n            not isinstance(class_index, int)\n            or isinstance(class_index, bool)\n            or class_index < 0\n        ):\n            raise ValueError(\n                "Every index_to_class key must be a non-negative int."\n            )\n        if not isinstance(class_name, str) or not class_name:\n            raise ValueError(\n                "Every index_to_class value must be a non-empty str."\n            )\n\n        normalized_name = (\n            "BACKGROUND" if class_index == 0 else class_name\n        )\n        if normalized_name in class_mapping:\n            raise ValueError(\n                f"Duplicate normalized class name: {normalized_name}."\n            )\n        class_mapping[normalized_name] = class_index\n\n    if class_mapping.get("BACKGROUND") != 0:\n        raise ValueError(\n            "index_to_class must contain the background label at index 0."\n        )\n    return class_mapping\n\n\n'''
    patched_text = _replace_once(
        patched_text,
        old_validation_block,
        new_validation_block,
        name="class mapping helper",
    )

    old_model_config_block = '''    model_config = DetectionModelConfig(\n        min_size=training_config.min_size,\n        max_size=training_config.max_size,\n        use_pretrained_weights=True,\n        use_pretrained_backbone=False,\n        progress=True,\n    )\n    _write_json(\n'''
    new_model_config_block = '''    model_config = DetectionModelConfig(\n        min_size=training_config.min_size,\n        max_size=training_config.max_size,\n        use_pretrained_weights=True,\n        use_pretrained_backbone=False,\n        progress=True,\n    )\n    class_mapping = _canonical_class_mapping(\n        model_config.index_to_class\n    )\n    _write_json(\n'''
    patched_text = _replace_once(
        patched_text,
        old_model_config_block,
        new_model_config_block,
        name="class mapping creation",
    )

    old_config_mapping = '''                "class_to_index": {\n                    class_name: class_index\n                    for class_index, class_name in model_config.index_to_class.items()\n                },\n'''
    new_config_mapping = '''                "class_to_index": class_mapping,\n'''
    patched_text = _replace_once(
        patched_text,
        old_config_mapping,
        new_config_mapping,
        name="config artifact mapping",
    )

    old_checkpoint_mapping = '''            class_mapping={\n                class_name: class_index\n                for class_index, class_name in model_config.index_to_class.items()\n            },\n'''
    new_checkpoint_mapping = '''            class_mapping=class_mapping,\n'''
    patched_text = _replace_once(
        patched_text,
        old_checkpoint_mapping,
        new_checkpoint_mapping,
        name="checkpoint mapping",
    )

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(target_path, backup_path)

    target_path.write_text(patched_text, encoding="utf-8")
    try:
        py_compile.compile(str(target_path), doraise=True)
    except Exception:
        shutil.copy2(backup_path, target_path)
        raise

    print("[PASS] Day 12 Stage 2 Background Mapping Hotfix 적용 완료")
    print(f"[FILE] {target_path}")
    print(f"[BACKUP] {backup_path}")
    print("[RULE] Checkpoint class_mapping은 BACKGROUND=0을 사용합니다.")


if __name__ == "__main__":
    main()
