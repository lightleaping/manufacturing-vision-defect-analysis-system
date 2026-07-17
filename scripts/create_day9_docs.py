"""Day 9 실제 Artifact를 기반으로 보고서와 README 섹션을 생성한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


README_START = "<!-- DAY9_OBJECT_DETECTION_DATASET_START -->"
README_END = "<!-- DAY9_OBJECT_DETECTION_DATASET_END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 9 Object Detection Dataset 보고서·README 생성",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="프로젝트 루트",
    )
    parser.add_argument(
        "--regression-test-count",
        type=int,
        required=True,
        help="최종 전체 회귀 테스트 passed 수",
    )
    parser.add_argument(
        "--warning-count",
        type=int,
        required=True,
        help="최종 전체 회귀 테스트 warning 수",
    )
    parser.add_argument(
        "--runtime-seconds",
        type=float,
        default=None,
        help="최종 전체 회귀 테스트 Runtime(초), 선택",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"필수 Artifact가 없습니다: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON Artifact를 읽을 수 없습니다: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 최상위 구조가 Object가 아닙니다: {path}")
    return payload


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 구조가 Object가 아닙니다.")
    return value


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"필수 수치 필드가 없습니다: {key}")
    return int(value)


def _number_text(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{float(value):.{digits}f}"


def _mapping_rows(mapping: Mapping[str, Any]) -> str:
    return "\n".join(
        f"| `{key}` | {value} |" for key, value in mapping.items()
    )


def _split_rows(statistics: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for split_name in ("train", "validation", "test"):
        split_stats = _require_mapping(
            statistics.get(split_name),
            name=f"split.statistics.{split_name}",
        )
        rows.append(
            "| "
            + split_name.capitalize()
            + " | "
            + str(_integer(split_stats, "image_count"))
            + " | "
            + str(_integer(split_stats, "box_count"))
            + " | "
            + str(_integer(split_stats, "duplicate_image_hash_group_count"))
            + " |"
        )
    return "\n".join(rows)


def _warning_rows(issue_counts: Mapping[str, Any]) -> str:
    if not issue_counts:
        return "| 없음 | 0 |"
    return "\n".join(
        f"| `{code}` | {count} |"
        for code, count in sorted(issue_counts.items())
    )


def _validate_split_payload(split: Mapping[str, Any]) -> dict[str, object]:
    """저장된 Split JSON에서 경로 중복과 Hash 누수를 다시 계산한다."""
    raw_splits = _require_mapping(split.get("splits"), name="split.splits")
    split_names = ("train", "validation", "test")
    path_sets: dict[str, set[str]] = {}
    hash_to_splits: dict[str, set[str]] = {}
    total_records = 0

    for split_name in split_names:
        items = raw_splits.get(split_name)
        if not isinstance(items, list):
            raise ValueError(f"split.splits.{split_name} 구조가 List가 아닙니다.")
        total_records += len(items)
        paths: set[str] = set()
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(f"{split_name} Split Record가 Object가 아닙니다.")
            image_path = str(item.get("image_path", ""))
            digest = str(item.get("image_sha256", ""))
            if not image_path or not digest:
                raise ValueError(f"{split_name} Split Record에 경로 또는 Hash가 없습니다.")
            paths.add(image_path)
            hash_to_splits.setdefault(digest, set()).add(split_name)
        if len(paths) != len(items):
            raise ValueError(f"{split_name} Split 내부에 중복 image_path가 있습니다.")
        path_sets[split_name] = paths

    overlap_count = 0
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1:]:
            overlap_count += len(path_sets[left] & path_sets[right])

    cross_hash_count = sum(
        1 for split_set in hash_to_splits.values() if len(split_set) > 1
    )
    expected_records = sum(
        int(_require_mapping(
            _require_mapping(split.get("statistics"), name="split.statistics").get(name),
            name=f"split.statistics.{name}",
        ).get("image_count", 0))
        for name in split_names
    )
    result = {
        "split_overlap_count": overlap_count,
        "cross_split_duplicate_hash_count": cross_hash_count,
        "total_manifest_records": total_records,
        "all_records_preserved": total_records == expected_records,
    }
    result["is_valid"] = (
        overlap_count == 0
        and cross_hash_count == 0
        and result["all_records_preserved"]
    )
    return result


def _validate_artifacts(
    analysis: Mapping[str, Any],
    split: Mapping[str, Any],
    visual: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    summary = _require_mapping(analysis.get("summary"), name="analysis.summary")
    statistics = _require_mapping(split.get("statistics"), name="split.statistics")
    validation = _validate_split_payload(split)

    # Dataset을 사용할 수 없는 상태에서 문서가 완료로 생성되는 것을 막는다.
    if _integer(summary, "error_issue_count") != 0:
        raise ValueError("Dataset 분석에 Error가 남아 있어 문서를 생성할 수 없습니다.")
    if _integer(summary, "valid_record_count") <= 0:
        raise ValueError("유효 Detection Record가 없습니다.")
    if not bool(validation.get("is_valid")):
        raise ValueError("Split Manifest 검증이 통과하지 않았습니다.")
    if str(visual.get("status", "")).upper() != "PASS":
        raise ValueError("Figure 육안 검증이 PASS가 아닙니다.")
    if not bool(visual.get("all_manual_checks_passed")):
        raise ValueError("Figure 수동 확인 항목이 모두 통과하지 않았습니다.")

    return summary, statistics, validation


def build_report(
    *,
    analysis: Mapping[str, Any],
    split: Mapping[str, Any],
    visual: Mapping[str, Any],
    regression_test_count: int,
    warning_count: int,
    runtime_seconds: float | None,
) -> str:
    summary, statistics, validation = _validate_artifacts(
        analysis,
        split,
        visual,
    )
    class_image_counts = _require_mapping(
        summary.get("class_image_counts"),
        name="summary.class_image_counts",
    )
    class_box_counts = _require_mapping(
        summary.get("class_box_counts"),
        name="summary.class_box_counts",
    )
    issue_counts = _require_mapping(
        summary.get("issue_counts_by_code", {}),
        name="summary.issue_counts_by_code",
    )
    boxes_per_image = _require_mapping(
        summary.get("boxes_per_image"),
        name="summary.boxes_per_image",
    )
    box_width = _require_mapping(summary.get("box_width"), name="summary.box_width")
    box_height = _require_mapping(summary.get("box_height"), name="summary.box_height")
    box_area = _require_mapping(
        summary.get("box_area_ratio"),
        name="summary.box_area_ratio",
    )
    box_aspect = _require_mapping(
        summary.get("box_aspect_ratio"),
        name="summary.box_aspect_ratio",
    )
    coordinates = _require_mapping(
        summary.get("coordinate_statistics"),
        name="summary.coordinate_statistics",
    )

    class_rows = "\n".join(
        f"| `{class_name}` | {class_image_counts.get(class_name, 0)} | "
        f"{class_box_counts.get(class_name, 0)} |"
        for class_name in class_image_counts
    )
    runtime_line = (
        f"- 전체 회귀 테스트 Runtime: `{runtime_seconds:.2f}초`\n"
        if runtime_seconds is not None
        else ""
    )

    return f"""# Day 9 — Object Detection Dataset Analysis

## 1. 목적

기존 Casting 분류 데이터는 이미지 전체의 `NORMAL` 또는 `DEFECT` Label만 제공하므로 결함의 종류와 위치를 학습할 수 없다. Day 9에서는 기존 분류 V1을 유지하면서, 신규 객체 탐지 데이터의 이미지와 Pascal VOC Bounding Box Annotation을 신뢰 가능한 학습 입력으로 검증했다.

## 2. Dataset 출처와 이용 조건

- Dataset: **NEU Surface Defect Database — NEU-DET**
- 원본 출처: Northeastern University 연구 데이터 페이지
- 다운로드 경로: Kaggle 미러 `kaustubhdikshit/neu-surface-defect-database`
- 압축 파일: `NEU-DET.zip`
- 압축 내부 README·LICENSE·CITATION 파일: 확인되지 않음
- 정책: 원본 데이터 파일은 저장소에 재배포하지 않고 출처와 이용 조건의 불명확성을 문서화한다.

## 3. Classification과 Detection의 분리

```text
Classification
입력 이미지 → NORMAL 또는 DEFECT

Object Detection
입력 이미지 → 결함 Class + Bounding Box + Confidence
```

기존 `data/raw/casting_product_images`와 신규 `data/raw/neu_det`를 분리하고, Classification Class Mapping과 Detection Class Mapping도 별도로 유지했다.

## 4. 실제 원본 구조

```text
data/raw/neu_det/NEU-DET/
├── train/
│   ├── images/<class>/*.jpg
│   └── annotations/*.xml
└── validation/
    ├── images/<class>/*.jpg
    └── annotations/*.xml
```

- 전체 이미지: **{_integer(summary, 'total_image_files'):,}장**
- 전체 XML Annotation: **{_integer(summary, 'total_annotation_files'):,}개**
- 유효 Record: **{_integer(summary, 'valid_record_count'):,}개**
- 유효 Bounding Box: **{_integer(summary, 'total_valid_bounding_boxes'):,}개**
- Class 수: **{_integer(summary, 'class_count')}개**
- 이미지 Mode: `{summary.get('image_mode_counts')}`

## 5. Class 분포

| Class | 이미지 수 | Bounding Box 수 |
|---|---:|---:|
{class_rows}

Class별 이미지 수와 Class별 Box 수는 서로 다른 지표로 분리해 기록했다.

## 6. 원본 데이터 품질 문제와 보정 정책

실제 전체 분석에서 다음 문제가 발견됐다.

1. `crazing_240.jpg`는 원본 `train/images`에 있지만 대응 XML은 `validation/annotations`에 있었다.
2. `patches_101.jpg`와 `patches_105.jpg`는 SHA-256이 같은 동일 이미지지만 Bounding Box 좌표가 조금 달랐다.
3. XML 내부 `<filename>`과 실제 Pair 파일명이 다른 Metadata 경고가 존재했다.
4. 동일 Class·동일 좌표의 중복 Box가 일부 존재했다.

원본 파일은 이동·삭제·수정하지 않았다. 전역에서 동일 stem 이미지와 XML이 각각 하나뿐인 경우에만 Manifest 수준에서 Pair를 연결했고, 동일 이미지 Hash 그룹은 삭제하지 않고 항상 하나의 최종 Split 안에 유지해 데이터 누수를 막았다.

- 원본 누락 Annotation: `{summary.get('raw_missing_annotation_count', 0)}`
- 원본 누락 이미지: `{summary.get('raw_missing_image_count', 0)}`
- 보정 후 누락 Annotation: `{_integer(summary, 'missing_annotation_count')}`
- 보정 후 누락 이미지: `{_integer(summary, 'missing_image_count')}`
- 교차 Partition Pair 보정: `{_integer(summary, 'reconciled_cross_partition_pair_count')}`
- 중복 이미지 Hash 그룹: `{_integer(summary, 'duplicate_image_hash_group_count')}`
- 손상 이미지: `{_integer(summary, 'corrupted_image_count')}`
- 잘못된 Annotation: `{_integer(summary, 'invalid_annotation_count')}`
- 잘못된 Bounding Box: `{_integer(summary, 'invalid_box_count')}`
- 최종 Error: `{_integer(summary, 'error_issue_count')}`

| 품질 이슈 | 개수 |
|---|---:|
{_warning_rows(issue_counts)}

## 7. Bounding Box 통계

| 지표 | 평균 | 중앙값 | 최소 | 최대 |
|---|---:|---:|---:|---:|
| 이미지당 Box 수 | {_number_text(boxes_per_image.get('mean'))} | {_number_text(boxes_per_image.get('median'))} | {_number_text(boxes_per_image.get('min'))} | {_number_text(boxes_per_image.get('max'))} |
| Box Width | {_number_text(box_width.get('mean'))} | {_number_text(box_width.get('median'))} | {_number_text(box_width.get('min'))} | {_number_text(box_width.get('max'))} |
| Box Height | {_number_text(box_height.get('mean'))} | {_number_text(box_height.get('median'))} | {_number_text(box_height.get('min'))} | {_number_text(box_height.get('max'))} |
| Box Area Ratio | {_number_text(box_area.get('mean'))} | {_number_text(box_area.get('median'))} | {_number_text(box_area.get('min'))} | {_number_text(box_area.get('max'))} |
| Box Aspect Ratio | {_number_text(box_aspect.get('mean'))} | {_number_text(box_aspect.get('median'))} | {_number_text(box_aspect.get('min'))} | {_number_text(box_aspect.get('max'))} |

## 8. 좌표 정책

- 추론된 원본 좌표 정책: `{coordinates.get('inferred_source_coordinate_policy')}`
- 최소 좌표 0 등장 수: `{coordinates.get('zero_min_coordinate_count')}`
- 최소 좌표 1 등장 수: `{coordinates.get('one_min_coordinate_count')}`
- `xmax == image_width` Box 수: `{coordinates.get('x_max_at_image_width_count')}`
- `ymax == image_height` Box 수: `{coordinates.get('y_max_at_image_height_count')}`

전체 결과는 Pascal VOC의 **1-based inclusive 좌표**일 가능성이 높다. Day 11 Torchvision Detection Dataset에서는 `(xmin - 1, ymin - 1, xmax, ymax)` 변환을 명시적으로 구현하고 테스트한다.

## 9. Split 정책과 데이터 누수 검사

- Split 정책: `{statistics.get('split_policy')}`
- 중복 Hash 정책: `{statistics.get('duplicate_hash_policy')}`
- Random Seed: `{split.get('random_seed')}`

| Split | 이미지 | Box | 내부 중복 Hash 그룹 |
|---|---:|---:|---:|
{_split_rows(statistics)}

Split 검증:

- 경로 중복: `{validation.get('split_overlap_count')}`
- Split 간 동일 이미지 Hash 누수: `{validation.get('cross_split_duplicate_hash_count')}`
- Manifest Record: `{validation.get('total_manifest_records')}`
- 전체 Record 보존: `{validation.get('all_records_preserved')}`
- 최종 유효성: `{validation.get('is_valid')}`

## 10. 생성 Artifact와 Figure

```text
reports/artifacts/day9_object_detection_dataset_analysis.json
reports/artifacts/day9_object_detection_dataset_split.json
reports/artifacts/day9_detection_visual_validation.json
data/processed/neu_det/splits.json

reports/figures/day9_detection_class_distribution.png
reports/figures/day9_detection_box_statistics.png
reports/figures/day9_detection_annotation_overview.png
```

Figure는 Pillow Decode 검사와 수동 육안 검증을 모두 통과했다.

## 11. 테스트

- Day 9 Dataset·Parser·Split·Visualization·실행 Script 테스트 통과
- 전체 회귀 테스트: **{regression_test_count:,} passed**
- 전체 회귀 테스트 경고: **{warning_count} warning**
{runtime_line}- 기존 Starlette/httpx 관련 Warning은 기능 실패와 분리해 기술부채로 유지한다.

## 12. 현재 범위와 다음 일정

Day 9에서는 Dataset과 Annotation Pipeline을 검증했다. 객체 탐지 모델 학습, mAP 평가, Detection API·Dashboard는 아직 완료되지 않았다.

```text
Day 10 — OpenCV Image Analysis Pipeline
Day 11 — Detection Dataset and Model Implementation
Day 12 — Detection Training, Evaluation and Failure Analysis
Day 13 — Detection FastAPI and Streamlit Integration
Day 14 — Final Integration, README, Portfolio and Interview
```
"""


def build_readme_section(
    *,
    analysis: Mapping[str, Any],
    split: Mapping[str, Any],
    regression_test_count: int,
    warning_count: int,
) -> str:
    summary = _require_mapping(analysis.get("summary"), name="analysis.summary")
    statistics = _require_mapping(split.get("statistics"), name="split.statistics")
    validation = _validate_split_payload(split)
    return f"""{README_START}
## Day 9 — Object Detection Dataset Analysis

기존 Casting 분류 데이터는 이미지 전체의 정상·불량 Label만 제공하므로, 결함 종류와 위치를 학습할 수 있는 **NEU-DET 객체 탐지 데이터셋**을 별도 `src/detection` 계층으로 추가했습니다.

```text
Classification: 이미지 → NORMAL / DEFECT
Detection:      이미지 → 결함 Class + Bounding Box + Confidence
```

### 실제 분석 결과

| 항목 | 결과 |
|---|---:|
| 이미지 | {_integer(summary, 'total_image_files'):,} |
| XML Annotation | {_integer(summary, 'total_annotation_files'):,} |
| 유효 Record | {_integer(summary, 'valid_record_count'):,} |
| Bounding Box | {_integer(summary, 'total_valid_bounding_boxes'):,} |
| Class | {_integer(summary, 'class_count')} |
| 손상 이미지 | {_integer(summary, 'corrupted_image_count')} |
| 잘못된 Box | {_integer(summary, 'invalid_box_count')} |
| 최종 Error | {_integer(summary, 'error_issue_count')} |

원본에서 `crazing_240` 이미지와 XML이 서로 다른 Partition에 배치된 문제와 동일 Hash 이미지 1개 그룹을 발견했습니다. 원본은 수정하지 않고 Manifest 수준에서 유일 Pair를 연결했으며, 동일 Hash 그룹을 하나의 최종 Split 안에 유지해 평가 누수를 방지했습니다.

| Split | 이미지 | Box | 중복 Hash 그룹 |
|---|---:|---:|---:|
{_split_rows(statistics)}

- Split 경로 중복: `{validation.get('split_overlap_count')}`
- Split 간 동일 Hash 누수: `{validation.get('cross_split_duplicate_hash_count')}`
- 전체 Record 보존: `{validation.get('all_records_preserved')}`
- 좌표 정책: `{_require_mapping(summary.get('coordinate_statistics'), name='coordinate_statistics').get('inferred_source_coordinate_policy')}`
- 전체 회귀 테스트: `{regression_test_count:,} passed, {warning_count} warning`

자세한 결과는 `reports/day9_object_detection_dataset_analysis_summary.md`와 Day 9 Artifact·Figure에서 확인할 수 있습니다.
{README_END}"""


def replace_readme_section(readme_text: str, section: str) -> str:
    """기존 Marker가 있으면 교체하고 없으면 README 끝에 추가한다."""
    start_index = readme_text.find(README_START)
    end_index = readme_text.find(README_END)

    if start_index == -1 and end_index == -1:
        return readme_text.rstrip() + "\n\n" + section + "\n"
    if start_index == -1 or end_index == -1 or end_index < start_index:
        raise ValueError("README Day 9 Marker가 불완전하거나 순서가 잘못됐습니다.")

    end_index += len(README_END)
    return (
        readme_text[:start_index].rstrip()
        + "\n\n"
        + section
        + "\n\n"
        + readme_text[end_index:].lstrip()
    ).rstrip() + "\n"


def create_day9_docs(
    *,
    project_root: Path,
    regression_test_count: int,
    warning_count: int,
    runtime_seconds: float | None,
) -> tuple[Path, Path]:
    if regression_test_count <= 0:
        raise ValueError("회귀 테스트 passed 수는 1 이상이어야 합니다.")
    if warning_count < 0:
        raise ValueError("warning 수는 음수일 수 없습니다.")

    project_root = project_root.resolve()
    artifact_dir = project_root / "reports" / "artifacts"
    analysis = _load_json(
        artifact_dir / "day9_object_detection_dataset_analysis.json"
    )
    split = _load_json(
        artifact_dir / "day9_object_detection_dataset_split.json"
    )
    visual = _load_json(
        artifact_dir / "day9_detection_visual_validation.json"
    )
    _validate_artifacts(analysis, split, visual)

    report_text = build_report(
        analysis=analysis,
        split=split,
        visual=visual,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
        runtime_seconds=runtime_seconds,
    )
    report_path = (
        project_root
        / "reports"
        / "day9_object_detection_dataset_analysis_summary.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text.rstrip() + "\n", encoding="utf-8")

    readme_path = project_root / "README.md"
    if not readme_path.is_file():
        raise FileNotFoundError(f"README.md가 없습니다: {readme_path}")
    readme_text = readme_path.read_text(encoding="utf-8")
    section = build_readme_section(
        analysis=analysis,
        split=split,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
    )
    updated_readme = replace_readme_section(readme_text, section)
    readme_path.write_text(updated_readme, encoding="utf-8")
    return report_path, readme_path


def main() -> int:
    args = parse_args()
    try:
        report_path, readme_path = create_day9_docs(
            project_root=args.project_root,
            regression_test_count=args.regression_test_count,
            warning_count=args.warning_count,
            runtime_seconds=args.runtime_seconds,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    print("[PASS] Day 9 report created")
    print("[PASS] README Day 9 section added or updated")
    print(f"[REPORT] {report_path}")
    print(f"[README] {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
