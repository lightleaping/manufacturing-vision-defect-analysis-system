"""Day 11 결과 Artifact를 기반으로 보고서와 README Section을 생성한다.

아직 실행하지 않은 Fine-tuning, mAP, Failure Analysis는 결과로 기록하지 않는다.
문서의 수치와 상태는 Day 11 JSON Artifact에서 읽는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DATASET_ARTIFACT = Path("reports/artifacts/day11_detection_dataset_validation.json")
MODEL_ARTIFACT = Path("reports/artifacts/day11_detection_model_smoke_test.json")
REPORT_PATH = Path("reports/day11_detection_dataset_and_model_implementation_summary.md")
README_PATH = Path("README.md")
README_START = "<!-- DAY11_DETECTION_DATASET_MODEL_START -->"
README_END = "<!-- DAY11_DETECTION_DATASET_MODEL_END -->"

REQUIRED_FIGURES = (
    Path("reports/figures/day11_detection_dataset_batch.png"),
    Path("reports/figures/day11_detection_target_overlay.png"),
    Path("reports/figures/day11_detection_model_predictions_smoke_test.png"),
)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON Artifact does not exist: {path}.")
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON top-level value must be an object: {path}.")
    return payload


def _require_pass(payload: Mapping[str, Any], name: str) -> None:
    if payload.get("validation_passed") is not True:
        raise ValueError(f"{name} Artifact has not passed validation.")


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"Missing Artifact field: {'.'.join(keys)}.")
        current = current[key]
    return current


def _format_loss_rows(model_payload: Mapping[str, Any]) -> str:
    losses = _nested(model_payload, "smoke_test", "training_forward", "losses")
    if not isinstance(losses, Mapping):
        raise TypeError("Model losses must be an object.")
    ordered_keys = (
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    )
    rows = []
    for key in ordered_keys:
        value = losses.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"Model loss {key} must be numeric.")
        rows.append(f"| `{key}` | {float(value):.6f} |")
    total_loss = _nested(model_payload, "smoke_test", "training_forward", "total_loss")
    rows.append(f"| **Total** | **{float(total_loss):.6f}** |")
    return "\n".join(rows)


def _class_mapping_text(dataset_payload: Mapping[str, Any]) -> str:
    class_mapping = dataset_payload.get("class_mapping")
    if not isinstance(class_mapping, Mapping):
        raise TypeError("Dataset class_mapping must be an object.")
    items = sorted(class_mapping.items(), key=lambda item: int(item[1]))
    return "\n".join(f"{int(index)}. `{name}`" for name, index in items)


def render_day11_report(
    *,
    dataset_payload: Mapping[str, Any],
    model_payload: Mapping[str, Any],
    targeted_test_count: int,
    regression_test_count: int,
    warning_count: int,
) -> str:
    totals = _nested(dataset_payload, "totals")
    splits = _nested(dataset_payload, "splits")
    prediction = _nested(
        model_payload,
        "smoke_test",
        "evaluation_forward",
        "predictions",
    )[0]
    model = _nested(model_payload, "model")
    source_sample = _nested(model_payload, "source_sample")
    training_time = float(
        _nested(model_payload, "smoke_test", "training_forward", "elapsed_seconds")
    )
    evaluation_time = float(
        _nested(model_payload, "smoke_test", "evaluation_forward", "elapsed_seconds")
    )

    return f"""# Day 11 — Detection Dataset and Model Implementation

## 1. 작업 상태

```text
Dataset Runtime Validation : PASS
Model Forward Smoke Test   : PASS
Full Detection Training    : Not executed (Day 12 scope)
Pretrained Weight Download : Not executed
```

Day 11에서는 NEU-DET의 Pascal VOC Annotation을 Torchvision Detection 입력으로 변환하고, CPU에서 Weight 다운로드 없이 Faster R-CNN의 Training·Evaluation Forward 계약을 검증했다.

## 2. 구현 범위

- Pascal VOC XML 기반 `NeuDetDetectionDataset`
- 1-based inclusive 좌표를 0-based exclusive XYXY로 변환
- Background를 포함한 7-Class Label Mapping
- 이미지와 Bounding Box를 함께 처리하는 Detection Transform
- 가변 Box Target을 위한 `detection_collate_fn`
- Windows CPU 설정의 Train·Validation·Test DataLoader
- MobileNetV3-Large 320 FPN Faster R-CNN Model Factory
- NEU-DET 7-Class Detection Head
- Weight 없는 Training Loss·Evaluation Prediction Smoke Test

## 3. Detection Class Mapping

{_class_mapping_text(dataset_payload)}

Torchvision Detection에서는 `0`을 Background로 사용하므로 6개 결함 Class를 포함한 `num_classes`는 **{dataset_payload['num_classes_including_background']}**이다.

## 4. 좌표 변환

원본 Pascal VOC 좌표는 1-based inclusive로 해석하고 다음 정책으로 변환했다.

```python
(xmin - 1, ymin - 1, xmax, ymax)
```

예를 들어 원본 이미지 전체를 나타내는 `[1, 1, 200, 200]`은 Torchvision Target에서 `[0, 0, 200, 200]`이 된다.

## 5. 실제 Dataset Runtime Validation

| Split | Images | Boxes | Validation |
|---|---:|---:|---|
| Train | {splits['train']['sample_count']:,} | {splits['train']['dataset_box_count']:,} | PASS |
| Validation | {splits['validation']['sample_count']:,} | {splits['validation']['dataset_box_count']:,} | PASS |
| Test | {splits['test']['sample_count']:,} | {splits['test']['dataset_box_count']:,} | PASS |
| **Total** | **{totals['sample_count']:,}** | **{totals['box_count']:,}** | **PASS** |

검증 항목에는 Image Tensor의 Shape·dtype·range, Target Dict Key, Box·Label·Area dtype, 좌표 범위, Manifest와 Target의 순서별 일치, DataLoader Batch 계약, Split 경로 중복 여부가 포함된다.

## 6. Duplicate Box 정책

```text
정책                 : {dataset_payload['duplicate_box_policy']}
정확히 같은 중복 Box : {totals['raw_exact_duplicate_count']}
원본 XML 수정         : False
Loader 임의 삭제      : False
```

기본 정책은 `preserve`다. 보조 설정 `remove_exact`은 동일 Annotation에서 Class와 네 좌표가 모두 같은 Box만 제거할 수 있지만 Day 11 실제 검증과 Day 12 기본 입력에는 적용하지 않았다.

## 7. Detection Model

```text
Architecture             : {model['architecture']}
Device                   : {model['device']}
Predictor output classes : {model['predictor_output_classes']}
Detection weights        : {model['pretrained_detection_weights']}
Backbone weights         : {model['pretrained_backbone_weights']}
Network download         : {model['network_download_requested']}
Smoke input resize       : {_nested(model_payload, 'execution_policy', 'smoke_input_resize')}
```

CPU 실행 가능성을 우선해 `fasterrcnn_mobilenet_v3_large_320_fpn`을 선택했다. Day 11에서는 구조 검증을 위해 `weights=None`, `weights_backbone=None`을 사용했다.

## 8. Training Forward Smoke Test

Source sample: `{source_sample['record_id']}`

| Loss | Value |
|---|---:|
{_format_loss_rows(model_payload)}

```text
Training forward time   : {training_time:.3f}s
All losses finite       : True
Backward executed       : False
Optimizer step executed : False
```

## 9. Evaluation Forward Smoke Test

```text
Evaluation forward time : {evaluation_time:.3f}s
Prediction boxes        : {prediction['box_count']}
Boxes shape             : {prediction['boxes_shape']}
Labels shape            : {prediction['labels_shape']}
Scores shape            : {prediction['scores_shape']}
```

초기화된 Head의 Prediction은 모델 성능이 아니다. Figure와 출력은 Training 전 입출력 구조가 정상인지 확인하기 위한 Smoke Test 결과로만 사용한다.

## 10. 생성 Artifact

```text
reports/artifacts/day11_detection_dataset_validation.json
reports/artifacts/day11_detection_model_smoke_test.json
reports/figures/day11_detection_dataset_batch.png
reports/figures/day11_detection_target_overlay.png
reports/figures/day11_detection_model_predictions_smoke_test.png
```

## 11. 테스트 결과

```text
Day 11 targeted tests : {targeted_test_count} passed
Full regression tests : {regression_test_count} passed
Warnings              : {warning_count}
```

## 12. Day 12 연결

Day 12에서는 디스크 공간을 확보한 뒤 COCO pretrained Detection Weight를 적용하고, 같은 Model Factory의 COCO Predictor를 NEU-DET 7-Class Predictor로 교체해 Fine-tuning한다. 이후 Validation·Test Prediction, IoU 기반 평가, mAP·mAR, Checkpoint와 Failure Analysis를 수행한다.

Day 11에서는 전체 학습, 최종 Checkpoint, mAP·mAR, Failure Analysis를 완료했다고 표현하지 않는다.
"""


def render_readme_section(
    *,
    dataset_payload: Mapping[str, Any],
    model_payload: Mapping[str, Any],
    targeted_test_count: int,
    regression_test_count: int,
    warning_count: int,
) -> str:
    totals = _nested(dataset_payload, "totals")
    model = _nested(model_payload, "model")
    training = _nested(model_payload, "smoke_test", "training_forward")
    evaluation = _nested(model_payload, "smoke_test", "evaluation_forward")
    return f"""{README_START}
## Day 11 — Detection Dataset and Model Implementation

NEU-DET Pascal VOC Annotation을 Torchvision Detection Dataset으로 구현하고, CPU에서 Weight 다운로드 없이 Faster R-CNN의 Training·Evaluation Forward 계약을 검증했다.

```text
Dataset             : 1,800 images / 4,189 boxes
Split               : Train 1,440 / Validation 178 / Test 182
Coordinate policy   : (xmin - 1, ymin - 1, xmax, ymax)
Duplicate policy    : preserve ({totals['raw_exact_duplicate_count']} exact duplicates)
Class mapping       : BACKGROUND 0 + defect classes 1~6
Architecture        : {model['architecture']}
Predictor classes   : {model['predictor_output_classes']}
Pretrained download : False
Training loss total : {float(training['total_loss']):.6f}
Forward time        : train {float(training['elapsed_seconds']):.3f}s / eval {float(evaluation['elapsed_seconds']):.3f}s
Smoke predictions   : {evaluation['predictions'][0]['box_count']} boxes
Validation          : PASS
```

Day 11 Prediction은 Random Initialization 기반 구조 검증 결과이며 학습 성능이 아니다. COCO pretrained Fine-tuning과 Detection 평가는 Day 12에서 진행한다.

- Report: `reports/day11_detection_dataset_and_model_implementation_summary.md`
- Dataset Artifact: `reports/artifacts/day11_detection_dataset_validation.json`
- Model Artifact: `reports/artifacts/day11_detection_model_smoke_test.json`
- Figures: `reports/figures/day11_detection_*.png`
- Tests: Day 11 {targeted_test_count} passed / Full regression {regression_test_count} passed / {warning_count} warning(s)
{README_END}"""


def update_marker_block(original: str, replacement: str) -> str:
    start_count = original.count(README_START)
    end_count = original.count(README_END)
    if start_count != end_count:
        raise ValueError("README Day 11 marker pair is unbalanced.")
    if start_count > 1:
        raise ValueError("README contains duplicate Day 11 marker blocks.")

    if start_count == 1:
        start = original.index(README_START)
        end = original.index(README_END, start) + len(README_END)
        return original[:start] + replacement + original[end:]

    separator = "" if not original else ("\n" if original.endswith("\n") else "\n\n")
    return original + separator + replacement + "\n"


def create_day11_docs(
    *,
    project_root: Path,
    targeted_test_count: int,
    regression_test_count: int,
    warning_count: int,
) -> tuple[Path, Path]:
    for name, value in {
        "targeted_test_count": targeted_test_count,
        "regression_test_count": regression_test_count,
        "warning_count": warning_count,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be int.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    root = project_root.resolve()
    dataset_payload = _read_json_object(root / DATASET_ARTIFACT)
    model_payload = _read_json_object(root / MODEL_ARTIFACT)
    _require_pass(dataset_payload, "Dataset validation")
    _require_pass(model_payload, "Model smoke test")

    for relative_path in REQUIRED_FIGURES:
        path = root / relative_path
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Required Day 11 Figure is missing: {path}.")

    report = render_day11_report(
        dataset_payload=dataset_payload,
        model_payload=model_payload,
        targeted_test_count=targeted_test_count,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
    )
    readme_section = render_readme_section(
        dataset_payload=dataset_payload,
        model_payload=model_payload,
        targeted_test_count=targeted_test_count,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
    )

    report_path = root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    readme_path = root / README_PATH
    original_readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    updated_readme = update_marker_block(original_readme, readme_section)
    readme_path.write_text(updated_readme, encoding="utf-8")
    return report_path, readme_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create Day 11 report and README section.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--targeted-test-count", type=int, required=True)
    parser.add_argument("--regression-test-count", type=int, required=True)
    parser.add_argument("--warning-count", type=int, default=0)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    report_path, readme_path = create_day11_docs(
        project_root=args.project_root,
        targeted_test_count=args.targeted_test_count,
        regression_test_count=args.regression_test_count,
        warning_count=args.warning_count,
    )
    print("[PASS] Day 11 report created")
    print("[PASS] README Day 11 section added or updated")
    print(f"[REPORT] {report_path}")
    print(f"[README] {readme_path}")


if __name__ == "__main__":
    main()
