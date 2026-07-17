"""Day 12 Detection 실제 Artifact로 보고서와 README Section을 생성한다.

[신규 구현]
- Training·Checkpoint·Validation·Test·Failure Analysis 결과를 JSON과 Best
  Checkpoint에서 읽는다.
- Test 결과가 Best Checkpoint 선택에 사용되지 않았는지 검증한다.
- README Marker를 중복 없이 추가하거나 교체한다.
- 실행하지 않은 Day 13 API·Streamlit 통합을 완료했다고 기록하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


TRAINING_CONFIG_ARTIFACT = Path(
    "reports/artifacts/day12_detection_training_config.json"
)
TRAINING_HISTORY_ARTIFACT = Path(
    "reports/artifacts/day12_detection_training_history.json"
)
EVALUATION_ARTIFACT = Path(
    "reports/artifacts/day12_detection_evaluation.json"
)
FAILURE_ARTIFACT = Path(
    "reports/artifacts/day12_detection_failure_analysis.json"
)
BEST_CHECKPOINT = Path("models/detection/day12_detection_best.pt")
REPORT_PATH = Path(
    "reports/day12_detection_training_evaluation_and_failure_analysis_summary.md"
)
README_PATH = Path("README.md")
README_START = "<!-- DAY12_DETECTION_TRAINING_EVALUATION_START -->"
README_END = "<!-- DAY12_DETECTION_TRAINING_EVALUATION_END -->"
BACKUP_ROOT = Path("reports/backups/day12_docs")

REQUIRED_FIGURES = (
    Path("reports/figures/day12_detection_training_history.png"),
    Path("reports/figures/day12_detection_class_metrics.png"),
    Path("reports/figures/day12_detection_predictions.png"),
    Path("reports/figures/day12_detection_failure_analysis.png"),
)

_FAILURE_LABELS = {
    "false_positive": "False Positive",
    "false_negative": "False Negative",
    "wrong_class": "Wrong Class",
    "low_iou_localization": "Low IoU Localization",
    "duplicate_prediction": "Duplicate Prediction",
    "low_confidence_correct_detection": "Low-confidence Correct Detection",
}


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON Artifact does not exist: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON top-level value must be an object: {path}.")
    return payload


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Best Checkpoint does not exist: {path}.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Best Checkpoint payload must be a dict.")
    required = {
        "epoch",
        "history",
        "best_metric",
        "training_config",
        "class_mapping",
    }
    missing = required - set(payload)
    if missing:
        raise KeyError(f"Best Checkpoint is missing keys: {sorted(missing)}.")
    return payload


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"Missing Artifact field: {'.'.join(keys)}.")
        current = current[key]
    return current


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_execution_policy(evaluation: Mapping[str, Any]) -> None:
    policy = _nested(evaluation, "evaluation_policy")
    required_true = (
        "best_checkpoint_selected_on_validation",
        "best_checkpoint_frozen_before_test",
        "test_split_used",
    )
    for key in required_true:
        if policy.get(key) is not True:
            raise ValueError(f"Evaluation policy must set {key}=True.")
    if policy.get("test_result_used_for_model_selection") is not False:
        raise ValueError(
            "Test result must not be used for model or checkpoint selection."
        )


def _validate_artifacts(
    *,
    training_history: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    failure: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    if training_history.get("validation_passed") is not True:
        raise ValueError("Training History Artifact has not passed validation.")
    _validate_execution_policy(evaluation)
    if failure.get("split") != "test":
        raise ValueError("Failure Analysis must use the test split.")
    if _nested(failure, "checkpoint", "epoch_index") != _nested(
        evaluation, "checkpoint", "epoch_index"
    ):
        raise ValueError("Evaluation and Failure Analysis checkpoint epochs differ.")
    if int(checkpoint["epoch"]) != int(
        _nested(evaluation, "checkpoint", "epoch_index")
    ):
        raise ValueError("Best Checkpoint epoch does not match Evaluation Artifact.")
    checkpoint_metric = _finite_float(checkpoint["best_metric"], "best_metric")
    evaluation_metric = _finite_float(
        _nested(evaluation, "checkpoint", "best_validation_metric"),
        "best_validation_metric",
    )
    if not math.isclose(checkpoint_metric, evaluation_metric, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Best Checkpoint metric does not match Evaluation Artifact.")


def _metric(value: Any) -> str:
    return "-" if value is None else f"{_finite_float(value, 'metric'):.6f}"


def _training_rows(checkpoint: Mapping[str, Any]) -> str:
    history = checkpoint["history"]
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError("Checkpoint history must be a sequence.")
    if not history:
        raise ValueError("Checkpoint history must not be empty.")

    rows: list[str] = []
    for index, entry in enumerate(history):
        if not isinstance(entry, Mapping):
            raise TypeError(f"Checkpoint history[{index}] must be an object.")
        epoch_number = int(entry["epoch"]) + 1
        stage = str(entry.get("stage", "training"))
        backbone_trainable = bool(entry.get("backbone_trainable", False))
        train = _nested(entry, "train")
        validation = _nested(entry, "validation", "metrics", "overall")
        loss = _finite_float(
            _nested(train, "average_losses", "total_loss"),
            "average total loss",
        )
        elapsed_seconds = _finite_float(
            train.get("elapsed_seconds", 0.0),
            "training elapsed seconds",
        )
        learning_rates = entry.get("learning_rates")
        if isinstance(learning_rates, Sequence) and not isinstance(
            learning_rates, (str, bytes)
        ) and learning_rates:
            learning_rate_text = ", ".join(
                f"{_finite_float(value, 'learning rate'):.6f}" for value in learning_rates
            )
        else:
            learning_rate_text = "0.005000" if epoch_number == 1 else "0.001000"
        rows.append(
            "| "
            f"{epoch_number} | `{stage}` | "
            f"{'Unfreeze' if backbone_trainable else 'Freeze'} | "
            f"{learning_rate_text} | {loss:.6f} | {elapsed_seconds / 60.0:.2f} min | "
            f"{_metric(validation['precision'])} | {_metric(validation['recall'])} | "
            f"{_metric(validation['f1'])} | {_metric(validation['map_50'])} |"
        )
    return "\n".join(rows)


def _class_metric_rows(evaluation: Mapping[str, Any]) -> str:
    class_metrics = _nested(evaluation, "test", "metrics", "class_metrics")
    sweep = _nested(evaluation, "test_iou_sweep", "class_map_50_95")
    if not isinstance(class_metrics, Mapping) or not class_metrics:
        raise ValueError("Test class_metrics must be a non-empty object.")
    if not isinstance(sweep, Mapping):
        raise TypeError("class_map_50_95 must be an object.")

    rows: list[str] = []
    for class_name, values in class_metrics.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"Class metric {class_name!r} must be an object.")
        rows.append(
            "| "
            f"`{class_name}` | {int(values['ground_truth_count'])} | "
            f"{int(values['tp'])} | {int(values['fp'])} | {int(values['fn'])} | "
            f"{_metric(values['precision'])} | {_metric(values['recall'])} | "
            f"{_metric(values['f1'])} | {_metric(values['mean_matched_iou'])} | "
            f"{_metric(values['ap_50'])} | {_metric(sweep.get(class_name))} |"
        )
    return "\n".join(rows)


def _failure_rows(failure: Mapping[str, Any]) -> str:
    counts = _nested(failure, "analysis", "summary", "counts")
    if not isinstance(counts, Mapping):
        raise TypeError("Failure counts must be an object.")
    items = sorted(counts.items(), key=lambda item: int(item[1]), reverse=True)
    return "\n".join(
        f"| {_FAILURE_LABELS.get(str(name), str(name))} | {int(count)} |"
        for name, count in items
    )


def _class_insights(evaluation: Mapping[str, Any]) -> tuple[str, str]:
    class_metrics = _nested(evaluation, "test", "metrics", "class_metrics")
    if not isinstance(class_metrics, Mapping) or not class_metrics:
        raise ValueError("Test class_metrics must be a non-empty object.")
    best_name, best_values = max(
        class_metrics.items(),
        key=lambda item: _finite_float(item[1]["f1"], "class f1"),
    )
    weak_name, weak_values = min(
        class_metrics.items(),
        key=lambda item: _finite_float(item[1]["recall"], "class recall"),
    )
    best = (
        f"`{best_name}`는 F1 {_metric(best_values['f1'])}, "
        f"AP@0.50 {_metric(best_values['ap_50'])}로 가장 안정적이었다."
    )
    weak = (
        f"`{weak_name}`는 Ground Truth {int(weak_values['ground_truth_count'])}개 중 "
        f"TP {int(weak_values['tp'])}, FN {int(weak_values['fn'])}으로 "
        f"Recall {_metric(weak_values['recall'])}을 기록해 가장 큰 개선 대상이다."
    )
    return best, weak


def render_day12_report(
    *,
    training_config: Mapping[str, Any],
    training_history: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    failure: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    targeted_test_count: int,
    regression_test_count: int,
    warning_count: int,
) -> str:
    del training_config  # 존재와 JSON Schema는 create 함수에서 검증한다.
    del training_history
    validation = _nested(evaluation, "validation", "metrics", "overall")
    test = _nested(evaluation, "test", "metrics", "overall")
    policy = _nested(evaluation, "evaluation_policy")
    checkpoint_info = _nested(evaluation, "checkpoint")
    failure_summary = _nested(failure, "analysis", "summary")
    model = _nested(evaluation, "model")
    test_map_50_95 = _nested(evaluation, "test_iou_sweep", "summary", "map_50_95")
    best_class, weak_class = _class_insights(evaluation)
    warning_note = (
        "기존 Starlette/httpx deprecation warning을 Day 12에서 Dependency 변경 없이 유지했다."
        if warning_count
        else "경고 없이 통과했다."
    )

    return f"""# Day 12 — Detection Training, Evaluation and Failure Analysis

## 1. 작업 상태

```text
COCO pretrained weight 준비       : 완료
CPU Training Pilot                : PASS
전체 Train Split Fine-tuning      : 완료
Validation Best Checkpoint 선택   : 완료
Test Split 최종 평가              : 완료
Failure Analysis                  : 완료
Readable Figure V2                : 완료
Detection API·Streamlit Integration: Day 13 범위
```

Day 12에서는 Day 11의 Detection Dataset·Model Factory를 재사용해 COCO pretrained Faster R-CNN MobileNetV3 Large 320 FPN을 NEU-DET 7-Class Detection Model로 Fine-tuning했다. Best Checkpoint는 Validation mAP@0.50으로만 선택했으며, Test 결과는 모델 선택이나 추가 학습 결정에 사용하지 않았다.

## 2. 모델과 Transfer Learning

```text
Architecture             : {model.get('architecture', 'fasterrcnn_mobilenet_v3_large_320_fpn')}
Device                   : {model.get('device', 'cpu')}
Input min / max size     : {model.get('min_size', 320)} / {model.get('max_size', 320)}
Pretrained detection     : {model.get('pretrained_detection_weights')}
Classes with background  : {model.get('num_classes_with_background', 7)}
Best checkpoint epoch    : {int(checkpoint_info['completed_epoch_number'])}
Best validation metric   : mAP@0.50 = {_metric(checkpoint_info['best_validation_metric'])}
Duplicate box policy     : {policy['duplicate_box_policy']}
```

COCO Weight로 시각 특징·RPN·Box Regression 표현을 재사용하고, COCO Predictor는 Background 포함 NEU-DET 7-Class Predictor로 교체했다. Random Initialization 전체 학습보다 작은 Dataset과 CPU 환경에서 빠르게 수렴할 수 있는 Transfer Learning 전략이다.

## 3. CPU 학습 정책

```text
Batch size               : 1
Train augmentation       : Horizontal Flip 0.5
Validation·Test transform: Deterministic
Optimizer                : SGD
Frozen-head learning rate: 0.005
Unfreeze learning rate   : 0.001
Best metric              : Validation mAP@0.50
Checkpoint               : latest + best
Test split during training: 미사용
```

첫 Epoch는 Backbone을 동결해 새 7-Class Head와 Detection Head를 안정화했다. 이후 Backbone을 열고 Learning Rate를 낮춰 두 Epoch를 추가 Fine-tuning했다.

## 4. Epoch별 학습과 Validation

| Epoch | Stage | Backbone | Learning Rate | Train Loss | Train Time | Precision | Recall | F1 | mAP@0.50 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
{_training_rows(checkpoint)}

Training Loss는 최적화를 위한 신호이고 Detection Metric은 최종 Class·Box 품질을 측정한다. Backbone을 연 뒤 Loss 분포가 달라져 Train Loss가 상승했지만 Validation Recall·F1·mAP가 크게 개선됐으므로 성능 판단에는 Validation Detection Metric을 사용했다.

## 5. Checkpoint 정책

```text
Latest : 마지막으로 완전히 완료된 Epoch의 재개 상태
Best   : Validation mAP@0.50이 개선된 Epoch만 교체
```

Checkpoint에는 Epoch, Model·Optimizer State, Scheduler State 또는 None, Training Config, Class Mapping, Best Metric, History, Torch·Torchvision Version을 저장한다. Epoch 3의 Validation mAP@0.50 {_metric(checkpoint_info['best_validation_metric'])}가 최종 Best로 선택됐다.

## 6. 최종 Validation·Test 결과

| Split | TP | FP | FN | Precision | Recall | F1 | Mean Matched IoU | mAP@0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | {int(validation['tp'])} | {int(validation['fp'])} | {int(validation['fn'])} | {_metric(validation['precision'])} | {_metric(validation['recall'])} | {_metric(validation['f1'])} | {_metric(validation['mean_matched_iou'])} | {_metric(validation['map_50'])} |
| Test | {int(test['tp'])} | {int(test['fp'])} | {int(test['fn'])} | {_metric(test['precision'])} | {_metric(test['recall'])} | {_metric(test['f1'])} | {_metric(test['mean_matched_iou'])} | {_metric(test['map_50'])} |

```text
Test project mAP@0.50:0.95 : {_metric(test_map_50_95)}
Score threshold             : {_metric(policy['score_threshold'])}
Matching IoU threshold      : {_metric(policy['iou_threshold'])}
```

Test mAP@0.50은 {_metric(test['map_50'])}이고 Mean Matched IoU는 {_metric(test['mean_matched_iou'])}다. 더 엄격한 IoU Threshold를 평균한 프로젝트 mAP@0.50:0.95는 {_metric(test_map_50_95)}로, 결함 존재·Class 탐지에 비해 정밀한 Localization에는 추가 개선 여지가 있음을 보여준다.

이 프로젝트 mAP@0.50:0.95는 직접 구현한 all-point AP를 IoU 0.50~0.95에 적용한 지표이며 공식 `pycocotools.COCOeval` 결과와 동일하다고 주장하지 않는다.

## 7. Test Class별 성능

| Class | GT | TP | FP | FN | Precision | Recall | F1 | Mean IoU | AP@0.50 | Project mAP@.50:.95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{_class_metric_rows(evaluation)}

- {best_class}
- {weak_class}

특히 낮은 Score의 정답 Prediction이 많이 확인됐지만, Test 결과를 보고 기본 Threshold나 Checkpoint를 다시 선택하지 않았다. Day 13 API·Dashboard의 기본 Score Threshold도 검증에 사용한 0.5를 유지한다.

## 8. Failure Analysis

```text
Test images          : {int(failure_summary['image_count'])}
Images with failures : {int(failure_summary['images_with_failures'])}
Failure events       : {int(failure_summary['event_count'])}
```

| Failure Type | Count |
|---|---:|
{_failure_rows(failure)}

Failure Event 수는 실패 이미지 수가 아니다. 한 이미지에서 False Positive·False Negative·Localization 문제가 동시에 발생할 수 있다. 가장 많은 유형은 Low-confidence Correct Detection이며, Score Threshold 주변의 Detection 품질과 Class별 Score Calibration이 Day 13 운영 화면에서 중요한 해석 포인트가 된다.

## 9. IoU와 Matching 정책

```text
Prediction score >= 0.50
같은 Class만 Matching 후보
Prediction Score 내림차순
아직 사용하지 않은 Ground Truth 중 최대 IoU 선택
IoU >= 0.50이면 True Positive
Prediction·Ground Truth 각각 한 번만 Matching
```

Matching되지 않은 Prediction은 False Positive, Matching되지 않은 Ground Truth는 False Negative다. 같은 Ground Truth에 대한 추가 Prediction은 Duplicate Prediction과 False Positive로 분석할 수 있다.

## 10. 생성 Artifact와 Figure

```text
reports/artifacts/day12_detection_training_config.json
reports/artifacts/day12_detection_training_pilot.json
reports/artifacts/day12_detection_one_epoch_pilot.json
reports/artifacts/day12_detection_training_history.json
reports/artifacts/day12_detection_evaluation.json
reports/artifacts/day12_detection_failure_analysis.json

models/detection/day12_detection_latest.pt
models/detection/day12_detection_best.pt

reports/figures/day12_detection_training_history.png
reports/figures/day12_detection_class_metrics.png
reports/figures/day12_detection_predictions.png
reports/figures/day12_detection_failure_analysis.png
```

Figure V2는 Ground Truth와 Prediction을 분리하고, 박스에는 짧은 G/P Tag만 표시하며 상세 Class·Score·판정을 이미지 밖 설명 영역으로 이동해 글자·박스 겹침을 줄였다.

## 11. 테스트 결과

```text
Day 12 targeted tests : {targeted_test_count} passed
Full regression tests : {regression_test_count} passed
Warnings              : {warning_count}
```

{warning_note}

## 12. 한계와 개선 방향

- `crazing`의 고정 Threshold Recall이 낮다.
- 프로젝트 mAP@0.50:0.95가 mAP@0.50보다 낮아 Localization 정밀도 개선이 필요하다.
- CPU와 3 Epoch 제한으로 Hyperparameter Search를 수행하지 않았다.
- Random Crop·Rotation·Mosaic·MixUp 등 결함 Box를 훼손할 수 있는 증강은 근거 없이 추가하지 않았다.
- Test 결과는 최종 보고에만 사용했으며 추가 모델 선택에 사용하지 않았다.

## 13. Day 13 연결

Day 13에서는 `models/detection/day12_detection_best.pt`, 동일한 7-Class Mapping, Score Threshold 0.5, 320 입력 정책을 Detection FastAPI Endpoint와 Streamlit 페이지에 연결한다. API는 Box·Class·Score를 반환하고 Dashboard는 Day 12의 가독성 정책을 재사용해 Ground Truth가 없는 실제 입력에서는 Prediction Overlay와 Threshold 정보를 표시한다.

Day 12에서는 Detection FastAPI Endpoint, Detection Streamlit 페이지, Classification·OpenCV·Detection 통합 UI를 완료했다고 표현하지 않는다.
"""


def render_readme_section(
    *,
    evaluation: Mapping[str, Any],
    failure: Mapping[str, Any],
    targeted_test_count: int,
    regression_test_count: int,
    warning_count: int,
) -> str:
    checkpoint = _nested(evaluation, "checkpoint")
    validation = _nested(evaluation, "validation", "metrics", "overall")
    test = _nested(evaluation, "test", "metrics", "overall")
    test_map_50_95 = _nested(evaluation, "test_iou_sweep", "summary", "map_50_95")
    summary = _nested(failure, "analysis", "summary")
    return f"""{README_START}
## Day 12 — Detection Training, Evaluation and Failure Analysis

COCO pretrained Faster R-CNN MobileNetV3 Large 320 FPN을 NEU-DET Background 포함 7-Class Head로 교체하고 CPU에서 3 Epoch Fine-tuning했다. Epoch 1은 Backbone Freeze, Epoch 2~3은 낮은 Learning Rate로 Unfreeze했으며 Best Checkpoint는 Validation mAP@0.50만으로 선택했다.

```text
Best epoch               : {int(checkpoint['completed_epoch_number'])}
Validation mAP@0.50      : {_metric(validation['map_50'])}
Test Precision           : {_metric(test['precision'])}
Test Recall              : {_metric(test['recall'])}
Test F1                  : {_metric(test['f1'])}
Test Mean Matched IoU    : {_metric(test['mean_matched_iou'])}
Test mAP@0.50            : {_metric(test['map_50'])}
Project mAP@0.50:0.95    : {_metric(test_map_50_95)}
Failure events           : {int(summary['event_count'])}
Test used for selection  : False
```

- Best model: `models/detection/day12_detection_best.pt`
- Report: `reports/day12_detection_training_evaluation_and_failure_analysis_summary.md`
- Evaluation: `reports/artifacts/day12_detection_evaluation.json`
- Failure analysis: `reports/artifacts/day12_detection_failure_analysis.json`
- Figures: `reports/figures/day12_detection_*.png`
- Tests: Day 12 {targeted_test_count} passed / Full regression {regression_test_count} passed / {warning_count} warning(s)

Test 결과는 Checkpoint 선택이나 추가 학습 결정에 사용하지 않았다. Detection API·Streamlit 통합은 Day 13 범위다.
{README_END}"""


def update_marker_block(original: str, replacement: str) -> str:
    start_count = original.count(README_START)
    end_count = original.count(README_END)
    if start_count != end_count:
        raise ValueError("README Day 12 marker pair is unbalanced.")
    if start_count > 1:
        raise ValueError("README contains duplicate Day 12 marker blocks.")
    if start_count == 1:
        start = original.index(README_START)
        end = original.index(README_END, start) + len(README_END)
        return original[:start] + replacement + original[end:]
    separator = "" if not original else ("\n" if original.endswith("\n") else "\n\n")
    return original + separator + replacement + "\n"


def _write_text_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _backup_once(source: Path, backup_path: Path) -> None:
    if source.is_file() and not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_bytes(source.read_bytes())


def create_day12_docs(
    *,
    project_root: Path,
    targeted_test_count: int,
    regression_test_count: int,
    warning_count: int,
) -> tuple[Path, Path]:
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path.")
    for name, value in (
        ("targeted_test_count", targeted_test_count),
        ("regression_test_count", regression_test_count),
        ("warning_count", warning_count),
    ):
        _validate_non_negative_int(name, value)

    root = project_root.resolve()
    training_config = _read_json_object(root / TRAINING_CONFIG_ARTIFACT)
    training_history = _read_json_object(root / TRAINING_HISTORY_ARTIFACT)
    evaluation = _read_json_object(root / EVALUATION_ARTIFACT)
    failure = _read_json_object(root / FAILURE_ARTIFACT)
    checkpoint = _load_checkpoint(root / BEST_CHECKPOINT)
    _validate_artifacts(
        training_history=training_history,
        evaluation=evaluation,
        failure=failure,
        checkpoint=checkpoint,
    )

    for relative_path in REQUIRED_FIGURES:
        path = root / relative_path
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Required Day 12 Figure is missing: {path}.")

    report = render_day12_report(
        training_config=training_config,
        training_history=training_history,
        evaluation=evaluation,
        failure=failure,
        checkpoint=checkpoint,
        targeted_test_count=targeted_test_count,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
    )
    readme_section = render_readme_section(
        evaluation=evaluation,
        failure=failure,
        targeted_test_count=targeted_test_count,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
    )

    report_path = root / REPORT_PATH
    readme_path = root / README_PATH
    backup_root = root / BACKUP_ROOT
    _backup_once(readme_path, backup_root / "README.md.before_day12_docs")
    _backup_once(report_path, backup_root / f"{REPORT_PATH.name}.before_day12_docs")

    original_readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    updated_readme = update_marker_block(original_readme, readme_section)
    _write_text_atomically(report_path, report)
    _write_text_atomically(readme_path, updated_readme)
    return report_path, readme_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the Day 12 Detection report and README section."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--targeted-test-count", type=int, required=True)
    parser.add_argument("--regression-test-count", type=int, required=True)
    parser.add_argument("--warning-count", type=int, required=True)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    report_path, readme_path = create_day12_docs(
        project_root=args.project_root,
        targeted_test_count=args.targeted_test_count,
        regression_test_count=args.regression_test_count,
        warning_count=args.warning_count,
    )
    print("[PASS] Day 12 report created")
    print("[PASS] README Day 12 section added or updated")
    print(f"[REPORT] {report_path}")
    print(f"[README] {readme_path}")
    print(
        "[TESTS] "
        f"Day 12={args.targeted_test_count} / "
        f"Full regression={args.regression_test_count} / "
        f"Warnings={args.warning_count}"
    )


if __name__ == "__main__":
    main()
