# Day 5 — Misclassified Image Analysis and Visualization

## 1. 목표

Day 4 ResNet18 Test 평가에서 발생한 오분류 19장을 분석하고 시각화했다.

입력 Artifact:

```text
reports/artifacts/day4_resnet18_test_evaluation.json
```

모델을 다시 추론하지 않고 Day 4 JSON의 `sample_results`와 저장된
`image_path`를 직접 사용했다.

## 2. Class 및 오류 정책

```text
0 = NORMAL
1 = DEFECT
Positive Class = DEFECT
Classification Threshold = 0.5
```

False Positive:

```text
Ground Truth = NORMAL
Prediction = DEFECT
```

False Negative:

```text
Ground Truth = DEFECT
Prediction = NORMAL
```

제조 관점에서 False Positive는 불필요한 폐기와 재검사 비용을 만들 수 있다.

False Negative는 실제 불량을 정상으로 통과시키므로 불량 출하와 품질 문제로
이어질 수 있어 더 위험한 오류다.

## 3. 분석 지표

Threshold Distance:

```python
threshold_distance = abs(
    defect_probability - 0.5
)
```

값이 작으면 결정 경계에 가까운 애매한 오류이고, 값이 크면 모델이 특정
Class를 강하게 선택했지만 틀린 오류다.

Wrong Prediction Confidence:

```python
if prediction == 1:
    wrong_prediction_confidence = defect_probability
else:
    wrong_prediction_confidence = 1 - defect_probability
```

## 4. 구현 파일

```text
src/evaluation/misclassification_analysis.py
src/evaluation/misclassification_visualization.py
scripts/run_day5_misclassification_analysis.py
tests/test_misclassification_analysis.py
tests/test_misclassification_visualization.py
tests/test_day5_misclassification_analysis.py
```

분석 모듈은 JSON Schema 검증, 오분류 추출, FP·FN 분리, 확신도 계산,
통계 생성과 Atomic JSON 저장을 담당한다.

시각화 모듈은 이미지 존재 여부와 손상 여부를 검사하고 RGB로 로딩한 뒤
FP·FN·전체 오분류 Grid를 고해상도 PNG로 저장한다.

## 5. JSON 검증 정책

실제 Day 4 표본 배열 Key:

```text
sample_results
```

각 표본의 필수 필드:

```text
sample_index
image_path
ground_truth_label
ground_truth_class_name
raw_logit
defect_probability
prediction
prediction_class_name
correct
```

추가 검증:

```text
sample_index 중복 금지
Label과 Prediction은 0 또는 1
Probability는 0.0 이상 1.0 이하
NaN과 Infinity 금지
correct는 bool
correct == (ground_truth_label == prediction) 교차 검증
FP와 FN의 Label 조합 검증
```

## 6. 실제 분석 결과

```text
Total Test Samples    : 715
Correct Samples       : 696
Misclassified Samples : 19
False Positive        : 13
False Negative        : 6
Error Rate            : 2.66%
```

오분류 중 False Positive는 13장, False Negative는 6장이었다.

현재 모델은 불량 누락을 줄이는 대신 일부 정상 제품을 보수적으로 불량으로
판정하는 경향을 보였다.

## 7. Confidence 통계

```text
Threshold Distance
Minimum = 0.004138648510
Maximum = 0.328241229057
Mean    = 0.110779961473

Wrong Prediction Confidence
Minimum = 0.504138648510
Maximum = 0.828241229057
Mean    = 0.610779961473
```

## 8. 주요 오분류

가장 확신한 오분류:

```text
Sample Index = 202
File = cast_ok_0_7839.jpeg
Error Type = FALSE_POSITIVE
Ground Truth = NORMAL
Prediction = DEFECT
P(DEFECT) = 0.828241
```

가장 확신한 False Negative:

```text
Sample Index = 342
File = cast_def_0_150.jpeg
Error Type = FALSE_NEGATIVE
Ground Truth = DEFECT
Prediction = NORMAL
P(DEFECT) = 0.256505
Wrong Prediction Confidence = 0.743495
```

결정 경계에 가장 가까운 오류:

```text
Sample Index = 157
File = cast_ok_0_4497.jpeg
Error Type = FALSE_POSITIVE
P(DEFECT) = 0.504139
Threshold Distance = 0.004139
```

## 9. 생성 Artifact

분석 JSON:

```text
reports/artifacts/day5_resnet18_misclassification_analysis.json
```

시각화:

```text
reports/figures/day5_resnet18_false_positives.png
reports/figures/day5_resnet18_false_negatives.png
reports/figures/day5_resnet18_all_misclassifications.png
```

각 이미지에는 Error Type, Sample Index, 파일명, Ground Truth,
Prediction, P(DEFECT), Wrong Prediction Confidence와 Threshold
Distance를 표시했다.

matplotlib `Agg` Backend를 사용해 Headless 환경에서도 동작하며,
`plt.show()`에 의존하지 않는다.

JSON과 PNG는 임시 파일에 먼저 저장한 뒤 `os.replace()`로 교체한다.

## 10. 테스트

Day 5 단위 및 통합 테스트:

```text
23 passed
```

프로젝트 전체 회귀 테스트:

```text
1164 passed
```

Day 5 기능 추가 후 기존 Dataset, DataLoader, CNN Baseline과 ResNet18
학습·평가 기능에 회귀 문제가 발생하지 않았다.

## 11. 완료 상태

```text
Day 4 평가 JSON 실제 Schema 연결 완료
715개 표본 검증 완료
오분류 19개 추출 완료
False Positive 13개 분석 완료
False Negative 6개 분석 완료
분석 JSON 생성 완료
FP·FN·전체 PNG 생성 완료
단위 및 통합 테스트 완료
전체 회귀 테스트 완료
```

Day 5의 오분류 분석 및 시각화 구현·실행·검증을 완료했다.
