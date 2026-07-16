# Day 4 - ResNet18 Transfer Learning Training and Evaluation Summary

## 1. Day 4 목표

Day 4의 목표는 Day 3에서 구현한 `CNNBaseline`을 제거하거나 대체하는 것이 아니라,
동일한 Dataset·Transform·Loss·Threshold·평가 지표를 사용하는
`ResNet18 Transfer Learning` 모델을 구현하여 공정하게 비교하는 것이었다.

프로젝트명:

```text
Manufacturing Vision Defect Analysis System
```

한국어명:

```text
제조 비전 결함 분석 시스템
```

---

## 2. 공정 비교 조건

CNNBaseline과 ResNet18은 다음 조건을 동일하게 사용했다.

| 항목 | 설정 |
|---|---|
| Train Samples | 5,306 |
| Validation Samples | 1,327 |
| Test Samples | 715 |
| NORMAL Test Samples | 262 |
| DEFECT Test Samples | 453 |
| Image Size | 224 × 224 |
| Class Mapping | NORMAL=0, DEFECT=1 |
| Positive Class | DEFECT |
| Loss | BCEWithLogitsLoss |
| Classification Threshold | 0.5 |
| Best Model 기준 | Lowest Validation Loss |
| 평가 지표 | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Confusion Matrix 순서 | `[[TN, FP], [FN, TP]]` |

모델 Architecture, Pretrained Weight, Trainable Parameter 수와 학습 시간만 달라질 수 있도록 구성했다.

---

## 3. ResNet18 전이학습 설계

구현 파일:

```text
src/models/resnet18_transfer.py
```

구조:

```text
Input [B, 3, 224, 224]

→ torchvision ResNet18
→ ImageNet Pretrained Weight
→ Frozen Backbone
→ 기존 FC 512→1000 제거
→ 새 FC 512→1
→ squeeze(dim=1)
→ Raw Logit [B]
```

핵심 정책:

```text
Pretrained Weight        : ResNet18_Weights.DEFAULT
Backbone                 : 전체 Freeze
Classification Head      : Linear(512, 1)
Model 내부 Sigmoid       : 사용하지 않음
Threshold                : 0.5
Grad-CAM Target Layer    : resnet18.layer4.1.conv2
```

Parameter:

| 구분 | 수 |
|---|---:|
| Total Parameters | 11,177,025 |
| Trainable Parameters | 513 |
| Frozen Parameters | 11,176,512 |

Trainable Parameter 513개는 새 FC Head의 Weight 512개와 Bias 1개다.

---

## 4. Frozen Backbone과 BatchNorm 정책

단순히 `requires_grad=False`만 적용하면 `model.train()` 호출 시
BatchNorm Running Mean·Variance가 변경될 수 있다.

따라서 `ResNet18Transfer.train()`을 Override하여 다음 상태를 유지했다.

```text
Wrapper Model            : train()
Frozen Backbone          : eval()
Classification Head      : train()
Frozen BatchNorm Layers  : eval()
```

Validation·Test에서는 전체 모델을 Evaluation Mode로 전환한다.

이 정책을 통해 사전학습 Backbone을 진정한 Fixed Feature Extractor로 사용하고,
학습 중에는 새 FC Head만 갱신되도록 했다.

---

## 5. 기존 학습 Pipeline 재사용

다음 공통 구성요소는 Day 3 구현을 그대로 재사용했다.

```text
src/data/data_loader.py
src/reproducibility.py
src/training/loss_function.py
src/training/optimizer.py
src/training/epoch_runner.py
src/training/training_pipeline.py
src/training/checkpoint_loader.py
```

ResNet18 전용 실행 파일:

```text
scripts/run_day4_resnet18_training.py
scripts/run_day4_resnet18_evaluation.py
```

Optimizer에는 전체 ResNet18이 아니라 새 Classification Head Parameter만 전달했다.

```text
Optimizer Parameter Count = 513
```

---

## 6. 학습 설정

| 항목 | 설정 |
|---|---|
| Epoch | 5 |
| Batch Size | 32 |
| Device | CPU |
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 0.0 |
| Loss | BCEWithLogitsLoss |
| Threshold | 0.5 |
| Best Model 기준 | Validation Loss |
| Random Seed | 42 |

Checkpoint:

```text
models/checkpoints/resnet18_transfer_best.pt
```

Training History:

```text
reports/artifacts/day4_resnet18_training_history.json
```

---

## 7. 실제 학습 결과

실행:

```powershell
python -m scripts.run_day4_resnet18_training
```

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Best |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.513752 | 78.78% | 0.359830 | 93.07% | Yes |
| 2 | 0.325349 | 91.76% | 0.251585 | 96.38% | Yes |
| 3 | 0.248269 | 94.27% | 0.197622 | 97.29% | Yes |
| 4 | 0.203352 | 95.59% | 0.178660 | 96.99% | Yes |
| 5 | 0.170601 | 96.46% | 0.157920 | 97.06% | Yes |

Best Result:

```text
Best Epoch                : 5
Best Validation Loss      : 0.157919717199
Best Validation Accuracy  : 97.06%
```

실제 학습 시간:

```text
2,642.89초
약 44.05분
```

---

## 8. Best Checkpoint 복원

학습 완료 후 다음 절차로 Best Checkpoint를 검증했다.

```text
1. weights=None ResNet18Transfer 새 객체 생성
2. Checkpoint Metadata 검증
3. Model State Key·Tensor Shape 검증
4. strict=True State Loading
5. Evaluation Mode 확인
6. Dummy Forward 확인
```

복원 상태:

```text
Model Training Mode       : False
Backbone Training Mode    : False
Head Training Mode        : False
```

---

## 9. 실제 Test 평가 결과

실행:

```powershell
python -m scripts.run_day4_resnet18_evaluation
```

Test Dataset:

```text
NORMAL = 262
DEFECT = 453
Total = 715
Batch = 23
```

평가 결과:

| Metric | 결과 |
|---|---:|
| Test Loss | 0.158721205493 |
| Accuracy | 97.34% |
| Precision | 97.17% |
| Recall | 98.68% |
| F1 Score | 97.92% |

Confusion Matrix:

```text
tensor(
    [
        [249, 13],
        [6, 447],
    ]
)
```

Count:

```text
TN = 249
FP = 13
FN = 6
TP = 447

Correct = 696
Incorrect = 19
```

Probability:

```text
Min  = 0.013476800174
Max  = 0.999903678894
Mean = 0.668905854225
```

평가 시간:

```text
49.64초
```

생성 Artifact:

```text
reports/artifacts/day4_resnet18_test_evaluation.json
reports/artifacts/day4_cnn_resnet18_comparison.json
```

`day4_resnet18_test_evaluation.json`에는 Test 이미지 715장 각각의
경로, Ground Truth, Raw Logit, DEFECT Probability, Prediction, 정답 여부가 저장된다.

---

## 10. CNNBaseline과 ResNet18 비교

| 항목 | CNNBaseline | ResNet18 | 변화 |
|---|---:|---:|---:|
| Test Loss | 0.453337 | 0.158721 | -0.294616 |
| Accuracy | 76.92% | 97.34% | +20.42%p |
| Precision | 82.88% | 97.17% | +14.29%p |
| Recall | 80.13% | 98.68% | +18.54%p |
| F1 Score | 81.48% | 97.92% | +16.44%p |
| False Positive | 75 | 13 | 62장 감소 |
| False Negative | 90 | 6 | 84장 감소 |
| 전체 오분류 | 165 | 19 | 146장 감소 |

제조 품질 검사 관점에서 중요한 결과:

```text
False Negative 90 → 6
불량 누락 84장 감소
불량 누락 약 93.33% 감소
```

False Positive도 다음과 같이 감소했다.

```text
False Positive 75 → 13
정상 제품 오탐 62장 감소
정상 제품 오탐 약 82.67% 감소
```

전체 오분류는 다음과 같이 감소했다.

```text
165 → 19
146장 감소
약 88.48% 감소
```

---

## 11. Validation과 Test 일반화 확인

```text
Validation Accuracy = 97.06%
Test Accuracy       = 97.34%
차이                = 약 0.28%p
```

Validation과 Test 성능 차이가 작아,
현재 고정 Test Split에서도 전이학습 성능이 안정적으로 유지됐다.

다만 하나의 Dataset과 하나의 Test Split에서 얻은 결과이므로
실무에서는 제품 종류, 촬영 조건, 조명, 생산 시점이 다른 외부 데이터로 추가 검증해야 한다.

---

## 12. 테스트

Day 4 관련 신규 테스트:

```text
ResNet18 Architecture
Offline weights=None 생성
Binary Head
Raw Logit Shape
Backbone Freeze
BatchNorm Evaluation Mode 고정
Grad-CAM Target Layer
Trainable Parameter 제한
기존 Training Pipeline 호환
Checkpoint 저장·복원
Evaluation Result 검증
715개 Sample Result 구조
CNN Metric 추출
CNN·ResNet18 비교 계산
Atomic JSON 저장
```

최종 전체 회귀 테스트:

```powershell
python -m pytest `
    .	ests `
    -v
```

실제 결과:

```text
1141 passed
```

---

## 13. Day 4 완료 파일

```text
src/models/resnet18_transfer.py

scripts/run_day4_resnet18_training.py
scripts/run_day4_resnet18_evaluation.py

tests/test_resnet18_transfer.py
tests/test_day4_resnet18_training.py
tests/test_day4_resnet18_evaluation.py

models/checkpoints/resnet18_transfer_best.pt

reports/artifacts/day4_resnet18_training_history.json
reports/artifacts/day4_resnet18_test_evaluation.json
reports/artifacts/day4_cnn_resnet18_comparison.json

reports/day4_resnet18_transfer_learning_training_and_evaluation_summary.md
```

---

## 14. 현재 결론

Day 4에서는 사전학습 ResNet18을 Frozen Feature Extractor로 사용하고,
새로운 `Linear(512, 1)` Head만 학습하는 전이학습 모델을 구현했다.

동일한 데이터와 평가 조건에서 ResNet18은 CNNBaseline보다
Accuracy, Precision, Recall, F1 모두 크게 향상됐다.

특히 제조 품질 검사에서 가장 중요한 불량 누락 수가
`90장`에서 `6장`으로 감소하여,
CNNBaseline의 개선 목표였던 False Negative 문제를 크게 완화했다.

현재 모델 비교 결과에 따라 이후 추론·오분류 분석·Grad-CAM·API 단계의
기본 주 모델은 ResNet18 Best Checkpoint를 우선 사용하는 것이 적절하다.

---

## 15. 다음 확장 범위

다음 단계에서 검토할 범위:

```text
오분류 이미지 분리·시각화
False Positive·False Negative 사례 분석
Grad-CAM
단일 이미지 추론 Pipeline
FastAPI 추론 API
Streamlit Dashboard
```

아직 구현하지 않은 위 기능은 Day 4 완료 범위에 포함하지 않는다.
