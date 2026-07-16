# Day 3 - CNN Baseline Training and Evaluation Summary

## 1. Day 3 목표

Day 3의 목표는 제조 이미지 정상·불량 이진 분류를 위한 경량 CNN Baseline을 직접 구현하고, 실제 데이터로 학습한 뒤 독립된 Test Dataset에서 최종 성능을 평가하는 것이었다.

이번 단계에서는 단순히 Model Class만 작성하지 않고 다음 전체 흐름을 완성했다.

```text
Image Dataset

↓

PyTorch DataLoader

↓

CNNBaseline

↓

Raw Logit

↓

BCEWithLogitsLoss

↓

Adam Optimizer

↓

Train Epoch

↓

Validation Epoch

↓

Validation Loss 기준 Best Model 저장

↓

Best Checkpoint 복원

↓

Test Dataset Evaluation

↓

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

↓

Sample별 Prediction Artifact
```

Day 3에서는 Test Dataset을 Model 선택이나 Hyperparameter 조정에 사용하지 않았다.

Model 선택은 Validation Loss만 사용했으며, Test Dataset은 최종 일반화 성능 평가에만 사용했다.

---

## 2. 프로젝트 정보

프로젝트명:

`Manufacturing Vision Defect Analysis System`

한국어 프로젝트명:

`제조 비전 결함 분석 시스템`

현재 분류 문제:

```text
0 = NORMAL

1 = DEFECT
```

Positive Class:

```text
DEFECT

Label 1
```

현재 Model:

```text
CNNBaseline
```

향후 비교 Model:

```text
ResNet18 Transfer Learning
```

---

## 3. Dataset 구성

사용 Dataset:

`Casting Product Image Data for Quality Inspection`

실제 사용 경로:

```text
data/raw/
casting_product_images/
casting_data/
casting_data/
```

Class Mapping:

```python
CLASS_TO_INDEX = {
    "ok_front": 0,
    "def_front": 1,
}
```

전체 Dataset:

| Split      | NORMAL | DEFECT | Total |
| ---------- | -----: | -----: | ----: |
| Train      |  2,300 |  3,006 | 5,306 |
| Validation |    575 |    752 | 1,327 |
| Test       |    262 |    453 |   715 |
| Total      |  3,137 |  4,211 | 7,348 |

Train·Validation 분할 기준:

```text
Validation Ratio

0.2
```

```text
Random Seed

42
```

```text
Stratified Split

사용
```

Test Dataset은 원본 Test Split을 그대로 유지했다.

---

## 4. Image Transform

공식 입력 크기:

```text
224 × 224
```

Channel:

```text
RGB

3 Channels
```

Normalization:

```python
IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)
```

Train Transform:

```text
Resize

Random Horizontal Flip

Small Random Rotation

Tensor 변환

ImageNet Normalization
```

Validation·Test Transform:

```text
Resize

Tensor 변환

ImageNet Normalization
```

Validation·Test에는 Random Augmentation을 적용하지 않았다.

---

## 5. CNNBaseline Architecture

현재 CNNBaseline은 CPU 환경에서도 빠르게 학습 가능한 경량 구조로 구현했다.

Architecture:

```text
Input

[B, 3, 224, 224]

↓

Conv2d

3 → 8

Kernel 3

Padding 1

↓

ReLU

↓

MaxPool2d

224 → 112

↓

Conv2d

8 → 16

Kernel 3

Padding 1

↓

ReLU

↓

MaxPool2d

112 → 56

↓

Conv2d

16 → 32

Kernel 3

Padding 1

↓

ReLU

↓

MaxPool2d

56 → 28

↓

AdaptiveAvgPool2d

[1, 1]

↓

Flatten

[B, 32]

↓

Linear

32 → 1

↓

Raw Logit

[B]
```

전체 Parameter:

```text
6,065
```

CNNBaseline은 완전 연결층에 큰 Feature Map을 직접 입력하지 않고 `AdaptiveAvgPool2d`를 사용했다.

이 설계의 장점:

```text
Parameter 수 감소

CPU 학습 비용 감소

입력 공간 크기 변화 대응

ResNet18 비교용 경량 Baseline 확보
```

Convolution Layer는 `Sequential` 안에 숨기지 않고 이름 있는 Layer로 구성했다.

이유:

```text
향후 Grad-CAM Target Layer 접근

Layer별 Feature 확인

Architecture 설명 용이
```

---

## 6. Binary Output 설계

CNNBaseline은 Sigmoid Probability를 직접 반환하지 않는다.

Model Output:

```text
Raw Logit

Shape:

[B]
```

Model 내부:

```text
Sigmoid 없음
```

Loss:

```text
BCEWithLogitsLoss
```

이 설계는 Sigmoid와 Binary Cross Entropy를 수치적으로 안정적인 하나의 연산으로 처리한다.

학습:

```text
Raw Logit

↓

BCEWithLogitsLoss
```

평가:

```text
Raw Logit

↓

Sigmoid

↓

DEFECT Probability
```

Prediction:

```python
prediction = (
    probability
    >= 0.5
)
```

현재 Threshold:

```text
0.5
```

Probability가 Threshold와 정확히 같으면 DEFECT로 분류한다.

---

## 7. Loss Function

사용 Loss:

```text
BCEWithLogitsLoss
```

Reduction:

```text
mean
```

현재 Class 불균형은 존재하지만 극단적이지 않으므로 CNN Baseline에서는 `pos_weight`를 사용하지 않았다.

현재 정책:

```text
USE_POSITIVE_CLASS_WEIGHT

False
```

Label은 Dataset에서 Integer로 유지한다.

```text
0

1
```

Loss 계산 직전에:

```text
torch.float32
```

로 변환한다.

이유:

```text
Dataset Label 의미 유지

Metric 계산 용이

BCE Target Dtype 요구 충족
```

---

## 8. Optimizer

사용 Optimizer:

```text
Adam
```

설정:

```text
Learning Rate

0.001
```

```text
Weight Decay

0.0
```

```text
Betas

(0.9, 0.999)
```

```text
Epsilon

1e-8
```

Gradient 초기화:

```python
optimizer.zero_grad(
    set_to_none=True
)
```

현재 CNN Baseline에서는 다음 기능을 추가하지 않았다.

```text
Learning Rate Scheduler

AdamW

SGD

Gradient Clipping

Mixed Precision

Early Stopping
```

이유:

현재 단계의 목적은 복잡한 최적화보다 명확한 비교 기준이 되는 Baseline을 만드는 것이기 때문이다.

---

## 9. Train·Validation Epoch Runner

Train 흐름:

```text
model.train()

↓

Batch

↓

Image Device 이동

↓

Label Target 변환

↓

Forward

↓

Loss

↓

zero_grad

↓

Backward

↓

Optimizer Step

↓

Loss·Accuracy 누적
```

Validation 흐름:

```text
model.eval()

↓

torch.inference_mode()

↓

Batch

↓

Forward

↓

Loss·Accuracy 누적
```

Average Loss는 Batch Loss의 단순 평균을 사용하지 않았다.

계산:

```text
Batch Mean Loss

×

Batch Sample Count

↓

전체 합

÷

전체 Sample Count
```

마지막 Batch 크기가 다를 수 있으므로 Sample 가중 평균을 사용했다.

---

## 10. Training Pipeline

실제 학습 설정:

| 항목            |               값 |
| ------------- | --------------: |
| Epoch         |               5 |
| Batch Size    |              32 |
| Device        |             CPU |
| Random Seed   |              42 |
| Threshold     |             0.5 |
| Best Model 기준 | Validation Loss |
| Parameter     |           6,065 |

Best Model 선택 기준:

```text
Lowest Validation Loss
```

Validation Accuracy가 가장 높은 Epoch가 아니라 Validation Loss가 가장 낮은 Epoch를 저장했다.

동일한 Validation Loss가 발생하면 먼저 등장한 Epoch를 유지하도록 구현했다.

---

## 11. 실제 CNNBaseline 학습 결과

실행 명령:

```powershell
python -m scripts.run_day3_cnn_baseline_training
```

실제 결과:

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Best Updated |
| ----: | ---------: | -------------: | --------------: | ------------------: | :----------: |
|     1 |   0.619515 |         64.68% |        0.476329 |              82.82% |      Yes     |
|     2 |   0.469879 |         79.16% |        0.465499 |              76.94% |      Yes     |
|     3 |   0.455239 |         80.21% |        0.492534 |              77.62% |      No      |
|     4 |   0.441022 |         80.49% |        0.479610 |              78.52% |      No      |
|     5 |   0.440413 |         80.63% |        0.509245 |              77.77% |      No      |

학습 시간:

```text
646.16 Seconds

약 10.77 Minutes
```

Best Epoch:

```text
Epoch 2
```

Best Validation Loss:

```text
0.465498844753
```

Best Validation Accuracy:

```text
76.94%
```

---

## 12. Training Result 해석

Train Loss:

```text
Epoch 1

0.619515

↓

Epoch 5

0.440413
```

Train Accuracy:

```text
Epoch 1

64.68%

↓

Epoch 5

80.63%
```

따라서 Model은 실제로 학습했다.

Epoch 2 이후:

```text
Train Loss

계속 감소
```

하지만:

```text
Validation Loss

다시 증가
```

했다.

이는 Epoch 2 이후 일반화 성능 개선이 정체되고 경미한 Overfitting이 시작된 신호로 해석할 수 있다.

Validation Loss 기준 Best Checkpoint를 저장했으므로 Epoch 3~5의 악화된 Validation Loss가 최종 Test 평가에 사용되지 않았다.

---

## 13. Best Checkpoint

저장 경로:

```text
models/checkpoints/
cnn_baseline_best.pt
```

Checkpoint Version:

```text
1
```

저장 정보:

```text
Model 이름

Model Module

Loss 이름

Optimizer 이름

Best Epoch

Configured Epoch

Classification Threshold

Best Model 선택 기준

Model State

Optimizer State

Train Result

Validation Result
```

Best Checkpoint:

```text
Epoch 2
```

Checkpoint Loader는 다음을 검증한다.

```text
파일 존재

.pt 또는 .pth 확장자

Checkpoint Version

필수 Key

Model 이름

Model Module

Epoch 범위

Threshold 범위

Metric 이름

Model State Key

Tensor Type

Tensor 유한성

strict=True Weight 복원
```

Loader는 Model Mode를 변경하지 않는다.

```text
Train Mode 입력

→

Train Mode 유지
```

```text
Evaluation Mode 입력

→

Evaluation Mode 유지
```

---

## 14. Evaluation Runner

Evaluation Runner는 Checkpoint를 직접 읽지 않는다.

역할 분리:

```text
Checkpoint Loader

→

Best Weight 복원
```

```text
Evaluation Runner

→

전달받은 Model 평가
```

Evaluation 흐름:

```text
model.eval()

↓

torch.inference_mode()

↓

Test Batch

↓

Raw Logit

↓

Loss

↓

Sigmoid

↓

DEFECT Probability

↓

Threshold

↓

Prediction

↓

전체 결과 CPU 수집
```

반환:

```text
Average Loss

Accuracy

Sample Count

Batch Count

Classification Threshold

Labels

Logits

Probabilities

Predictions
```

Evaluation 후:

```text
Model State 변경 없음

Parameter Gradient 없음

Model Evaluation Mode 유지
```

---

## 15. Classification Metrics

Metric 모듈은 Model Forward를 수행하지 않는다.

입력:

```text
Ground Truth Labels

Predictions
```

출력:

```text
Accuracy

Precision

Recall

F1 Score

TN

FP

FN

TP

Confusion Matrix
```

Confusion Matrix 순서:

```text
[
    [TN, FP],
    [FN, TP],
]
```

Positive Class:

```text
1 = DEFECT
```

Zero Division 정책:

```text
Precision 분모 0

→

0.0
```

```text
Recall 분모 0

→

0.0
```

```text
F1 분모 0

→

0.0
```

---

## 16. 실제 Test Dataset 평가

실행 명령:

```powershell
python -m scripts.run_day3_cnn_baseline_evaluation
```

평가 Dataset:

| Class  | Count |
| ------ | ----: |
| NORMAL |   262 |
| DEFECT |   453 |
| Total  |   715 |

Test Batch:

```text
23
```

Best Model:

```text
CNNBaseline

Epoch 2
```

평가 시간:

```text
8.97 Seconds
```

---

## 17. 실제 Test 성능

| Metric    |       결과 |
| --------- | -------: |
| Test Loss | 0.453337 |
| Accuracy  |   76.92% |
| Precision |   82.88% |
| Recall    |   80.13% |
| F1 Score  |   81.48% |

정확한 값:

```text
Test Loss

0.453337371391
```

```text
Accuracy

0.769230769231
```

```text
Precision

0.828767123288
```

```text
Recall

0.801324503311
```

```text
F1 Score

0.814814814815
```

---

## 18. Confusion Matrix

실제 결과:

```text
tensor(
    [
        [187, 75],
        [90, 363],
    ]
)
```

구조:

```text
                Predicted

                NORMAL    DEFECT

Actual NORMAL      187       75

Actual DEFECT       90      363
```

Count:

| 항목             |   수 |
| -------------- | --: |
| True Negative  | 187 |
| False Positive |  75 |
| False Negative |  90 |
| True Positive  | 363 |
| Correct        | 550 |
| Incorrect      | 165 |

---

## 19. Precision 해석

DEFECT Prediction:

```text
438
```

실제 DEFECT:

```text
363
```

Precision:

```text
363

÷

438

=

82.88%
```

의미:

Model이 DEFECT라고 예측한 이미지 100장 중 약 83장이 실제 DEFECT다.

False Positive:

```text
75
```

이는 실제 NORMAL 이미지를 DEFECT로 잘못 분류한 경우다.

---

## 20. Recall 해석

실제 DEFECT:

```text
453
```

정확히 탐지:

```text
363
```

Recall:

```text
363

÷

453

=

80.13%
```

실제 DEFECT 중 약 80%를 탐지했다.

False Negative:

```text
90
```

이는 실제 DEFECT를 NORMAL로 잘못 분류한 경우다.

제조 품질 검사에서는 실제 불량을 정상으로 통과시키는 오류이므로 중요한 위험 지표다.

False Negative Rate:

```text
90

÷

453

=

19.87%
```

현재 CNN Baseline은 실제 불량 약 100장 중 약 20장을 놓쳤다.

---

## 21. NORMAL Class 성능

실제 NORMAL:

```text
262
```

정확한 NORMAL:

```text
187
```

Specificity:

```text
187

÷

262

=

71.37%
```

False Positive Rate:

```text
75

÷

262

=

28.63%
```

현재 CNN Baseline은 실제 정상 약 100장 중 약 29장을 불량으로 잘못 분류했다.

---

## 22. Validation과 Test 비교

Best Validation:

```text
Loss

0.465499
```

```text
Accuracy

76.94%
```

Test:

```text
Loss

0.453337
```

```text
Accuracy

76.92%
```

Accuracy 차이:

```text
약 0.02 Percentage Point
```

Validation과 Test Accuracy가 거의 동일하다.

이는 현재 결과에서 다음을 의미한다.

```text
Validation 성능이 Test에서도 유지됨

심한 Overfitting 징후 없음

Validation Loss 기준 Best Epoch 선택이 정상 동작

Test 일반화 성능이 안정적
```

---

## 23. Class 불균형 기준 비교

Test Dataset DEFECT 비율:

```text
453

÷

715

=

63.36%
```

모든 이미지를 DEFECT로만 예측하는 단순 Majority Class Baseline:

```text
Accuracy

63.36%
```

CNNBaseline:

```text
Accuracy

76.92%
```

향상:

```text
약 13.57 Percentage Point
```

CNNBaseline은 단순히 다수 Class인 DEFECT만 반복 예측한 것이 아니다.

Prediction 분포:

```text
Predicted NORMAL

277
```

```text
Predicted DEFECT

438
```

두 Class를 모두 분류했다.

---

## 24. Probability 분포

DEFECT Probability:

```text
Minimum

0.034242305905
```

```text
Maximum

0.978111147881
```

```text
Mean

0.539899587631
```

현재 Threshold:

```text
0.5
```

0에 가까운 NORMAL 확신 Sample과 1에 가까운 DEFECT 확신 Sample이 모두 존재했다.

다만 Probability 평균만으로 Probability Calibration 성능을 판단하지는 않는다.

---

## 25. CNNBaseline의 의미

현재 CNNBaseline:

```text
Parameter

6,065
```

최종 성능:

```text
Accuracy

76.92%
```

```text
F1

81.48%
```

매우 작은 Parameter 수로 의미 있는 분류 성능을 확보했다.

따라서 이후 ResNet18 Transfer Learning의 성능을 비교할 명확한 기준선이 생겼다.

향후 비교 항목:

| 항목             | CNNBaseline |
| -------------- | ----------: |
| Parameter      |       6,065 |
| Best Epoch     |           2 |
| Test Loss      |      0.4533 |
| Accuracy       |      76.92% |
| Precision      |      82.88% |
| Recall         |      80.13% |
| F1             |      81.48% |
| False Positive |          75 |
| False Negative |          90 |

ResNet18에서는 단순 Accuracy뿐 아니라 다음을 중점 비교한다.

```text
Recall 증가 여부

False Negative 감소 여부

F1 증가 여부

Validation·Test 일반화 유지 여부
```

---

## 26. 현재 한계

현재 CNNBaseline의 한계:

```text
매우 작은 Feature Channel

3개의 단순 Convolution Block

Pretrained Feature 없음

복잡한 결함 Texture 표현 한계

False Negative 90장

False Positive 75장
```

현재 버전에서 수행하지 않은 작업:

```text
Threshold 최적화

Class Weight

Focal Loss

Learning Rate Scheduler

Early Stopping

Data Augmentation 탐색

Hyperparameter Search
```

이 기능들은 CNNBaseline에 과도한 복잡성을 추가하지 않고 ResNet18 비교 후 필요성을 판단한다.

---

## 27. 생성 Artifact

Best Checkpoint:

```text
models/checkpoints/
cnn_baseline_best.pt
```

Training History:

```text
reports/artifacts/
day3_cnn_baseline_training_history.json
```

Test Evaluation:

```text
reports/artifacts/
day3_cnn_baseline_test_evaluation.json
```

Test Evaluation JSON에는 715개 Sample별 다음 정보가 포함된다.

```text
Sample Index

Image Path

Ground Truth Label

Ground Truth Class Name

Raw Logit

DEFECT Probability

Prediction

Prediction Class Name

Correct 여부
```

이 결과는 향후 다음 기능에 재사용한다.

```text
오분류 이미지 분석

False Positive 분석

False Negative 분석

Confidence 기반 정렬

Grad-CAM 대상 선택
```

---

## 28. 주요 구현 파일

Model:

```text
src/models/
cnn_baseline.py
```

Loss:

```text
src/training/
loss_function.py
```

Optimizer:

```text
src/training/
optimizer.py
```

Epoch Runner:

```text
src/training/
epoch_runner.py
```

Training Pipeline:

```text
src/training/
training_pipeline.py
```

Checkpoint Loader:

```text
src/training/
checkpoint_loader.py
```

Evaluation Runner:

```text
src/evaluation/
evaluation_runner.py
```

Classification Metrics:

```text
src/evaluation/
classification_metrics.py
```

Training Script:

```text
scripts/
run_day3_cnn_baseline_training.py
```

Evaluation Script:

```text
scripts/
run_day3_cnn_baseline_evaluation.py
```

---

## 29. 테스트

전체 테스트 명령:

```powershell
python -m pytest `
    .\tests `
    -v
```

최종 결과:

```text
1110 passed
```

Test 범위:

```text
Dataset Config

Dataset Analysis

Dataset Visualization

Dataset Split

Image Dataset

Image Transform

DataLoader

Reproducibility

CNNBaseline

Loss Function

Optimizer

Train Epoch

Validation Epoch

Training Pipeline

Checkpoint Loader

Evaluation Runner

Classification Metrics

Training Script

Evaluation Script
```

`1110 passed`는 Parameterized Test의 각 입력 Case를 개별 Test Case로 집계한 결과다.

정상 경로뿐 아니라 다음 오류 조건을 포함한다.

```text
잘못된 Type

잘못된 Shape

잘못된 Dtype

빈 Tensor

잘못된 Label

NaN

Positive Infinity

Negative Infinity

Device 불일치

Checkpoint 필수 Key 누락

Checkpoint Version 불일치

Model 이름 불일치

Model Module 불일치

Model State Key 불일치

Weight Shape 불일치

Metric 내부 불일치

Confusion Matrix 불일치

빈 DataLoader

잘못된 JSON Artifact
```

---

## 30. 면접 설명

### Q. 왜 CNNBaseline을 먼저 구현했나요?

ResNet18 같은 전이학습 Model의 성능이 실제로 개선되었는지 판단하려면 비교 기준이 필요하기 때문이다.

CNNBaseline은 6,065개의 작은 Parameter로 직접 학습했으며 Test Accuracy 76.92%, F1 81.48%를 기록했다.

이 결과를 기준으로 ResNet18의 성능 향상과 계산 비용 증가를 비교할 수 있다.

---

### Q. 왜 Model 내부에 Sigmoid를 넣지 않았나요?

학습에서는 `BCEWithLogitsLoss`를 사용했다.

이 Loss는 Sigmoid와 Binary Cross Entropy를 하나의 수치적으로 안정적인 연산으로 처리한다.

따라서 Model은 Raw Logit만 반환하고, Probability가 필요한 평가 단계에서만 Sigmoid를 적용했다.

---

### Q. 왜 Validation Accuracy가 가장 높은 Epoch 1이 아니라 Epoch 2를 저장했나요?

현재 Best Model 선택 기준을 Validation Loss로 고정했기 때문이다.

Epoch 1의 Validation Accuracy는 더 높았지만 Epoch 2의 Validation Loss가 더 낮았다.

Accuracy는 Threshold를 적용한 이산 결과이고 Loss는 예측 Confidence까지 반영한다.

현재 Pipeline은 사전에 정한 Validation Loss 기준을 일관되게 사용했다.

---

### Q. Test Dataset은 언제 사용했나요?

Train·Validation 단계가 완료되고 Best Epoch 2가 확정된 후 한 번 최종 평가에 사용했다.

Test 결과를 보고 Epoch나 Hyperparameter를 다시 선택하지 않았다.

이를 통해 Test Leakage를 방지했다.

---

### Q. 제조 품질 검사에서 어떤 Metric이 중요하다고 생각하나요?

Accuracy뿐 아니라 Recall과 False Negative를 함께 봐야 한다.

현재 False Negative는 90장으로 실제 DEFECT가 NORMAL로 잘못 분류된 경우다.

제조 환경에서는 실제 불량이 정상으로 통과하는 위험이 있으므로 이후 ResNet18 비교에서도 Recall 증가와 False Negative 감소를 중요하게 확인할 예정이다.

---

### Q. Validation과 Test 결과 차이는 어땠나요?

Best Validation Accuracy는 76.94%, Test Accuracy는 76.92%였다.

차이는 약 0.02 Percentage Point로 매우 작았다.

현재 결과에서는 Validation 성능이 Test에서도 안정적으로 유지되었으며 심한 Overfitting 징후는 확인되지 않았다.

---

### Q. AI 도구는 어떻게 사용했나요?

AI 도구를 코드 초안과 검증 항목 설계에 활용했다.

하지만 각 파일의 역할, 입력·출력, Class 정의, Tensor Shape, Loss 계산, Checkpoint 정책, Metric 공식과 예외 처리를 직접 확인했다.

또한 실제 CPU 환경에서 학습과 Test 평가를 실행하고 결과를 검증했으며, 전체 테스트를 실행해 `1110 passed`를 확인했다.

---

## 31. Day 3 최종 결론

Day 3에서는 제조 이미지 정상·불량 이진 분류를 위한 CNNBaseline의 전체 학습·평가 Pipeline을 완성했다.

완료 기능:

```text
CNNBaseline

BCEWithLogitsLoss

Adam Optimizer

Train Epoch

Validation Epoch

Best Checkpoint 저장

Best Checkpoint 복원

Test Evaluation

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Sample별 Prediction JSON

전체 회귀 테스트
```

실제 결과:

```text
Best Epoch

2
```

```text
Test Accuracy

76.92%
```

```text
Test Precision

82.88%
```

```text
Test Recall

80.13%
```

```text
Test F1 Score

81.48%
```

```text
전체 테스트

1110 passed
```

현재 CNNBaseline은 향후 ResNet18 Transfer Learning과 비교할 검증된 성능 기준선이다.

다음 단계에서는 Day 3 결과를 README에 반영한 뒤 ResNet18 Transfer Learning 구현 단계로 이동한다.
