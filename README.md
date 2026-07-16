# Manufacturing Vision Defect Analysis System

제조 이미지를 활용하여 정상·불량을 분류하고, 모델 학습부터 성능 평가·오분류 분석·시각적 설명·추론 API·대시보드까지 연결하는 제조 Vision 프로젝트입니다.

한국어 프로젝트명:

**제조 비전 결함 분석 시스템**

---

## 1. Project Overview

제조 품질 검사에서는 제품 이미지에서 불량 여부를 빠르고 일관되게 판단하는 것이 중요합니다.

수작업 검사만 사용하는 경우 검사자의 경험과 피로도에 따라 판단 편차가 발생할 수 있으며, 생산량이 증가하면 모든 이미지를 동일한 기준으로 검사하기 어렵습니다.

이 프로젝트는 제조 이미지를 입력받아 다음 흐름을 구현하는 것을 목표로 합니다.

```text
Manufacturing Image

↓

Image Data Analysis

↓

PyTorch Dataset

↓

DataLoader

↓

CNN Baseline

↓

ResNet18 Transfer Learning

↓

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

↓

Misclassification Analysis

↓

Grad-CAM

↓

FastAPI Inference API

↓

Streamlit Dashboard
```

현재까지는 이미지 데이터 분석, Dataset·DataLoader, CNN Baseline 학습, Best Checkpoint 저장·복원, 독립 Test Dataset 평가까지 완료했습니다.

---

## 2. Current Implementation Status

### Completed

```text
[완료] 제조 이미지 데이터 구조 분석

[완료] 정상·불량 Class Mapping

[완료] 이미지 파일 수 분석

[완료] 이미지 크기·Channel 분석

[완료] 손상 이미지 검증

[완료] Train·Validation Stratified Split

[완료] PyTorch Dataset

[완료] Train·Validation·Test Transform

[완료] PyTorch DataLoader

[완료] Random Seed 고정

[완료] CNN Baseline Architecture

[완료] BCEWithLogitsLoss

[완료] Adam Optimizer

[완료] Train Epoch Runner

[완료] Validation Epoch Runner

[완료] Training Pipeline

[완료] Validation Loss 기준 Best Checkpoint 저장

[완료] Best Checkpoint Loader

[완료] Binary Evaluation Runner

[완료] Accuracy

[완료] Precision

[완료] Recall

[완료] F1 Score

[완료] Confusion Matrix

[완료] 실제 CNN Baseline 학습

[완료] 실제 Test Dataset 평가

[완료] Sample별 Prediction JSON

[완료] 전체 자동 테스트
```

### Planned

```text
[예정] ResNet18 Transfer Learning

[예정] CNN Baseline·ResNet18 성능 비교

[예정] 오분류 이미지 분석

[예정] False Positive·False Negative 분석

[예정] Grad-CAM

[예정] FastAPI 추론 API

[예정] Streamlit Dashboard

[예정] 최종 포트폴리오 문서
```

현재 README에서는 아직 구현하지 않은 기능을 완료 기능으로 표시하지 않습니다.

---

## 3. Binary Classification Definition

현재 이진 분류 기준은 다음과 같습니다.

| Label | Class  | 의미        |
| ----: | ------ | --------- |
|     0 | NORMAL | 정상 제품 이미지 |
|     1 | DEFECT | 불량 제품 이미지 |

Class Mapping:

```python
CLASS_TO_INDEX = {
    "ok_front": 0,
    "def_front": 1,
}
```

Positive Class:

```text
1 = DEFECT
```

Precision·Recall·F1 Score는 DEFECT를 Positive Class로 계산합니다.

---

## 4. Dataset

사용 데이터:

**Casting Product Image Data for Quality Inspection**

실제 데이터 경로:

```text
data/
└── raw/
    └── casting_product_images/
        └── casting_data/
            └── casting_data/
                ├── train/
                │   ├── def_front/
                │   └── ok_front/
                └── test/
                    ├── def_front/
                    └── ok_front/
```

분석 대상:

```text
train/def_front

train/ok_front

test/def_front

test/ok_front
```

현재 프로젝트에서는 별도의 `casting_512x512` 데이터를 사용하지 않습니다.

동일하거나 유사한 이미지가 다른 경로에 중복될 가능성을 줄이고, 데이터 중복과 평가 누수 위험을 방지하기 위해 공식 Train·Test 구조만 사용합니다.

---

## 5. Dataset Summary

전체 이미지:

| Split          | NORMAL | DEFECT | Total |
| -------------- | -----: | -----: | ----: |
| Original Train |  2,875 |  3,758 | 6,633 |
| Original Test  |    262 |    453 |   715 |
| Total          |  3,137 |  4,211 | 7,348 |

이미지 특성:

| 항목          | 결과  |
| ----------- | --- |
| Width       | 300 |
| Height      | 300 |
| Channel     | RGB |
| 손상 이미지      | 없음  |
| 지원하지 않는 이미지 | 없음  |

---

## 6. Train·Validation·Test Split

원본 Train Dataset을 Stratified 방식으로 Train·Validation으로 분할했습니다.

설정:

```text
Validation Ratio

0.2
```

```text
Random Seed

42
```

최종 구성:

| Split      | NORMAL | DEFECT | Total |
| ---------- | -----: | -----: | ----: |
| Train      |  2,300 |  3,006 | 5,306 |
| Validation |    575 |    752 | 1,327 |
| Test       |    262 |    453 |   715 |
| Total      |  3,137 |  4,211 | 7,348 |

원본 Test Dataset은 Train·Validation 분할에 포함하지 않았습니다.

Test Dataset은 Best Model을 선택한 이후 최종 일반화 성능 평가에만 사용합니다.

---

## 7. Image Preprocessing

공식 Model 입력 크기:

```text
224 × 224
```

Channel:

```text
RGB

3 Channels
```

ImageNet Normalization:

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

### Train Transform

```text
Resize

↓

Random Horizontal Flip

↓

Small Random Rotation

↓

Tensor 변환

↓

ImageNet Normalization
```

### Validation·Test Transform

```text
Resize

↓

Tensor 변환

↓

ImageNet Normalization
```

Validation·Test에는 Random Augmentation을 적용하지 않습니다.

동일한 입력에 대해 평가 결과가 불필요하게 달라지지 않도록 결정적인 Transform을 사용합니다.

---

## 8. PyTorch Data Pipeline

데이터 흐름:

```text
ImageSample

↓

CastingDefectDataset

↓

Image Transform

↓

PyTorch DataLoader

↓

Image Batch

[B, 3, 224, 224]

+

Label Batch

[B]
```

현재 DataLoader 설정:

| 항목                 |     값 |
| ------------------ | ----: |
| Batch Size         |    32 |
| Num Workers        |     0 |
| Pin Memory         | False |
| Drop Last          | False |
| Persistent Workers | False |
| Random Seed        |    42 |

현재 실행 환경은 Windows·CPU이므로 안정성과 재현성을 우선하여 `num_workers=0`을 사용합니다.

---

## 9. CNN Baseline

CNN Baseline은 ResNet18 Transfer Learning과 비교하기 위한 직접 학습 Model입니다.

Architecture:

```text
Input

[B, 3, 224, 224]

↓

Conv2d

3 → 8

↓

ReLU

↓

MaxPool2d

224 → 112

↓

Conv2d

8 → 16

↓

ReLU

↓

MaxPool2d

112 → 56

↓

Conv2d

16 → 32

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

Model Parameter:

```text
6,065
```

설계 특징:

```text
경량 CPU Baseline

Adaptive Average Pooling

작은 Parameter 수

Raw Logit 출력

Grad-CAM을 고려한 이름 있는 Convolution Layer
```

---

## 10. Why Raw Logits?

CNNBaseline 내부에는 Sigmoid를 넣지 않았습니다.

Model Output:

```text
Raw Logit

Shape:

[B]
```

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

↓

Threshold

↓

Binary Prediction
```

`BCEWithLogitsLoss`는 Sigmoid와 Binary Cross Entropy를 하나의 수치적으로 안정적인 연산으로 처리합니다.

현재 Threshold:

```text
0.5
```

Prediction:

```python
prediction = (
    probability
    >= 0.5
)
```

---

## 11. Loss Function

사용 Loss:

```text
BCEWithLogitsLoss
```

Reduction:

```text
mean
```

현재 CNN Baseline에서는 Positive Class Weight를 사용하지 않습니다.

```text
USE_POSITIVE_CLASS_WEIGHT

False
```

현재 Class 불균형이 존재하지만 극단적이지 않으며, 먼저 기본 Loss로 명확한 Baseline을 확보하는 것을 우선했습니다.

---

## 12. Optimizer

사용 Optimizer:

```text
Adam
```

설정:

| 항목            |     값 |
| ------------- | ----: |
| Learning Rate | 0.001 |
| Weight Decay  |   0.0 |
| Beta 1        |   0.9 |
| Beta 2        | 0.999 |
| Epsilon       |  1e-8 |

Gradient 초기화:

```python
optimizer.zero_grad(
    set_to_none=True
)
```

현재 Baseline에는 다음 기능을 추가하지 않았습니다.

```text
Learning Rate Scheduler

Early Stopping

Gradient Clipping

Mixed Precision

Hyperparameter Search
```

---

## 13. Training Pipeline

Train 흐름:

```text
model.train()

↓

Image Batch

↓

Model Forward

↓

Raw Logit

↓

BCEWithLogitsLoss

↓

zero_grad

↓

Backward

↓

Optimizer Step

↓

Train Loss·Accuracy
```

Validation 흐름:

```text
model.eval()

↓

torch.inference_mode()

↓

Validation Forward

↓

Validation Loss·Accuracy
```

Average Loss는 Batch 평균을 다시 단순 평균하지 않습니다.

```text
Batch Mean Loss

×

Batch Sample Count

↓

전체 합

÷

전체 Sample Count
```

마지막 Batch 크기가 다를 수 있으므로 Sample 가중 평균을 사용합니다.

---

## 14. Best Model Selection

실제 학습 설정:

| 항목                       |               값 |
| ------------------------ | --------------: |
| Epoch                    |               5 |
| Batch Size               |              32 |
| Device                   |             CPU |
| Random Seed              |              42 |
| Classification Threshold |             0.5 |
| Best Model Selection     | Validation Loss |

Best Model 기준:

```text
Lowest Validation Loss
```

Validation Accuracy가 가장 높은 Epoch가 아니라 Validation Loss가 가장 낮은 Epoch를 저장합니다.

---

## 15. CNN Baseline Training Result

실행:

```powershell
python -m scripts.run_day3_cnn_baseline_training
```

실제 학습 결과:

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Best |
| ----: | ---------: | -------------: | --------------: | ------------------: | :--: |
|     1 |   0.619515 |         64.68% |        0.476329 |              82.82% |  Yes |
|     2 |   0.469879 |         79.16% |        0.465499 |              76.94% |  Yes |
|     3 |   0.455239 |         80.21% |        0.492534 |              77.62% |  No  |
|     4 |   0.441022 |         80.49% |        0.479610 |              78.52% |  No  |
|     5 |   0.440413 |         80.63% |        0.509245 |              77.77% |  No  |

Best Epoch:

```text
2
```

Best Validation Loss:

```text
0.465498844753
```

Best Validation Accuracy:

```text
76.94%
```

실제 학습 시간:

```text
646.16 Seconds

약 10.77 Minutes
```

---

## 16. Training Result Analysis

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

Model은 실제 데이터 패턴을 학습했습니다.

Epoch 2 이후에는 Train Loss가 계속 감소했지만 Validation Loss는 다시 증가했습니다.

```text
Train Loss

계속 감소

+

Validation Loss

다시 증가
```

이는 일반화 성능 개선이 정체되고 경미한 Overfitting이 시작된 신호로 해석할 수 있습니다.

Validation Loss 기준 Best Checkpoint를 저장하여 이후 Epoch의 Validation 성능 악화를 최종 Test 평가에 사용하지 않았습니다.

---

## 17. Best Checkpoint

Checkpoint:

```text
models/
└── checkpoints/
    └── cnn_baseline_best.pt
```

현재 Best Checkpoint:

```text
Epoch

2
```

Checkpoint Metadata:

```text
Checkpoint Version

Model Name

Model Module

Loss Function Name

Optimizer Name

Best Epoch

Configured Epoch Count

Classification Threshold

Best Model Selection Metric

Model State

Optimizer State

Train Result

Validation Result
```

Checkpoint Loader 검증:

```text
Checkpoint 존재

.pt·.pth 확장자

Checkpoint Version

필수 Key

Model 이름

Model Module

Epoch 범위

Threshold 범위

Selection Metric

Model State Key

Tensor Type

Tensor Shape

NaN·inf

strict=True Weight Loading
```

---

## 18. Test Evaluation Pipeline

실제 평가 흐름:

```text
새 CNNBaseline 생성

↓

CPU 이동

↓

Best Epoch 2 Checkpoint 복원

↓

Test DataLoader

↓

model.eval()

↓

torch.inference_mode()

↓

Raw Logit

↓

Test Loss

↓

Sigmoid

↓

DEFECT Probability

↓

Threshold 0.5

↓

Prediction

↓

Accuracy

Precision

Recall

F1 Score

Confusion Matrix
```

Evaluation Runner는 Checkpoint를 직접 읽지 않습니다.

역할:

```text
Checkpoint Loader

→ Best Weight 복원
```

```text
Evaluation Runner

→ 전달받은 Model 평가
```

---

## 19. CNN Baseline Test Result

실행:

```powershell
python -m scripts.run_day3_cnn_baseline_evaluation
```

Test Dataset:

| Class  | Count |
| ------ | ----: |
| NORMAL |   262 |
| DEFECT |   453 |
| Total  |   715 |

실제 Test 결과:

| Metric    |   Result |
| --------- | -------: |
| Test Loss | 0.453337 |
| Accuracy  |   76.92% |
| Precision |   82.88% |
| Recall    |   80.13% |
| F1 Score  |   81.48% |

정확한 결과:

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

평가 시간:

```text
8.97 Seconds
```

---

## 20. Confusion Matrix

Confusion Matrix:

```text
tensor(
    [
        [187, 75],
        [90, 363],
    ]
)
```

Matrix 순서:

```text
[
    [TN, FP],
    [FN, TP],
]
```

표:

|               | Predicted NORMAL | Predicted DEFECT |
| ------------- | ---------------: | ---------------: |
| Actual NORMAL |              187 |               75 |
| Actual DEFECT |               90 |              363 |

Confusion Count:

| 항목             | Count |
| -------------- | ----: |
| True Negative  |   187 |
| False Positive |    75 |
| False Negative |    90 |
| True Positive  |   363 |
| Correct        |   550 |
| Incorrect      |   165 |

---

## 21. Metric Interpretation

### Accuracy

```text
550

÷

715

=

76.92%
```

전체 Test 이미지 중 약 76.92%를 정확히 분류했습니다.

---

### Precision

```text
363

÷

438

=

82.88%
```

Model이 DEFECT라고 예측한 이미지 중 약 82.88%가 실제 DEFECT였습니다.

False Positive:

```text
75
```

실제 NORMAL을 DEFECT로 잘못 분류한 경우입니다.

---

### Recall

```text
363

÷

453

=

80.13%
```

실제 DEFECT 중 약 80.13%를 탐지했습니다.

False Negative:

```text
90
```

실제 DEFECT를 NORMAL로 잘못 분류한 경우입니다.

제조 품질 검사에서는 실제 불량이 정상으로 통과할 수 있는 오류이므로 중요한 위험 지표입니다.

False Negative Rate:

```text
90

÷

453

=

19.87%
```

---

### F1 Score

```text
81.48%
```

Precision과 Recall의 균형을 나타냅니다.

현재 Precision과 Recall이 한쪽으로 크게 치우치지 않았습니다.

---

## 22. Validation·Test Comparison

| Metric   | Validation |     Test |
| -------- | ---------: | -------: |
| Loss     |   0.465499 | 0.453337 |
| Accuracy |     76.94% |   76.92% |

Accuracy 차이:

```text
약 0.02 Percentage Point
```

Validation과 Test Accuracy가 거의 같습니다.

현재 결과에서는 다음을 확인할 수 있습니다.

```text
Validation 성능이 Test에서도 유지됨

심한 Overfitting 징후 없음

Best Epoch 선택 정상 동작

안정적인 Test 일반화 성능
```

---

## 23. Majority Class Baseline Comparison

Test Dataset의 DEFECT 비율:

```text
453

÷

715

=

63.36%
```

모든 이미지를 DEFECT로 예측하는 단순 Majority Class Model:

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

CNNBaseline은 단순히 다수 Class만 반복 예측하지 않았습니다.

Prediction 분포:

```text
Predicted NORMAL

277
```

```text
Predicted DEFECT

438
```

---

## 24. Probability Summary

DEFECT Probability:

| 항목      |        값 |
| ------- | -------: |
| Minimum | 0.034242 |
| Maximum | 0.978111 |
| Mean    | 0.539900 |

현재 Threshold:

```text
0.5
```

0에 가까운 NORMAL 예측과 1에 가까운 DEFECT 예측이 모두 존재합니다.

Probability 평균만으로 Model Calibration 성능을 판단하지는 않습니다.

---

## 25. CNN Baseline Summary

| 항목                       | CNNBaseline |
| ------------------------ | ----------: |
| Parameter                |       6,065 |
| Epoch                    |           5 |
| Best Epoch               |           2 |
| Best Validation Loss     |    0.465499 |
| Best Validation Accuracy |      76.94% |
| Test Loss                |    0.453337 |
| Test Accuracy            |      76.92% |
| Test Precision           |      82.88% |
| Test Recall              |      80.13% |
| Test F1                  |      81.48% |
| False Positive           |          75 |
| False Negative           |          90 |

현재 CNNBaseline은 이후 ResNet18 Transfer Learning과 비교할 성능 기준선입니다.

ResNet18 비교에서는 다음 항목을 중점적으로 확인할 예정입니다.

```text
Accuracy 증가

Recall 증가

F1 증가

False Negative 감소

Validation·Test 일반화 유지

Parameter·계산 비용 증가
```

---

## 26. Generated Artifacts

Dataset 분석 결과:

```text
reports/
└── artifacts/
```

CNN Best Checkpoint:

```text
models/
└── checkpoints/
    └── cnn_baseline_best.pt
```

Training History:

```text
reports/
└── artifacts/
    └── day3_cnn_baseline_training_history.json
```

Test Evaluation:

```text
reports/
└── artifacts/
    └── day3_cnn_baseline_test_evaluation.json
```

Test Evaluation JSON에는 715개 이미지별 다음 정보를 저장합니다.

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

향후 다음 분석에 재사용합니다.

```text
False Positive 이미지

False Negative 이미지

오분류 Confidence

오분류 이미지 시각화

Grad-CAM 대상 선택
```

---

## 27. Project Structure

```text
manufacturing-vision-defect-analysis-system/
│
├── data/
│   └── raw/
│
├── models/
│   └── checkpoints/
│       └── cnn_baseline_best.pt
│
├── reports/
│   ├── artifacts/
│   │   ├── day3_cnn_baseline_training_history.json
│   │   └── day3_cnn_baseline_test_evaluation.json
│   │
│   └── day3_cnn_baseline_training_and_evaluation_summary.md
│
├── scripts/
│   ├── run_day3_cnn_baseline_training.py
│   └── run_day3_cnn_baseline_evaluation.py
│
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── dataset_analysis.py
│   │   ├── dataset_config.py
│   │   ├── dataset_split.py
│   │   ├── dataset_visualization.py
│   │   ├── image_dataset.py
│   │   └── image_transforms.py
│   │
│   ├── evaluation/
│   │   ├── classification_metrics.py
│   │   └── evaluation_runner.py
│   │
│   ├── models/
│   │   └── cnn_baseline.py
│   │
│   ├── training/
│   │   ├── checkpoint_loader.py
│   │   ├── epoch_runner.py
│   │   ├── loss_function.py
│   │   ├── optimizer.py
│   │   └── training_pipeline.py
│   │
│   └── reproducibility.py
│
├── tests/
│   ├── test_checkpoint_loader.py
│   ├── test_classification_metrics.py
│   ├── test_cnn_baseline.py
│   ├── test_evaluation_runner.py
│   ├── test_run_day3_cnn_baseline_evaluation.py
│   ├── test_training_pipeline.py
│   └── ...
│
├── README.md
├── requirements.txt
└── pytest.ini
```

---

## 28. Environment

실제 개발·검증 환경:

| 항목          | 값          |
| ----------- | ---------- |
| OS          | Windows    |
| Python      | 3.11.9     |
| PyTorch     | 2.12.0+cpu |
| Torchvision | 0.27.0+cpu |
| CUDA        | False      |
| Device      | CPU        |

현재 실제 실행 장치:

```text
Intel Core i5-1035G7

CPU
```

---

## 29. Installation

가상 환경 생성:

```powershell
python -m venv .venv
```

가상 환경 활성화:

```powershell
.\.venv\Scripts\Activate.ps1
```

Dependency 설치:

```powershell
python -m pip install `
    -r .\requirements.txt
```

Dependency 검증:

```powershell
python -m pip check
```

---

## 30. Run CNN Baseline Training

실제 학습:

```powershell
python -m scripts.run_day3_cnn_baseline_training
```

구성만 검증:

```powershell
python -m scripts.run_day3_cnn_baseline_training `
    --validate-only
```

생성 파일:

```text
models/checkpoints/
cnn_baseline_best.pt
```

```text
reports/artifacts/
day3_cnn_baseline_training_history.json
```

---

## 31. Run CNN Baseline Test Evaluation

구성 검증:

```powershell
python -m scripts.run_day3_cnn_baseline_evaluation `
    --validate-only
```

실제 Test 평가:

```powershell
python -m scripts.run_day3_cnn_baseline_evaluation
```

생성 파일:

```text
reports/artifacts/
day3_cnn_baseline_test_evaluation.json
```

---

## 32. Run Tests

전체 테스트:

```powershell
python -m pytest `
    .\tests `
    -v
```

현재 실제 결과:

```text
1110 passed
```

`1110 passed`는 Parameterized Test의 각 입력 Case가 개별 Test Case로 집계된 결과입니다.

검증 범위:

```text
Dataset Configuration

Dataset Analysis

Dataset Visualization

Train·Validation Split

PyTorch Dataset

Image Transform

DataLoader

Random Seed

CNNBaseline

Loss Function

Optimizer

Train Epoch

Validation Epoch

Training Pipeline

Best Checkpoint

Checkpoint Loader

Evaluation Runner

Classification Metrics

Training Script

Evaluation Script
```

오류 방어:

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

Checkpoint 손상

Checkpoint Key 누락

Checkpoint Version 불일치

Model State 불일치

Metric 내부 불일치

Confusion Matrix 불일치

빈 DataLoader

잘못된 JSON Artifact
```

---

## 33. Design Principles

### Evidence-based Evaluation

Accuracy만 사용하지 않습니다.

```text
Accuracy

Precision

Recall

F1 Score

Confusion Matrix

False Positive

False Negative
```

를 함께 확인합니다.

---

### Validation before Test

```text
Train

→ Model 학습
```

```text
Validation

→ Best Model 선택
```

```text
Test

→ 최종 일반화 성능 평가
```

Test 결과를 보고 Best Epoch를 다시 선택하지 않습니다.

---

### Separation of Responsibilities

```text
CNN Model

→ Raw Logit
```

```text
Loss Module

→ Training Loss
```

```text
Epoch Runner

→ Train·Validation 실행
```

```text
Training Pipeline

→ Epoch 관리·Best Model 저장
```

```text
Checkpoint Loader

→ Best Weight 복원
```

```text
Evaluation Runner

→ Test Forward·Prediction 수집
```

```text
Classification Metrics

→ Accuracy·Precision·Recall·F1
```

각 기능을 분리하여 테스트와 재사용이 쉽도록 구성했습니다.

---

### Reproducibility

```text
Random Seed

42
```

Dataset Split·DataLoader·Model 실행에서 재현 가능한 기준을 유지합니다.

---

## 34. Current Limitations

현재 CNNBaseline의 한계:

```text
작은 Feature Channel

3개 Convolution Block

Pretrained Feature 없음

복잡한 결함 Texture 표현 한계

False Positive 75장

False Negative 90장
```

현재 수행하지 않은 최적화:

```text
Threshold Search

Class Weight

Focal Loss

Learning Rate Scheduler

Early Stopping

Large Hyperparameter Search
```

먼저 ResNet18 Transfer Learning 결과와 비교한 뒤 추가 최적화 필요성을 판단할 예정입니다.

---

## 35. Next Step

다음 단계:

```text
ResNet18 Transfer Learning
```

예정 흐름:

```text
Pretrained ResNet18

↓

Final Fully Connected Layer 교체

↓

Binary Raw Logit

↓

Transfer Learning

↓

Validation Loss 기준 Best Checkpoint

↓

Test Evaluation

↓

CNNBaseline과 성능 비교
```

비교 기준:

```text
Accuracy

Precision

Recall

F1 Score

False Positive

False Negative

Parameter Count

Training Time
```

---

## 36. Portfolio Summary

문제:

제조 이미지의 정상·불량 판정을 자동화하고, Accuracy뿐 아니라 불량 미탐 위험까지 확인할 수 있는 Vision 분류 Pipeline이 필요했습니다.

해결:

PyTorch 기반 Dataset·DataLoader를 구성하고, 6,065 Parameter의 CNNBaseline을 직접 구현했습니다.

`BCEWithLogitsLoss`와 Adam Optimizer를 사용해 실제 CPU 환경에서 학습했으며, Validation Loss 기준 Best Checkpoint를 저장·복원했습니다.

독립 Test Dataset 715장에서 Accuracy·Precision·Recall·F1·Confusion Matrix를 계산하고 715개 Sample별 Prediction 결과를 JSON으로 저장했습니다.

결과:

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
전체 자동 테스트

1110 passed
```

현재 결과는 이후 ResNet18 Transfer Learning 성능을 비교하기 위한 검증된 Baseline으로 사용합니다.

---

## 37. AI Tool Usage

AI 도구는 코드 초안과 테스트 항목 설계에 활용했습니다.

다음 항목은 직접 실행·검증했습니다.

```text
Dataset 구조

Class Mapping

Tensor Shape

Transform

DataLoader

CNN Architecture

Loss 입력·출력

Optimizer 설정

Train·Validation 흐름

Best Checkpoint 정책

Checkpoint Metadata

Test Evaluation

Metric 공식

Confusion Matrix 순서

False Positive·False Negative 의미

실제 CPU 학습

실제 Test 평가

전체 자동 테스트
```

AI가 생성한 코드를 그대로 제출하는 것이 아니라, 각 파일·Class·함수의 역할과 입력·출력·설계 이유를 확인하고 실제 실행 결과를 기준으로 수정·검증·문서화했습니다.
