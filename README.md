# Manufacturing Vision Defect Analysis System

제조 이미지를 활용하여 정상·불량을 분류하고, 모델 학습부터 평가·해석·추론 API·대시보드까지 연결하는 제조 Vision 프로젝트입니다.

한국어 프로젝트명:

**제조 비전 결함 분석 시스템**

---

## Project Goal

이 프로젝트의 목표는 다음과 같습니다.

* 제조 이미지 정상·불량 이진 분류
* 이미지 데이터 분석
* Train·Validation·Test 데이터 분리
* PyTorch Dataset·DataLoader 구현
* 이미지 전처리 및 데이터 증강
* CNN Baseline 구현 및 학습
* ResNet18 전이학습
* CNN Baseline과 ResNet18 성능 비교
* Accuracy·Precision·Recall·F1 Score 평가
* Confusion Matrix 분석
* False Positive·False Negative 분석
* 오분류 이미지 분석
* Grad-CAM 기반 모델 해석
* 이미지 추론 Pipeline
* FastAPI 이미지 추론 API
* Streamlit Dashboard
* 자동 테스트
* 실행 결과·설계 근거·면접 설명 문서화

---

## Dataset

사용 데이터셋:

```text
Casting Product Image Data for Quality Inspection
```

프로젝트 클래스 정의:

```text
ok_front

→ Label 0

→ NORMAL
```

```text
def_front

→ Label 1

→ DEFECT
```

불량을 Positive Class로 정의합니다.

향후 Precision·Recall·F1 Score와 False Negative는 불량 검출 관점에서 해석합니다.

---

## Dataset Analysis Result

Day 1 실제 분석 결과:

| Split | NORMAL | DEFECT | Total |
| ----- | -----: | -----: | ----: |
| Train |  2,875 |  3,758 | 6,633 |
| Test  |    262 |    453 |   715 |
| Total |  3,137 |  4,211 | 7,348 |

전체 클래스 비율:

| Class  | Count |  Ratio |
| ------ | ----: | -----: |
| NORMAL | 3,137 | 42.69% |
| DEFECT | 4,211 | 57.31% |

이미지 속성:

```text
전체 이미지

7,348장
```

```text
이미지 크기

300 × 300
```

```text
이미지 Mode

RGB
```

```text
Channel 수

3
```

```text
지원 이미지

7,348장
```

```text
손상 이미지

0장
```

```text
지원하지 않는 파일

0개
```

---

## Data Split

기존 Test 데이터는 최종 평가용으로 유지하고, 기존 Train 데이터만 새로운 Train과 Validation으로 분리했습니다.

분리 설정:

```python
VALIDATION_RATIO = 0.20

RANDOM_SEED = 42
```

클래스 비율을 유지하기 위해 Stratified Split을 사용했습니다.

| Split         | NORMAL | DEFECT | Total |
| ------------- | -----: | -----: | ----: |
| New Train     |  2,300 |  3,006 | 5,306 |
| Validation    |    575 |    752 | 1,327 |
| Existing Test |    262 |    453 |   715 |

클래스 비율:

| Split          | NORMAL | DEFECT |
| -------------- | -----: | -----: |
| Original Train | 43.34% | 56.66% |
| New Train      | 43.35% | 56.65% |
| Validation     | 43.33% | 56.67% |

검증 결과:

```text
전체 Sample 수 보존

True
```

```text
Train·Validation 이미지 중복

False
```

이미지 파일을 새 폴더로 복사하지 않고 이미지 Path와 Label을 분리하여 원본 데이터를 보존하고 중복 저장을 방지했습니다.

---

## PyTorch Input Pipeline

현재 구현된 전체 데이터 흐름:

```text
Casting 원본 이미지

→ 이미지 Path·Label 수집

→ Stratified Train·Validation Split

→ ImageSample

→ Custom PyTorch Dataset

→ PIL 이미지 로드

→ RGB 3채널 변환

→ Train·Validation·Test Transform

→ Resize

→ Data Augmentation

→ ToTensor

→ ImageNet Normalize

→ DataLoader

→ Image Batch

→ 향후 CNN·ResNet18 입력
```

---

## Custom Dataset

구현 파일:

```text
src/data/image_dataset.py
```

`CastingDefectDataset`의 역할:

```text
ImageSample 저장

→ 이미지 Path 검증

→ Label 검증

→ 이미지 파일 로드

→ RGB 변환

→ Transform 적용

→ Tensor 구조 검증

→ Image Tensor·Label 반환
```

Dataset은 전체 이미지를 생성 시점에 메모리에 올리지 않습니다.

```text
Dataset 생성

→ Image Path·Label만 저장
```

```text
dataset[index]

→ 해당 이미지 한 장 로드

→ Transform

→ Tensor 반환
```

Lazy Loading 방식으로 필요한 이미지만 읽습니다.

개별 Dataset 출력:

```text
Image

[3, 224, 224]

torch.float32
```

```text
Label

0 또는 1
```

---

## Image Transforms

구현 파일:

```text
src/data/image_transforms.py
```

모델 입력 크기:

```python
IMAGE_SIZE = (
    224,
    224,
)
```

ImageNet 정규화 설정:

```python
IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)
```

```python
IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)
```

Train Transform:

```text
Resize

→ RandomHorizontalFlip

→ RandomRotation

→ ToTensor

→ Normalize
```

Validation·Test Transform:

```text
Resize

→ ToTensor

→ Normalize
```

현재 데이터 증강 설정:

```python
HORIZONTAL_FLIP_PROBABILITY = 0.50

ROTATION_DEGREES = 5.0
```

제조 이미지의 결함 특징을 과도하게 훼손하지 않도록 보수적인 증강을 적용했습니다.

현재 제외한 증강:

```text
상하 반전

큰 각도 회전

강한 Crop

강한 색상 변화

강한 Blur

Random Erasing
```

향후 Validation 성능과 오분류 이미지 분석 결과를 기준으로 증강 정책을 조정합니다.

---

## DataLoader

구현 파일:

```text
src/data/data_loader.py
```

현재 설정:

```python
BATCH_SIZE = 32

NUM_WORKERS = 0

PIN_MEMORY = False

DROP_LAST = False

PERSISTENT_WORKERS = False
```

현재 개발 환경:

```text
Operating System

Windows
```

```text
Python

3.11.9
```

```text
CPU

Intel Core i5-1035G7
```

```text
CUDA

False
```

CPU 전용 Windows 환경에서 안정적인 실행과 오류 추적을 우선하여 다음 설정으로 시작했습니다.

```text
num_workers

0
```

```text
pin_memory

False
```

실제 학습 전후에 필요하면 `num_workers=0`과 `num_workers=2`의 처리 속도를 비교합니다.

---

## Batch Result

실제 Dataset 수:

| Split      | Dataset Size |
| ---------- | -----------: |
| Train      |        5,306 |
| Validation |        1,327 |
| Test       |          715 |

실제 Batch 수:

| Split      | Batch Count |
| ---------- | ----------: |
| Train      |         166 |
| Validation |          42 |
| Test       |          23 |

첫 Batch 결과:

```text
Image Shape

(32, 3, 224, 224)
```

```text
Label Shape

(32,)
```

```text
Image dtype

torch.float32
```

```text
Label dtype

torch.int64
```

```text
Image finite

True
```

Train:

```text
RandomSampler
```

Validation:

```text
SequentialSampler
```

Test:

```text
SequentialSampler
```

Train은 Epoch마다 Sample 순서를 섞고, Validation과 Test는 Prediction과 이미지 Path 연결을 유지하기 위해 고정 순서를 사용합니다.

---

## Reproducibility

구현 파일:

```text
src/reproducibility.py
```

기본 Random Seed:

```python
DEFAULT_RANDOM_SEED = 42
```

적용 범위:

```text
Python random

NumPy random

PyTorch CPU random

CUDA 사용 가능 시 CUDA random

Train·Validation Split

DataLoader Shuffle

향후 모델 Weight 초기화

향후 Dropout

Train Data Augmentation
```

현재 환경 결과:

```text
Random Seed

42
```

```text
Deterministic Algorithms

False
```

```text
CUDA Available

False
```

```text
CUDA Seed Applied

False
```

```text
Default Device

cpu
```

난수 재현성 검증:

```text
Python Equal

True
```

```text
NumPy Equal

True
```

```text
PyTorch Equal

True
```

Seed 고정은 같은 환경에서 재현 가능성을 높이지만 운영체제, Python·PyTorch 버전, CPU·GPU, CUDA, 병렬 처리 방식이 달라지면 완전히 동일한 결과를 항상 보장하지는 않습니다.

---

## Current Progress

### Day 1 — Project Setup and Dataset Analysis

* [x] 프로젝트 폴더 생성
* [x] Python 3.11.9 가상환경 생성
* [x] 기본 프로젝트 구조 생성
* [x] `.gitignore` 생성
* [x] `requirements.txt` 생성
* [x] 개발 Library 설치
* [x] 제조 이미지 데이터셋 확정
* [x] Dataset Train·Test 구조 확인
* [x] NORMAL·DEFECT 클래스 정의
* [x] 이미지 개수 분석
* [x] 클래스 비율 분석
* [x] 이미지 확장자 분석
* [x] 이미지 크기 분석
* [x] 이미지 Mode·Channel 분석
* [x] 손상 이미지 검사
* [x] 이미지별 분석 CSV 생성
* [x] Dataset 요약 JSON 생성
* [x] 클래스 분포 그래프 생성
* [x] 정상·불량 Sample 이미지 생성
* [x] Day 1 자동 테스트
* [x] Day 1 보고서

Day 1 테스트 결과:

```text
15 passed
```

---

### Day 2 — PyTorch Dataset, Transform, DataLoader

* [x] CPU·GPU 환경 확인
* [x] CPU 전용 PyTorch 설치
* [x] `torch==2.12.0+cpu`
* [x] `torchvision==0.27.0+cpu`
* [x] `scikit-learn==1.9.0`
* [x] 기존 Train 이미지 수집
* [x] Train·Validation Stratified Split
* [x] 기존 Test 데이터 보존
* [x] ImageSample 구현
* [x] Train·Validation 중복 검사
* [x] Custom PyTorch Dataset 구현
* [x] Lazy Image Loading
* [x] PIL 이미지 로드
* [x] RGB 3채널 변환
* [x] Dataset 입력·출력 검증
* [x] Train Transform 구현
* [x] Validation Transform 구현
* [x] Test Transform 구현
* [x] `224 × 224` Resize
* [x] Random Horizontal Flip
* [x] Small Random Rotation
* [x] ToTensor
* [x] ImageNet Normalize
* [x] Train DataLoader 구현
* [x] Validation DataLoader 구현
* [x] Test DataLoader 구현
* [x] Batch Size 32 구성
* [x] RandomSampler·SequentialSampler 검증
* [x] Image Batch 검증
* [x] Label Batch 검증
* [x] 마지막 작은 Batch 유지
* [x] 전역 Random Seed 설정
* [x] Python·NumPy·PyTorch 난수 재현성 검증
* [x] Day 2 자동 테스트
* [x] Day 2 보고서
* [x] `requirements.txt` 갱신
* [x] 의존성 검사

Day 2 전체 테스트 결과:

```text
131 passed
```

의존성 검사:

```text
No broken requirements found.
```

---

## Project Structure

```text
manufacturing-vision-defect-analysis-system/
├── data/
│   └── raw/
│       └── casting_product_images/
│
├── reports/
│   ├── artifacts/
│   │   ├── day1_class_distribution.png
│   │   ├── day1_dataset_analysis.csv
│   │   ├── day1_dataset_summary.json
│   │   └── day1_sample_images.png
│   │
│   ├── day1_project_setup_and_dataset_analysis.md
│   └── day2_dataset_dataloader_transform.md
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── dataset_analysis.py
│   │   ├── dataset_config.py
│   │   ├── dataset_split.py
│   │   ├── dataset_visualization.py
│   │   ├── image_dataset.py
│   │   └── image_transforms.py
│   │
│   └── reproducibility.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_dataset_config_and_analysis.py
│   ├── test_dataset_split.py
│   ├── test_dataset_summary_and_visualization.py
│   ├── test_image_dataset.py
│   ├── test_image_transforms.py
│   └── test_reproducibility.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Day 1 Artifacts

```text
reports/artifacts/
├── day1_class_distribution.png
├── day1_dataset_analysis.csv
├── day1_dataset_summary.json
└── day1_sample_images.png
```

Day 1 보고서:

```text
reports/
└── day1_project_setup_and_dataset_analysis.md
```

---

## Day 2 Report

```text
reports/
└── day2_dataset_dataloader_transform.md
```

---

## Test

전체 테스트 실행:

```powershell
python -m pytest `
    .\tests `
    -v
```

현재 결과:

```text
131 passed
```

---

## Dependency Check

```powershell
python -m pip check
```

현재 결과:

```text
No broken requirements found.
```

---

## Next Step

### Day 3 — CNN Baseline Model and Training Pipeline

구현 예정:

* CNN Baseline 모델 구조 설계
* Convolution Layer
* ReLU Activation
* Max Pooling
* Adaptive Average Pooling
* Fully Connected Layer
* Forward 흐름
* 모델 입력·출력 Shape 검증
* Binary Classification Logit
* Loss Function 설계
* Optimizer 설계
* 학습·검증 Loop
* Epoch별 Loss 기록
* Epoch별 Accuracy 기록
* Best Model 저장
* CPU 학습 실행
* CNN Baseline 자동 테스트
* Day 3 보고서

---

## Project Status

현재 프로젝트는 단순 예제 코드를 작성하는 단계가 아니라 실제 포트폴리오·지원 제출에 사용할 프로젝트를 직접 구현하고 실행·검증하는 단계입니다.

기존 프로젝트의 검증된 PyTorch 구조와 테스트 방식을 참고하되, Vision 데이터에 필요한 Dataset, 이미지 Transform, Data Augmentation, Image Batch 검증은 새로 구현했습니다.

구현한 각 파일과 함수에 대해 다음 내용을 설명할 수 있도록 진행합니다.

```text
파일 역할

함수 역할

입력

처리 과정

출력

호출 관계

설계 이유

예외 처리

테스트 방법

실무 확장 방향
```

현재 완료 범위:

```text
Day 1

Project Setup

+

Dataset Analysis
```

```text
Day 2

PyTorch Dataset

+

Transform

+

DataLoader

+

Reproducibility
```

다음 단계에서는 현재 완성된 이미지 입력 Pipeline을 CNN Baseline 모델에 연결합니다.
