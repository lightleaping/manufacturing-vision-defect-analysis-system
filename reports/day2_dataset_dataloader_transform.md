# Day 2 — PyTorch Dataset, Transform, DataLoader

## Manufacturing Vision Defect Analysis System

한국어 프로젝트명:

`제조 비전 결함 분석 시스템`

---

## 1. Day 2 목표

Day 1에서는 Casting 이미지 데이터의 구조, 클래스 수, 이미지 크기, 채널, 확장자, 손상 여부를 분석하고 학습 가능한 데이터인지 검증했다.

Day 2에서는 Day 1에서 검증한 이미지 데이터를 PyTorch 모델이 학습할 수 있는 Tensor와 Batch 형태로 변환하는 입력 Pipeline을 구현했다.

Day 2의 전체 목표는 다음과 같다.

```text
PyTorch·torchvision 설치

→ CPU·GPU 실행 환경 확인

→ 기존 Train 데이터의 Train·Validation 분리

→ Random Seed 고정

→ Stratified Split

→ 이미지 Path·Label Sample 구성

→ Custom PyTorch Dataset

→ PIL 이미지 로드

→ RGB 3채널 변환

→ Train Transform

→ Validation Transform

→ Test Transform

→ Resize

→ Data Augmentation

→ ToTensor

→ ImageNet Normalize

→ DataLoader

→ Batch 생성

→ Image·Label Tensor 검증

→ 재현성 설정

→ 자동 테스트
```

Day 2의 최종 출력은 향후 CNN Baseline과 ResNet18이 사용할 다음 Batch다.

```text
Image Batch

[Batch, Channel, Height, Width]

[32, 3, 224, 224]
```

```text
Label Batch

[32]
```

현재 클래스 정의는 Day 1과 동일하게 유지했다.

```text
0

→ NORMAL

→ 정상
```

```text
1

→ DEFECT

→ 불량
```

불량을 Positive Class로 유지하며, 이후 Precision·Recall·F1·False Negative를 불량 검출 관점에서 평가한다.

---

# 2. Day 1 결과가 Day 2에서 사용된 방식

Day 1의 데이터 분석 결과는 단순 문서가 아니라 Day 2 입력 Pipeline 설계의 근거로 사용했다.

Day 1에서 확인한 데이터 구조:

```text
casting_data/
├── train/
│   ├── def_front/
│   └── ok_front/
│
└── test/
    ├── def_front/
    └── ok_front/
```

Day 1에서 정의한 클래스 Mapping:

```python
CLASS_TO_INDEX = {
    "ok_front": 0,
    "def_front": 1,
}
```

Day 2에서는 클래스 Label을 새로 정의하지 않고 기존 `dataset_config.py`의 설정을 단일 기준으로 재사용했다.

이를 통해 파일별 Label 정의가 달라지는 문제를 방지했다.

Day 1 이미지 분석 결과:

```text
전체 이미지

7,348장
```

```text
기존 Train

6,633장
```

```text
기존 Test

715장
```

이미지 속성:

```text
크기

300 × 300
```

```text
Mode

RGB
```

```text
Channel

3
```

```text
손상 이미지

0장
```

Day 2에서는 모든 이미지가 정상적인 RGB 이미지라는 결과를 기반으로 Dataset을 구현했다.

다만 향후 데이터 추가 가능성을 고려해 Dataset에서 모든 이미지를 다시 명시적으로 RGB로 변환했다.

```python
image.convert("RGB")
```

---

# 3. 개발 환경 확인

## 3.1 운영 환경

```text
Operating System

Windows
```

```text
Terminal

PowerShell
```

```text
Python

3.11.9
```

프로젝트 경로:

```text
C:\Users\kflow\Downloads\
manufacturing-vision-defect-analysis-system
```

가상환경:

```text
C:\Users\kflow\Downloads\
manufacturing-vision-defect-analysis-system\
.venv
```

---

## 3.2 CPU 환경

확인 결과:

```text
CPU

Intel(R) Core(TM) i5-1035G7 CPU @ 1.20GHz
```

```text
Physical Cores

4
```

```text
Logical Processors

8
```

---

## 3.3 GPU 환경

확인된 그래픽 장치:

```text
Intel(R) Iris(R) Plus Graphics
```

NVIDIA GPU 확인 명령:

```powershell
nvidia-smi
```

결과:

```text
nvidia-smi 명령을 찾을 수 없음
```

현재 환경에는 PyTorch CUDA 학습에 사용할 NVIDIA GPU가 없는 것으로 판단했다.

따라서 Day 2부터 CPU 전용 PyTorch 환경으로 진행한다.

---

## 3.4 PyTorch 설치 결과

설치 버전:

```text
torch

2.12.0+cpu
```

```text
torchvision

0.27.0+cpu
```

검증 결과:

```text
torch version

2.12.0+cpu
```

```text
torchvision version

0.27.0+cpu
```

```text
CUDA available

False
```

```text
CUDA device count

0
```

```text
selected device

cpu
```

`+cpu`는 CUDA Package가 아닌 CPU 전용 PyTorch Build를 의미한다.

---

## 3.5 기본 Tensor 검증

생성한 이미지 Tensor:

```text
torch.Size([4, 3, 224, 224])
```

Label Tensor:

```text
torch.Size([4])
```

dtype:

```text
Image

torch.float32
```

```text
Label

torch.int64
```

Device:

```text
cpu
```

PyTorch 이미지 Batch의 기본 차원 순서는 다음과 같다.

```text
N × C × H × W
```

각 기호의 의미:

```text
N

→ Batch Size
```

```text
C

→ Channel
```

```text
H

→ Height
```

```text
W

→ Width
```

---

# 4. Train·Validation 분리 설계

## 4.1 기존 Test 데이터 보존

기존 데이터는 Train과 Test로 이미 구분되어 있다.

Day 2에서는 기존 Train만 새로운 Train과 Validation으로 분리했다.

```text
기존 Train

6,633장

↓

새 Train

+

Validation
```

기존 Test:

```text
715장

→ 분리에 사용하지 않음

→ 최종 모델 평가용으로 유지
```

각 데이터의 역할:

```text
Train

→ 모델 Parameter 학습
```

```text
Validation

→ Epoch별 모델 상태 확인

→ 모델 선택

→ 과적합 확인
```

```text
Test

→ 최종 모델 평가

→ Accuracy

→ Precision

→ Recall

→ F1 Score

→ Confusion Matrix

→ 오분류 이미지 분석
```

Test를 학습 과정에서 반복적으로 확인하며 모델 구조나 Hyperparameter를 변경하면 Test 결과에 간접적으로 맞춰질 수 있다.

따라서 기존 Test는 최종 평가 전용으로 분리했다.

---

## 4.2 Validation 비율

설정:

```python
VALIDATION_RATIO = 0.20
```

기존 Train 데이터의 80%를 새로운 Train으로 사용하고 20%를 Validation으로 사용한다.

```text
새 Train

80%
```

```text
Validation

20%
```

---

## 4.3 Stratified Split

기존 Train 클래스 수:

```text
NORMAL

2,875장
```

```text
DEFECT

3,758장
```

기존 Train 클래스 비율:

```text
NORMAL

43.34%
```

```text
DEFECT

56.66%
```

단순 Random Split 대신 클래스 비율을 유지하는 Stratified Split을 사용했다.

핵심 설정:

```python
train_test_split(
    samples,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=labels,
)
```

분리 결과:

| Split      | NORMAL | DEFECT |    전체 |
| ---------- | -----: | -----: | ----: |
| 기존 Train   |  2,875 |  3,758 | 6,633 |
| 새 Train    |  2,300 |  3,006 | 5,306 |
| Validation |    575 |    752 | 1,327 |

클래스 비율:

```text
기존 Train

NORMAL

43.34%

DEFECT

56.66%
```

```text
새 Train

NORMAL

43.35%

DEFECT

56.65%
```

```text
Validation

NORMAL

43.33%

DEFECT

56.67%
```

Stratified Split을 통해 Train과 Validation의 클래스 비율이 거의 동일하게 유지됐다.

---

## 4.4 이미지 파일을 복사하지 않는 분리 구조

Train·Validation 분리를 위해 이미지 파일을 새로운 폴더에 복사하지 않았다.

사용하지 않은 구조:

```text
data/processed/
├── train/
└── validation/
```

대신 이미지 Path와 Label만 분리했다.

```python
ImageSample(
    image_path=Path(
        ".../ok_front/image.jpeg"
    ),
    label=0,
)
```

장점:

```text
원본 이미지 보존

디스크 공간 절약

중복 파일 생성 방지

Seed 변경 가능

분리 결과 재현 가능
```

---

# 5. `src/data/dataset_split.py`

## 5.1 책임

```text
이미지 클래스 디렉터리 탐색

→ 이미지 Path 수집

→ 클래스 Label 연결

→ ImageSample 생성

→ Stratified Split

→ Train·Validation 중복 검증

→ 클래스 분포 계산

→ 분리 결과 출력
```

---

## 5.2 주요 설정

```python
VALIDATION_RATIO = 0.20
```

```python
RANDOM_SEED = 42
```

---

## 5.3 `ImageSample`

```python
@dataclass(frozen=True)
class ImageSample:
    image_path: Path
    label: int
```

이미지 한 장의 경로와 Label을 저장한다.

`frozen=True`를 사용해 생성 이후 Path와 Label이 실수로 변경되지 않도록 했다.

---

## 5.4 주요 함수

```python
normalize_image_extensions()
```

역할:

```text
확장자 공백 제거

→ 소문자 변환

→ 점 추가

→ 중복 제거
```

---

```python
collect_image_samples()
```

역할:

```text
데이터 Root 확인

→ 클래스 폴더 확인

→ 지원 이미지 탐색

→ Path·Label 연결

→ ImageSample 목록 반환
```

---

```python
validate_split_arguments()
```

검증:

```text
Sample 존재 여부

Validation 비율

클래스 수

클래스별 최소 Sample 수

Train·Validation 최소 크기
```

---

```python
split_train_validation_samples()
```

역할:

```text
Label 추출

→ Stratified Split

→ Path 정렬

→ 중복 검사

→ 전체 수 보존 검사
```

---

```python
validate_no_sample_overlap()
```

역할:

```text
Train 이미지 Path

∩

Validation 이미지 Path

=

빈 집합
```

동일 이미지가 Train과 Validation에 동시에 존재하면 데이터 누수가 발생할 수 있으므로 예외 처리한다.

---

## 5.5 실제 실행 결과

```text
ORIGINAL TRAIN

6,633장
```

```text
NEW TRAIN

5,306장
```

```text
VALIDATION

1,327장
```

검증:

```text
sample count preserved

True
```

```text
train-validation overlap

False
```

---

# 6. PyTorch Dataset

## 6.1 Dataset 역할

Dataset은 개별 Sample을 어떻게 읽고 반환할지 정의한다.

현재 프로젝트에서 Sample 하나:

```text
이미지 한 장

+

Label 하나
```

Dataset 호출:

```python
image_tensor, label = dataset[0]
```

반환:

```text
Image Tensor

[3, 224, 224]
```

```text
Label

0 또는 1
```

---

## 6.2 Dataset과 DataLoader 차이

Dataset:

```text
이미지 한 장을 어떻게 읽을 것인가?
```

DataLoader:

```text
여러 Sample을 어떻게 Batch로 묶을 것인가?
```

Dataset 결과:

```text
[3, 224, 224]
```

DataLoader 결과:

```text
[32, 3, 224, 224]
```

---

# 7. `src/data/image_dataset.py`

## 7.1 책임

```text
ImageSample 저장

→ 이미지 Path 검증

→ Label 검증

→ PIL 이미지 로드

→ RGB 변환

→ Transform 실행

→ Tensor 검증

→ Image Tensor·Label 반환
```

---

## 7.2 `__init__()`

Dataset 객체 생성 시 실행된다.

처리:

```text
Sample 존재 확인

Transform 호출 가능 여부 확인

허용 Label 확인

ImageSample 구조 확인

Path 존재 확인

중복 Path 확인
```

중요:

`__init__()`에서는 전체 이미지를 메모리에 읽지 않는다.

다음만 보관한다.

```text
Image Path

Label

Transform
```

---

## 7.3 `__len__()`

Dataset의 Sample 수를 반환한다.

현재 결과:

```text
Train

5,306
```

```text
Validation

1,327
```

```text
Test

715
```

---

## 7.4 `__getitem__()`

처리 순서:

```text
Index 확인

→ ImageSample 조회

→ Image.open()

→ RGB 변환

→ Transform

→ Tensor 타입 확인

→ C × H × W 확인

→ RGB 3채널 확인

→ Floating Point 확인

→ Tensor·Label 반환
```

---

## 7.5 Lazy Loading

전체 이미지를 Dataset 생성 시 메모리에 올리지 않는다.

```text
Dataset 생성

→ Path·Label만 저장
```

실제 이미지 요청:

```text
dataset[index]

→ 해당 이미지 한 장 로드
```

이 구조는 이미지 전체를 메모리에 저장하는 것보다 메모리를 효율적으로 사용할 수 있다.

---

## 7.6 RGB 변환

현재 모든 데이터는 Day 1에서 RGB로 확인됐다.

그래도 다음 코드를 유지했다.

```python
image.convert("RGB")
```

이유:

향후 다음 이미지가 추가되더라도 모델 입력 Channel을 3개로 유지한다.

```text
Grayscale

RGBA

Palette Image
```

---

## 7.7 Dataset Smoke Test

실제 결과:

```text
Dataset Size

6,633
```

```text
Image Shape

torch.Size([3, 300, 300])
```

```text
Image dtype

torch.float32
```

```text
Image Minimum

0.0
```

```text
Image Maximum

1.0
```

```text
Label

0
```

현재 Smoke Test는 정식 Transform 구현 전 `ToTensor()`만 사용했으므로 원본 크기인 `300 × 300`이 유지됐다.

---

# 8. 이미지 Transform

## 8.1 Transform 전체 흐름

Train:

```text
PIL RGB Image

→ Resize

→ RandomHorizontalFlip

→ RandomRotation

→ ToTensor

→ Normalize

→ Tensor [3, 224, 224]
```

Validation·Test:

```text
PIL RGB Image

→ Resize

→ ToTensor

→ Normalize

→ Tensor [3, 224, 224]
```

---

## 8.2 Resize

원본:

```text
300 × 300
```

모델 입력:

```text
224 × 224
```

설정:

```python
IMAGE_SIZE = (
    224,
    224,
)
```

이유:

```text
CNN Baseline과 ResNet18 입력 통일

ResNet18 전이학습과 호환

원본 대비 연산량 감소

두 모델의 공정한 비교
```

---

## 8.3 ToTensor

변환 전:

```text
Height × Width × Channel

H × W × C
```

변환 후:

```text
Channel × Height × Width

C × H × W
```

일반적인 8-bit 이미지 값:

```text
0 ~ 255
```

Tensor:

```text
0.0 ~ 1.0
```

dtype:

```text
torch.float32
```

---

## 8.4 Normalize

ImageNet 평균:

```python
IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)
```

ImageNet 표준편차:

```python
IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)
```

계산:

```text
normalized_pixel

=

(pixel - mean)

/ std
```

Normalize 이후 값은 더 이상 `0.0~1.0`으로 제한되지 않는다.

실제 결과:

```text
Train Minimum

-2.117904
```

```text
Train Maximum

2.535425 이상
```

음수 또는 `1.0`보다 큰 값은 정상이다.

---

# 9. Train Data Augmentation

## 9.1 적용한 증강

좌우 반전:

```python
HORIZONTAL_FLIP_PROBABILITY = 0.50
```

작은 회전:

```python
ROTATION_DEGREES = 5.0
```

회전 범위:

```text
약 -5도 ~ +5도
```

---

## 9.2 증강 목적

모델이 학습 이미지의 정확한 위치나 방향만 암기하지 않고 작은 위치·방향 변화에서도 정상·불량 특징을 인식하도록 돕는다.

---

## 9.3 보수적인 증강 설정

현재 제외:

```text
상하 반전

큰 각도 회전

강한 Crop

강한 Color Jitter

강한 Blur

Random Erasing
```

이유:

```text
결함 영역 손실 가능성

실제 제조 분포와 다른 이미지 생성 가능성

인위적인 Artifact 발생 가능성

Baseline 단계의 불필요한 복잡도 방지
```

향후 Validation 성능과 오분류 이미지를 확인한 뒤 증강 정책을 다시 검토한다.

---

## 9.4 Train에만 랜덤 증강 적용

Train:

```text
RandomHorizontalFlip

RandomRotation
```

Validation·Test:

```text
랜덤 증강 없음
```

Validation과 Test는 같은 이미지에 항상 같은 Tensor를 생성해야 모델 성능을 일관된 조건에서 비교할 수 있다.

---

# 10. `src/data/image_transforms.py`

## 10.1 주요 설정

```python
IMAGE_SIZE = (
    224,
    224,
)
```

```python
HORIZONTAL_FLIP_PROBABILITY = 0.50
```

```python
ROTATION_DEGREES = 5.0
```

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

---

## 10.2 주요 함수

```python
create_train_transform()
```

반환:

```text
Resize

→ RandomHorizontalFlip

→ RandomRotation

→ ToTensor

→ Normalize
```

---

```python
create_evaluation_transform()
```

반환:

```text
Resize

→ ToTensor

→ Normalize
```

---

```python
create_validation_transform()
```

Validation 역할을 명확하게 표현하기 위한 별도 함수다.

---

```python
create_test_transform()
```

Test 역할을 명확하게 표현하기 위한 별도 함수다.

현재 Validation과 Test의 실제 Transform 구성은 동일하다.

---

## 10.3 Transform 실제 결과

```text
Train Shape

torch.Size([3, 224, 224])
```

```text
Validation Shape

torch.Size([3, 224, 224])
```

```text
Test Shape

torch.Size([3, 224, 224])
```

```text
Train dtype

torch.float32
```

```text
Validation dtype

torch.float32
```

```text
Train finite

True
```

```text
Validation finite

True
```

평가 Transform 재현성:

```text
validation repeat

True
```

Validation·Test 결과:

```text
validation-test equal

True
```

---

# 11. DataLoader

## 11.1 DataLoader 역할

Dataset이 반환하는 개별 Sample을 Batch로 묶어 모델에 공급한다.

Dataset:

```text
Image

[3, 224, 224]
```

DataLoader:

```text
Images

[32, 3, 224, 224]
```

Label:

```text
[32]
```

---

## 11.2 Batch

Batch는 한 번의 모델 Forward 연산에 함께 전달되는 Sample 묶음이다.

현재:

```python
BATCH_SIZE = 32
```

Train:

```text
이미지 32장

→ 모델 예측

→ Loss 계산

→ 역전파

→ Weight 변경
```

이 과정을 전체 Train Dataset에 반복한다.

---

## 11.3 Shuffle

Train:

```python
shuffle=True
```

Validation:

```python
shuffle=False
```

Test:

```python
shuffle=False
```

Train은 매 Epoch 다양한 Batch 구성을 만들기 위해 순서를 섞는다.

Validation과 Test는 Prediction과 이미지 Path를 일정한 순서로 연결하기 위해 고정 순서를 사용한다.

---

## 11.4 Sampler

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

실제 실행에서 의도한 Sampler가 적용된 것을 확인했다.

---

## 11.5 `num_workers`

현재:

```python
NUM_WORKERS = 0
```

의미:

```text
별도 데이터 로딩 Process 없음

Main Process에서 이미지 로드·Transform 수행
```

현재 환경:

```text
Windows

CPU 전용

4 Core

8 Logical Processor
```

Day 2에서는 안정적인 실행과 오류 추적을 우선해 `0`을 사용했다.

향후 학습 속도를 측정한 뒤 다음 비교를 검토할 수 있다.

```text
num_workers=0

vs

num_workers=2
```

Worker 수가 많다고 항상 빠른 것은 아니며 Process 생성 비용과 추가 메모리 사용도 고려해야 한다.

---

## 11.6 `pin_memory`

현재:

```python
PIN_MEMORY = False
```

Pinned Memory는 주로 CPU Tensor를 CUDA GPU로 전송할 때 사용한다.

현재 환경:

```text
CUDA available

False
```

따라서 `False`를 적용했다.

향후 CUDA 환경에서는 실제 성능을 측정한 뒤 `True` 사용을 검토할 수 있다.

---

## 11.7 `drop_last`

현재:

```python
DROP_LAST = False
```

Train:

```text
5,306장

32 × 165

=

5,280장
```

남은 이미지:

```text
26장
```

마지막 26장도 학습에 사용한다.

Validation 마지막 Batch:

```text
15장
```

Test 마지막 Batch:

```text
11장
```

현재는 전체 데이터를 사용하기 위해 마지막 작은 Batch를 제거하지 않는다.

---

## 11.8 `persistent_workers`

현재:

```python
PERSISTENT_WORKERS = False
```

현재 `num_workers=0`이므로 유지할 별도 Worker Process가 없다.

---

# 12. `src/data/data_loader.py`

## 12.1 책임

```text
기존 Train Sample 수집

→ Train·Validation 분리

→ 기존 Test Sample 수집

→ 역할별 Dataset 생성

→ 역할별 Transform 연결

→ 역할별 DataLoader 생성

→ Sampler 검증

→ Batch Tensor 검증

→ 실행 결과 출력
```

---

## 12.2 주요 설정

```python
BATCH_SIZE = 32
```

```python
NUM_WORKERS = 0
```

```python
PIN_MEMORY = False
```

```python
DROP_LAST = False
```

```python
PERSISTENT_WORKERS = False
```

---

## 12.3 `VisionDataLoaders`

```python
@dataclass(frozen=True)
class VisionDataLoaders:
    train_dataset
    validation_dataset
    test_dataset

    train_loader
    validation_loader
    test_loader
```

Train·Validation·Test Dataset과 DataLoader를 하나의 객체로 관리한다.

---

## 12.4 주요 함수

```python
validate_data_loader_arguments()
```

검증:

```text
Dataset 타입

Dataset 크기

Batch Size

Worker 수

Boolean 설정

persistent_workers 조건
```

---

```python
create_data_loader()
```

역할:

```text
독립 PyTorch Generator 생성

→ Random Seed 설정

→ DataLoader 생성
```

---

```python
create_vision_data_loaders()
```

전체 호출:

```text
TRAIN_ROOT

→ 이미지 수집

→ Train·Validation 분리

→ Train Dataset

→ Validation Dataset

→ Train DataLoader

→ Validation DataLoader
```

```text
TEST_ROOT

→ 이미지 수집

→ Test Dataset

→ Test DataLoader
```

---

```python
validate_image_label_batch()
```

검증:

```text
Image Tensor 여부

Label Tensor 여부

Image Batch 4차원

개별 이미지 [3, 224, 224]

Image float dtype

NaN·inf 없음

Label 1차원

Label torch.int64

Image·Label 수 일치

Label 0·1 범위
```

---

## 12.5 실제 Dataset 수

```text
Train

5,306장
```

```text
Validation

1,327장
```

```text
Test

715장
```

---

## 12.6 실제 Batch 수

```text
Train

166 Batch
```

```text
Validation

42 Batch
```

```text
Test

23 Batch
```

---

## 12.7 실제 첫 Batch

Train:

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

---

Validation:

```text
Image Shape

(32, 3, 224, 224)
```

```text
Label Shape

(32,)
```

```text
Image finite

True
```

---

Test:

```text
Image Shape

(32, 3, 224, 224)
```

```text
Label Shape

(32,)
```

```text
Image finite

True
```

---

# 13. Validation 첫 Batch가 DEFECT만 포함된 이유

Validation 첫 Batch:

```text
Label

1, 1, 1, ...
```

전체 Validation은 다음 클래스 분포를 가진다.

```text
NORMAL

575장
```

```text
DEFECT

752장
```

Validation 전체가 DEFECT인 것은 아니다.

분리 후 Path를 문자열 기준으로 정렬했으며:

```text
def_front

→ 먼저 정렬
```

```text
ok_front

→ 이후 정렬
```

Validation은 `shuffle=False`이므로 첫 Batch에 DEFECT Sample이 연속으로 나타났다.

평가 지표는 첫 Batch가 아니라 전체 42 Batch를 사용하여 계산한다.

---

# 14. Test 첫 Batch가 NORMAL만 포함된 이유

Test Sample은 Label 순서로 수집한다.

```text
ok_front

→ Label 0

→ 먼저 수집
```

```text
def_front

→ Label 1

→ 이후 수집
```

Test는 `shuffle=False`이므로 첫 Batch가 NORMAL로 구성됐다.

최종 평가는 전체 Test 715장을 사용한다.

---

# 15. Random Seed와 재현성

## 15.1 기본 Seed

```python
DEFAULT_RANDOM_SEED = 42
```

---

## 15.2 적용 범위

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

---

## 15.3 `src/reproducibility.py`

책임:

```text
Seed 검증

→ Python Seed

→ NumPy Seed

→ PyTorch CPU Seed

→ CUDA Seed

→ 결정적 알고리즘 옵션

→ 현재 Device 반환

→ 설정 결과 출력
```

---

## 15.4 실제 설정

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

---

## 15.5 난수 재현성 결과

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

PyTorch 결과:

```text
first

tensor([0.8823, 0.9150, 0.3829])
```

```text
second

tensor([0.8823, 0.9150, 0.3829])
```

---

## 15.6 Seed 고정의 한계

Seed를 고정하면 같은 환경에서 실험 재현 가능성을 높일 수 있다.

하지만 다음 조건이 다르면 완전히 동일한 결과를 항상 보장할 수는 없다.

```text
운영체제

Python 버전

PyTorch 버전

torchvision 버전

CPU·GPU

CUDA 버전

병렬 처리

사용 연산

Library 구현
```

---

# 16. Day 2 생성 파일

```text
src/data/
├── dataset_split.py
├── image_dataset.py
├── image_transforms.py
└── data_loader.py
```

```text
src/
└── reproducibility.py
```

테스트:

```text
tests/
├── test_dataset_split.py
├── test_image_dataset.py
├── test_image_transforms.py
├── test_data_loader.py
└── test_reproducibility.py
```

문서:

```text
reports/
└── day2_dataset_dataloader_transform.md
```

수정:

```text
requirements.txt
```

---

# 17. Day 2 테스트 결과

## 17.1 Train·Validation 분리

```text
17 passed
```

검증:

```text
확장자 정규화

이미지 Path·Label 연결

지원하지 않는 파일 제외

데이터 Root 누락

클래스 폴더 누락

빈 클래스 폴더

Validation 비율

클래스 수

Stratified Split

전체 수 보존

Random Seed 재현성

Train·Validation 중복

실제 이미지 수

실제 클래스 수

실제 분리 결과
```

---

## 17.2 Custom Dataset

```text
20 passed
```

검증:

```text
빈 Sample

Transform 타입

허용 Label

ImageSample 타입

Boolean Label

정의되지 않은 Label

존재하지 않는 이미지

디렉터리 Path

중복 Path

Dataset 길이

읽기 전용 Property

Grayscale → RGB

Index 타입

Index 범위

손상 이미지

Transform 반환 타입

Tensor 차원

Channel 수

Floating dtype

실제 Casting 이미지
```

---

## 17.3 Image Transform

```text
12 passed
```

검증:

```text
IMAGE_SIZE

ImageNet Mean

ImageNet Standard Deviation

Train Transform 순서

Train 랜덤 증강

평가 Transform 구성

평가 Transform 재현성

Validation·Test 동일성

Normalize 계산

Train Tensor Shape

Random Seed 재현성

실제 Casting 이미지
```

---

## 17.4 DataLoader

```text
44 passed
```

검증:

```text
기본 설정

Dataset 타입

빈 Dataset

Batch Size

num_workers

Boolean 옵션

persistent_workers

Shuffle 타입

Random Seed 타입

RandomSampler

SequentialSampler

마지막 Batch 유지

drop_last

Shuffle 재현성

Image Batch

Label Batch

Tensor 차원

Channel·크기

dtype

NaN·inf

Label 범위

실제 Dataset 수

실제 Batch 수

실제 Sampler

실제 첫 Batch

평가 Batch 재현성
```

---

## 17.5 재현성

```text
23 passed
```

검증:

```text
기본 Seed

Seed 경계값

Seed 타입

Seed 범위

deterministic option

현재 Device

설정 객체

Python 난수

NumPy 난수

PyTorch 난수

다른 Seed

결정적 알고리즘

CUDA 분기

설정 출력

불변 객체

main()
```

---

## 17.6 전체 테스트

```text
131 passed
```

테스트 구성:

| 테스트 영역           |    통과 수 |
| ---------------- | ------: |
| Day 1 데이터 분석·시각화 |      15 |
| Dataset Split    |      17 |
| Custom Dataset   |      20 |
| Image Transform  |      12 |
| DataLoader       |      44 |
| Reproducibility  |      23 |
| **전체**           | **131** |

---

# 18. requirements.txt 갱신

추가된 직접 의존성:

```text
torch==2.12.0+cpu
```

```text
torchvision==0.27.0+cpu
```

```text
scikit-learn==1.9.0
```

CPU PyTorch Package 저장소:

```text
--extra-index-url

https://download.pytorch.org/whl/cpu
```

현재 설치 검증:

```text
Requirement already satisfied
```

의존성 검사:

```text
No broken requirements found.
```

---

# 19. 기존 프로젝트 참고·재사용 정리

## [기존 코드 참고]

참고 프로젝트:

```text
Manufacturing AI Quality Agent
```

참고 대상:

```text
PyTorch Dataset 구조

DataLoader 구조

Random Seed

입력 검증

Tensor 검증

테스트 구조

환경 설정
```

---

## [그대로 재사용]

다음 PyTorch 표준 구조를 사용했다.

```text
Dataset

DataLoader

Tensor

torch.Generator

pytest
```

Day 1 설정:

```text
TRAIN_ROOT

TEST_ROOT

CLASS_TO_INDEX

INDEX_TO_CLASS_NAME

SUPPORTED_IMAGE_EXTENSIONS
```

도 그대로 재사용했다.

---

## [수정하여 재사용]

기존 프로젝트 입력:

```text
설비 수치 Feature

→ Tensor
```

현재 Vision 프로젝트 입력:

```text
이미지 Path

→ PIL Image

→ RGB

→ Transform

→ Image Tensor
```

기존 수치 데이터 전처리:

```text
StandardScaler
```

현재 이미지 전처리:

```text
Resize

Data Augmentation

ToTensor

ImageNet Normalize
```

---

## [신규 구현]

Vision 프로젝트에 새로 구현:

```text
ImageSample

이미지 Path 수집

Stratified 이미지 분리

이미지 Path 중복 검증

PIL 이미지 로드

RGB 변환

Image Transform

이미지 Data Augmentation

Image Tensor 검증

Vision DataLoader

Image Batch 검증

Train·Validation·Test Dataset 통합
```

---

## [변경 이유]

기존 프로젝트는 표 형태의 설비 데이터를 사용했다.

현재 프로젝트는 이미지의 공간 정보와 RGB Channel을 사용하므로 다음 처리가 새로 필요했다.

```text
이미지 파일 관리

Channel 통일

이미지 크기 통일

공간 Data Augmentation

Channel별 Normalize

4차원 Image Batch
```

---

# 20. 실무 확장 고려사항

현재 구현 범위는 포트폴리오 프로젝트의 안정적인 Baseline 입력 Pipeline이다.

향후 실제 제조 환경에서는 다음 항목을 추가 검토할 수 있다.

```text
제품 ID 기반 Group Split

생산 Batch 기반 Group Split

설비 ID 기반 Group Split

생산 라인 기반 Group Split

촬영 시간 기반 Temporal Split

Manifest CSV·Parquet

Dataset Version 관리

중복 이미지 Hash 검사

Near-Duplicate 이미지 검사

DataLoader Worker Benchmark

Batch Size Benchmark

증강 Ablation Test

Dataset Drift 감지
```

현재 데이터에는 제품 ID·생산 Batch·설비 ID Metadata가 확인되지 않았으므로 클래스 Label 기반 Stratified Split을 사용했다.

---

# 21. 면접 질문과 답변

## Q1. Dataset과 DataLoader의 차이는 무엇인가요?

Dataset은 개별 Sample을 어떻게 읽고 반환할지 정의합니다.

현재 프로젝트에서는 이미지 Path를 받아 Pillow로 이미지를 읽고 RGB 변환과 Transform을 적용한 뒤 이미지 Tensor와 Label을 반환합니다.

DataLoader는 Dataset의 여러 Sample을 Batch로 묶고 Shuffle, Worker 수, 마지막 Batch 처리 등을 관리합니다.

현재 Dataset은 한 장의 `[3, 224, 224]` Tensor를 반환하고 DataLoader는 `[32, 3, 224, 224]` Batch를 생성합니다.

---

## Q2. `__init__()`, `__len__()`, `__getitem__()`의 역할은 무엇인가요?

`__init__()`은 Image Path, Label, Transform을 저장하고 입력 구조를 검증합니다.

`__len__()`은 Dataset의 전체 Sample 수를 반환합니다.

`__getitem__()`은 특정 Index의 이미지를 실제로 읽고 RGB 변환과 Transform을 적용한 뒤 Tensor와 Label을 반환합니다.

현재 구현은 Dataset 생성 시 전체 이미지를 메모리에 읽지 않고 필요한 이미지 한 장만 읽는 Lazy Loading 방식을 사용했습니다.

---

## Q3. 왜 기존 Train을 Train과 Validation으로 다시 분리했나요?

기존 데이터는 Train과 Test만 제공했습니다.

Test를 학습 중 모델 선택에 반복 사용하면 Test 결과에 간접적으로 맞춰질 수 있습니다.

따라서 기존 Train을 새로운 Train과 Validation으로 나누고 기존 Test는 최종 평가용으로 보존했습니다.

---

## Q4. 왜 Stratified Split을 사용했나요?

정상과 불량의 클래스 비율을 Train과 Validation에서 유사하게 유지하기 위해 사용했습니다.

기존 Train의 NORMAL 비율은 약 43.34%, DEFECT는 약 56.66%였으며 분리 후에도 두 Split에서 거의 같은 비율이 유지됐습니다.

---

## Q5. 왜 Train에만 랜덤 증강을 적용했나요?

Train에서는 작은 위치나 방향 변화에도 결함 특징을 인식하도록 일반화 성능을 높이기 위해 랜덤 증강을 사용했습니다.

Validation과 Test는 동일한 조건에서 성능을 측정해야 하므로 랜덤 증강을 제외했습니다.

---

## Q6. 어떤 증강을 적용했나요?

50% 확률의 좌우 반전과 약 `-5도~+5도`의 작은 회전을 적용했습니다.

제조 이미지에서는 강한 증강이 실제 결함 영역을 훼손하거나 현실에 없는 패턴을 만들 수 있으므로 보수적으로 설정했습니다.

---

## Q7. 왜 `224 × 224`를 사용했나요?

CNN Baseline과 ResNet18이 동일한 입력 크기를 사용하도록 했습니다.

ResNet18 전이학습 구조와 호환되며 원본 `300 × 300`보다 연산량을 줄일 수 있습니다.

---

## Q8. 왜 ImageNet Normalize를 사용했나요?

이후 ImageNet 사전학습 ResNet18을 사용할 예정이므로 사전학습 입력 분포와 맞추기 위해 사용했습니다.

CNN Baseline에도 같은 입력 전처리를 적용하여 두 모델의 비교 조건을 통일했습니다.

---

## Q9. 왜 `num_workers=0`을 사용했나요?

현재 환경은 Windows CPU 환경이며 Day 2에서는 안정적인 실행과 오류 추적을 우선했습니다.

Worker 수가 증가하면 항상 빨라지는 것은 아니므로 실제 학습 전후에 `0`과 `2`를 Benchmark한 뒤 필요하면 조정할 수 있습니다.

---

## Q10. 왜 `pin_memory=False`인가요?

Pinned Memory는 주로 CPU Tensor를 CUDA GPU로 전송할 때 사용합니다.

현재 환경에는 CUDA GPU가 없으므로 `False`를 사용했습니다.

---

## Q11. 왜 `drop_last=False`인가요?

마지막 Batch가 Batch Size보다 작더라도 전체 이미지를 학습과 평가에 사용하기 위해서입니다.

현재 Train 마지막 Batch의 26장도 제외하지 않고 사용합니다.

---

## Q12. Seed를 왜 고정했나요?

Train·Validation Split, DataLoader Shuffle, 모델 Weight 초기화, Dropout, Random Transform은 난수의 영향을 받을 수 있습니다.

같은 조건의 실험을 비교하기 위해 기본 Seed를 `42`로 통일했습니다.

다만 Library 버전과 하드웨어가 달라지면 완전히 동일한 결과를 항상 보장할 수는 없습니다.

---

# 22. Day 2 최종 결론

Day 2에서는 Day 1에서 검증한 Casting 이미지 데이터를 PyTorch 모델이 사용할 수 있는 Dataset·Transform·DataLoader Pipeline으로 연결했다.

완성된 흐름:

```text
Casting 이미지

→ Path·Label 수집

→ Stratified Train·Validation Split

→ Custom Dataset

→ PIL 이미지 로드

→ RGB 변환

→ 역할별 Transform

→ Resize

→ Train Data Augmentation

→ ToTensor

→ ImageNet Normalize

→ DataLoader

→ Image Batch

[32, 3, 224, 224]

→ Label Batch

[32]
```

실제 데이터:

```text
Train

5,306장
```

```text
Validation

1,327장
```

```text
Test

715장
```

최종 검증:

```text
전체 테스트

131 passed
```

```text
의존성 검사

No broken requirements found.
```

Day 2 구현 목표인 PyTorch Dataset, DataLoader, Transform, Batch 구성, Tensor 검증, Random Seed 기반 재현성 설정을 완료했다.

다음 Day에서는 이 입력 Pipeline을 기반으로 CNN Baseline 모델의 구조, Forward 흐름, Loss Function, Optimizer, 학습 Loop를 구현한다.
