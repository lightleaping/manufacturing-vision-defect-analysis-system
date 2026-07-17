# Day 9 — Object Detection Dataset Analysis

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

- 전체 이미지: **1,800장**
- 전체 XML Annotation: **1,800개**
- 유효 Record: **1,800개**
- 유효 Bounding Box: **4,189개**
- Class 수: **6개**
- 이미지 Mode: `{'RGB': 1800}`

## 5. Class 분포

| Class | 이미지 수 | Bounding Box 수 |
|---|---:|---:|
| `crazing` | 300 | 689 |
| `inclusion` | 382 | 1011 |
| `patches` | 342 | 881 |
| `pitted_surface` | 301 | 432 |
| `rolled_in_scale` | 300 | 628 |
| `scratches` | 300 | 548 |

Class별 이미지 수와 Class별 Box 수는 서로 다른 지표로 분리해 기록했다.

## 6. 원본 데이터 품질 문제와 보정 정책

실제 전체 분석에서 다음 문제가 발견됐다.

1. `crazing_240.jpg`는 원본 `train/images`에 있지만 대응 XML은 `validation/annotations`에 있었다.
2. `patches_101.jpg`와 `patches_105.jpg`는 SHA-256이 같은 동일 이미지지만 Bounding Box 좌표가 조금 달랐다.
3. XML 내부 `<filename>`과 실제 Pair 파일명이 다른 Metadata 경고가 존재했다.
4. 동일 Class·동일 좌표의 중복 Box가 일부 존재했다.

원본 파일은 이동·삭제·수정하지 않았다. 전역에서 동일 stem 이미지와 XML이 각각 하나뿐인 경우에만 Manifest 수준에서 Pair를 연결했고, 동일 이미지 Hash 그룹은 삭제하지 않고 항상 하나의 최종 Split 안에 유지해 데이터 누수를 막았다.

- 원본 누락 Annotation: `1`
- 원본 누락 이미지: `1`
- 보정 후 누락 Annotation: `0`
- 보정 후 누락 이미지: `0`
- 교차 Partition Pair 보정: `1`
- 중복 이미지 Hash 그룹: `1`
- 손상 이미지: `0`
- 잘못된 Annotation: `0`
- 잘못된 Bounding Box: `0`
- 최종 Error: `0`

| 품질 이슈 | 개수 |
|---|---:|
| `cross_partition_pair_reconciled` | 1 |
| `duplicate_box` | 3 |
| `duplicate_image_hash` | 1 |
| `filename_mismatch` | 174 |

## 7. Bounding Box 통계

| 지표 | 평균 | 중앙값 | 최소 | 최대 |
|---|---:|---:|---:|---:|
| 이미지당 Box 수 | 2.3272 | 2.0000 | 1.0000 | 9.0000 |
| Box Width | 71.4015 | 55.0000 | 8.0000 | 199.0000 |
| Box Height | 95.0093 | 77.0000 | 9.0000 | 199.0000 |
| Box Area Ratio | 0.1745 | 0.1179 | 0.0027 | 0.9900 |
| Box Aspect Ratio | 1.1326 | 0.6818 | 0.0455 | 18.0000 |

## 8. 좌표 정책

- 추론된 원본 좌표 정책: `pascal_voc_one_based_inclusive_likely`
- 최소 좌표 0 등장 수: `0`
- 최소 좌표 1 등장 수: `1500`
- `xmax == image_width` Box 수: `258`
- `ymax == image_height` Box 수: `317`

전체 결과는 Pascal VOC의 **1-based inclusive 좌표**일 가능성이 높다. Day 11 Torchvision Detection Dataset에서는 `(xmin - 1, ymin - 1, xmax, ymax)` 변환을 명시적으로 구현하고 테스트한다.

## 9. Split 정책과 데이터 누수 검사

- Split 정책: `preserve_source_train_and_hash_group_split_source_validation_pool`
- 중복 Hash 정책: `preserve_duplicate_records_and_keep_each_identical_hash_group_inside_one_final_split`
- Random Seed: `42`

| Split | 이미지 | Box | 내부 중복 Hash 그룹 |
|---|---:|---:|---:|
| Train | 1440 | 3335 | 1 |
| Validation | 178 | 425 | 0 |
| Test | 182 | 429 | 0 |

Split 검증:

- 경로 중복: `0`
- Split 간 동일 이미지 Hash 누수: `0`
- Manifest Record: `1800`
- 전체 Record 보존: `True`
- 최종 유효성: `True`

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
- 전체 회귀 테스트: **1,368 passed**
- 전체 회귀 테스트 경고: **1 warning**
- 전체 회귀 테스트 Runtime: `80.33초`
- 기존 Starlette/httpx 관련 Warning은 기능 실패와 분리해 기술부채로 유지한다.

## 12. 현재 범위와 다음 일정

Day 9에서는 Dataset과 Annotation Pipeline을 검증했다. 객체 탐지 모델 학습, mAP 평가, Detection API·Dashboard는 아직 완료되지 않았다.

```text
Day 10 — OpenCV Image Analysis Pipeline
Day 11 — Detection Dataset and Model Implementation
Day 12 — Detection Training, Evaluation and Failure Analysis
Day 13 — Detection FastAPI and Streamlit Integration
Day 14 — Final Integration, README, Portfolio and Interview
```
