# Day 12 — Detection Training, Evaluation and Failure Analysis

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
Architecture             : fasterrcnn_mobilenet_v3_large_320_fpn
Device                   : cpu
Input min / max size     : 320 / 320
Pretrained detection     : FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
Classes with background  : 7
Best checkpoint epoch    : 3
Best validation metric   : mAP@0.50 = 0.677418
Duplicate box policy     : preserve
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
| 1 | `frozen_backbone_full_split_pilot` | Freeze | 0.005000 | 0.722109 | 11.79 min | 0.590604 | 0.207059 | 0.306620 | 0.402362 |
| 2 | `unfrozen_backbone_fine_tuning` | Unfreeze | 0.001000 | 0.954561 | 22.28 min | 0.847619 | 0.418824 | 0.560630 | 0.637687 |
| 3 | `unfrozen_backbone_fine_tuning` | Unfreeze | 0.001000 | 1.016442 | 20.90 min | 0.833333 | 0.505882 | 0.629575 | 0.677418 |

Training Loss는 최적화를 위한 신호이고 Detection Metric은 최종 Class·Box 품질을 측정한다. Backbone을 연 뒤 Loss 분포가 달라져 Train Loss가 상승했지만 Validation Recall·F1·mAP가 크게 개선됐으므로 성능 판단에는 Validation Detection Metric을 사용했다.

## 5. Checkpoint 정책

```text
Latest : 마지막으로 완전히 완료된 Epoch의 재개 상태
Best   : Validation mAP@0.50이 개선된 Epoch만 교체
```

Checkpoint에는 Epoch, Model·Optimizer State, Scheduler State 또는 None, Training Config, Class Mapping, Best Metric, History, Torch·Torchvision Version을 저장한다. Epoch 3의 Validation mAP@0.50 0.677418가 최종 Best로 선택됐다.

## 6. 최종 Validation·Test 결과

| Split | TP | FP | FN | Precision | Recall | F1 | Mean Matched IoU | mAP@0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 215 | 43 | 210 | 0.833333 | 0.505882 | 0.629575 | 0.763037 | 0.677418 |
| Test | 226 | 52 | 203 | 0.812950 | 0.526807 | 0.639321 | 0.752338 | 0.707726 |

```text
Test project mAP@0.50:0.95 : 0.310533
Score threshold             : 0.500000
Matching IoU threshold      : 0.500000
```

Test mAP@0.50은 0.707726이고 Mean Matched IoU는 0.752338다. 더 엄격한 IoU Threshold를 평균한 프로젝트 mAP@0.50:0.95는 0.310533로, 결함 존재·Class 탐지에 비해 정밀한 Localization에는 추가 개선 여지가 있음을 보여준다.

이 프로젝트 mAP@0.50:0.95는 직접 구현한 all-point AP를 IoU 0.50~0.95에 적용한 지표이며 공식 `pycocotools.COCOeval` 결과와 동일하다고 주장하지 않는다.

## 7. Test Class별 성능

| Class | GT | TP | FP | FN | Precision | Recall | F1 | Mean IoU | AP@0.50 | Project mAP@.50:.95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `crazing` | 79 | 2 | 1 | 77 | 0.666667 | 0.025316 | 0.048780 | 0.586356 | 0.522723 | 0.154379 |
| `inclusion` | 74 | 50 | 11 | 24 | 0.819672 | 0.675676 | 0.740741 | 0.745394 | 0.764932 | 0.328212 |
| `patches` | 105 | 82 | 8 | 23 | 0.911111 | 0.780952 | 0.841026 | 0.784679 | 0.888495 | 0.481216 |
| `pitted_surface` | 42 | 31 | 7 | 11 | 0.815789 | 0.738095 | 0.775000 | 0.752443 | 0.849149 | 0.422694 |
| `rolled_in_scale` | 66 | 25 | 16 | 41 | 0.609756 | 0.378788 | 0.467290 | 0.707393 | 0.495917 | 0.185080 |
| `scratches` | 63 | 36 | 9 | 27 | 0.800000 | 0.571429 | 0.666667 | 0.728660 | 0.725137 | 0.291613 |

- `patches`는 F1 0.841026, AP@0.50 0.888495로 가장 안정적이었다.
- `crazing`는 Ground Truth 79개 중 TP 2, FN 77으로 Recall 0.025316을 기록해 가장 큰 개선 대상이다.

특히 낮은 Score의 정답 Prediction이 많이 확인됐지만, Test 결과를 보고 기본 Threshold나 Checkpoint를 다시 선택하지 않았다. Day 13 API·Dashboard의 기본 Score Threshold도 검증에 사용한 0.5를 유지한다.

## 8. Failure Analysis

```text
Test images          : 182
Images with failures : 129
Failure events       : 229
```

| Failure Type | Count |
|---|---:|
| Low-confidence Correct Detection | 140 |
| False Negative | 37 |
| Low IoU Localization | 25 |
| False Positive | 23 |
| Duplicate Prediction | 3 |
| Wrong Class | 1 |

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
Day 12 targeted tests : 84 passed
Full regression tests : 1576 passed
Warnings              : 1
```

기존 Starlette/httpx deprecation warning을 Day 12에서 Dependency 변경 없이 유지했다.

## 12. 한계와 개선 방향

- `crazing`의 고정 Threshold Recall이 낮다.
- 프로젝트 mAP@0.50:0.95가 mAP@0.50보다 낮아 Localization 정밀도 개선이 필요하다.
- CPU와 3 Epoch 제한으로 Hyperparameter Search를 수행하지 않았다.
- Random Crop·Rotation·Mosaic·MixUp 등 결함 Box를 훼손할 수 있는 증강은 근거 없이 추가하지 않았다.
- Test 결과는 최종 보고에만 사용했으며 추가 모델 선택에 사용하지 않았다.

## 13. Day 13 연결

Day 13에서는 `models/detection/day12_detection_best.pt`, 동일한 7-Class Mapping, Score Threshold 0.5, 320 입력 정책을 Detection FastAPI Endpoint와 Streamlit 페이지에 연결한다. API는 Box·Class·Score를 반환하고 Dashboard는 Day 12의 가독성 정책을 재사용해 Ground Truth가 없는 실제 입력에서는 Prediction Overlay와 Threshold 정보를 표시한다.

Day 12에서는 Detection FastAPI Endpoint, Detection Streamlit 페이지, Classification·OpenCV·Detection 통합 UI를 완료했다고 표현하지 않는다.
