# Day 11 — Detection Dataset and Model Implementation

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

0. `background`
1. `crazing`
2. `inclusion`
3. `patches`
4. `pitted_surface`
5. `rolled_in_scale`
6. `scratches`

Torchvision Detection에서는 `0`을 Background로 사용하므로 6개 결함 Class를 포함한 `num_classes`는 **7**이다.

## 4. 좌표 변환

원본 Pascal VOC 좌표는 1-based inclusive로 해석하고 다음 정책으로 변환했다.

```python
(xmin - 1, ymin - 1, xmax, ymax)
```

예를 들어 원본 이미지 전체를 나타내는 `[1, 1, 200, 200]`은 Torchvision Target에서 `[0, 0, 200, 200]`이 된다.

## 5. 실제 Dataset Runtime Validation

| Split | Images | Boxes | Validation |
|---|---:|---:|---|
| Train | 1,440 | 3,335 | PASS |
| Validation | 178 | 425 | PASS |
| Test | 182 | 429 | PASS |
| **Total** | **1,800** | **4,189** | **PASS** |

검증 항목에는 Image Tensor의 Shape·dtype·range, Target Dict Key, Box·Label·Area dtype, 좌표 범위, Manifest와 Target의 순서별 일치, DataLoader Batch 계약, Split 경로 중복 여부가 포함된다.

## 6. Duplicate Box 정책

```text
정책                 : preserve
정확히 같은 중복 Box : 3
원본 XML 수정         : False
Loader 임의 삭제      : False
```

기본 정책은 `preserve`다. 보조 설정 `remove_exact`은 동일 Annotation에서 Class와 네 좌표가 모두 같은 Box만 제거할 수 있지만 Day 11 실제 검증과 Day 12 기본 입력에는 적용하지 않았다.

## 7. Detection Model

```text
Architecture             : fasterrcnn_mobilenet_v3_large_320_fpn
Device                   : cpu
Predictor output classes : 7
Detection weights        : None
Backbone weights         : None
Network download         : False
Smoke input resize       : [64, 64]
```

CPU 실행 가능성을 우선해 `fasterrcnn_mobilenet_v3_large_320_fpn`을 선택했다. Day 11에서는 구조 검증을 위해 `weights=None`, `weights_backbone=None`을 사용했다.

## 8. Training Forward Smoke Test

Source sample: `train/crazing_1`

| Loss | Value |
|---|---:|
| `loss_classifier` | 1.958665 |
| `loss_box_reg` | 1.202479 |
| `loss_objectness` | 0.690874 |
| `loss_rpn_box_reg` | 0.000582 |
| **Total** | **3.852600** |

```text
Training forward time   : 0.255s
All losses finite       : True
Backward executed       : False
Optimizer step executed : False
```

## 9. Evaluation Forward Smoke Test

```text
Evaluation forward time : 0.069s
Prediction boxes        : 10
Boxes shape             : [10, 4]
Labels shape            : [10]
Scores shape            : [10]
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
Day 11 targeted tests : 52 passed
Full regression tests : 1492 passed
Warnings              : 1
```

## 12. Day 12 연결

Day 12에서는 디스크 공간을 확보한 뒤 COCO pretrained Detection Weight를 적용하고, 같은 Model Factory의 COCO Predictor를 NEU-DET 7-Class Predictor로 교체해 Fine-tuning한다. 이후 Validation·Test Prediction, IoU 기반 평가, mAP·mAR, Checkpoint와 Failure Analysis를 수행한다.

Day 11에서는 전체 학습, 최종 Checkpoint, mAP·mAR, Failure Analysis를 완료했다고 표현하지 않는다.
