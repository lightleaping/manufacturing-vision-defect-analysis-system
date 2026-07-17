# Day 10 — OpenCV Image Analysis Pipeline

## 1. 완료 상태

```text
Project              : Manufacturing Vision Defect Analysis System
한글명               : 제조 비전 결함 분석 시스템
Day                  : 10
Target tests         : 62 passed
Full regression      : 1440 passed
Warnings             : 1
Regression runtime   : 95.56 seconds
Visual validation    : PASS
```

Day 10에서는 학습 모델이 아닌 **OpenCV 기반 이미지 명암·경계·형태 특성 보조 분석 파이프라인**을 구현했다. OpenCV 결과는 Classification이나 Object Detection을 대체하지 않는다.

## 2. 기능 구분

| 기능 | 역할 | 출력 |
|---|---|---|
| Classification | 이미지 전체가 NORMAL 또는 DEFECT인지 판단 | Class, probability |
| OpenCV Analysis | 명암·경계·Threshold·형태 계산 결과를 사람이 확인 | Metrics, masks, edges, contour candidates |
| Object Detection | 학습된 모델이 결함 종류와 위치를 예측 | Class, bounding box, confidence |

Contour는 Adaptive Threshold와 Morphology 결과에서 계산된 **후보 외곽선**이다. 실제 결함 Ground Truth, 객체 탐지 Bounding Box 또는 Detection Prediction으로 해석하지 않는다.

## 3. 처리 파이프라인

```text
Pillow Image
→ RGB 정규화
→ OpenCV BGR
→ Grayscale
→ Histogram
→ CLAHE
→ Gaussian Blur
→ Canny Edge
→ Adaptive Threshold
→ Morphology Opening·Closing
→ Minimum Area Ratio Filter
→ Contour Candidate Overlay
→ JSON Metrics + PNG Figures
```

각 단계의 목적은 다음과 같다.

| 단계 | 목적 |
|---|---|
| Grayscale | 색상 채널을 단일 명암 채널로 변환 |
| Histogram | 0~255 명암 분포와 Peak 확인 |
| CLAHE | 국소 영역 대비를 제한적으로 향상 |
| Gaussian Blur | 작은 Noise와 급격한 픽셀 변화를 완화 |
| Canny Edge | 명암 변화가 큰 경계 픽셀 추출 |
| Adaptive Threshold | 영역별 밝기 기준으로 전경과 배경 분리 |
| Morphology | 작은 전경 Noise 제거와 끊어진 영역 연결 |
| Contour Overlay | Threshold·Morphology 기반 후보 형태 시각화 |

## 4. 결정론적 Config

| Parameter | Value |
|---|---:|
| CLAHE clip limit | `2.000000` |
| CLAHE tile grid | `8 × 8` |
| Gaussian kernel | `5 × 5` |
| Gaussian sigma X | `0.000000` |
| Canny low threshold | `50` |
| Canny high threshold | `150` |
| Adaptive threshold block | `11` |
| Adaptive threshold C | `2.000000` |
| Adaptive threshold invert | `True` |
| Morphology kernel | `3 × 3` |
| Morphology opening iterations | `1` |
| Morphology closing iterations | `1` |
| Minimum contour area ratio | `0.000500` |
| Maximum contours | `500` |
| Contour line thickness | `1` |

파라미터를 불변 Dataclass로 분리해 동일 입력과 동일 설정에서 같은 처리 결과를 만들고, 실행 Artifact에 실제 설정을 남겼다. Kernel 홀수 여부, Canny Threshold 순서, Contour 면적 비율 범위 등의 오류는 실행 전에 검증한다.

## 5. 이미지 변환 정책

```text
Pillow RGB : R, G, B
OpenCV BGR : B, G, R
Grayscale  : Height × Width
dtype      : uint8
```

Pillow 입력은 RGB로 정규화한 뒤 OpenCV BGR로 변환한다. Figure에 표시할 때는 다시 RGB로 변환한다. 빈 배열, 잘못된 Shape, 지원하지 않는 dtype과 이미지 확장자는 명시적으로 거부한다.

## 6. 실제 이미지 분석

전체 Dataset을 반복 분석하지 않고 재현 가능한 고정 샘플 3장만 사용했다.

| Sample | File | Shape | Mean | Contrast | Edge ratio | Foreground ratio | Contours | Largest contour ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Casting NORMAL | `cast_ok_0_7631.jpeg` | 300×300 | 151.496 | 66.031 | 0.037011 | 0.231433 | 7 | 0.590917 |
| Casting DEFECT | `cast_def_0_1414.jpeg` | 300×300 | 131.717 | 54.715 | 0.052589 | 0.209833 | 20 | 0.194039 |
| NEU-DET Defect Image | `crazing_1.jpg` | 200×200 | 160.637 | 28.498 | 0.307775 | 0.425400 | 83 | 0.045900 |

Casting과 NEU-DET는 Dataset의 의미와 촬영 조건이 다르므로 위 수치를 모델 성능이나 Dataset 우열 비교로 해석하지 않는다.

## 7. 정량 지표

| Metric | 의미 |
|---|---|
| Mean Brightness | Grayscale 픽셀 평균 밝기 |
| Brightness Standard Deviation | 이미지 내 명암 변화 정도 |
| Histogram Peak | 가장 많은 픽셀이 분포한 명암값 |
| Otsu Threshold | 전체 Histogram 기반 참고 Threshold |
| Edge Pixel Ratio | 전체 픽셀 중 Canny Edge 비율 |
| Threshold Foreground Ratio | Morphology 결과의 전경 픽셀 비율 |
| Contour Count | 최소 면적 기준을 통과한 후보 Contour 수 |
| Largest Contour Area Ratio | 가장 큰 후보 Contour 면적 비율 |
| Average Contour Area Ratio | 후보 Contour 평균 면적 비율 |

이 값들은 이미지의 계산 특성을 설명하는 보조 지표이며 실제 결함 검출 성능 지표가 아니다.

## 8. 생성 Artifact

| Artifact | Path |
|---|---|
| OpenCV analysis JSON | `reports/artifacts/day10_opencv_image_analysis.json` |
| Pipeline overview | `reports/figures/day10_opencv_pipeline_overview.png` |
| Histogram and metrics | `reports/figures/day10_opencv_histogram_and_metrics.png` |
| Contour candidate analysis | `reports/figures/day10_opencv_contour_analysis.png` |
| Visual validation JSON | `reports/artifacts/day10_opencv_visual_validation.json` |

세 Figure는 Pillow Decode, PNG 형식, Width·Height 및 파일 크기를 자동 검증했다. 이후 Layout, RGB 표시, Metrics 가독성, Contour 후보 주의 문구를 직접 확인했고 육안 검증 결과를 별도 JSON으로 기록했다.

## 9. Dependency

| Dependency | Version |
|---|---:|
| opencv-python | `4.13.0.92` |
| cv2 | `4.13.0` |
| NumPy | `2.4.6` |
| Pillow | `12.3.0` |
| Matplotlib | `3.11.0` |

기존 NumPy와의 호환성을 유지하면서 기본 `opencv-python`만 추가했다. `opencv-contrib-python`, `scikit-image`, YOLO Framework 등은 Day 10 범위에 필요하지 않아 추가하지 않았다.

## 10. 테스트

```text
Synthetic-image unit and integration tests : 62 passed
Full project regression tests              : 1440 passed
Warnings                                   : 1
Runtime                                    : 95.56 seconds
```

검정·흰색·Gradient·사각형·Noise 합성 이미지를 사용해 Config, 채널 변환, 단계별 Shape·dtype, 입력 불변성, 결정론, Metrics, PNG 생성, 실제 샘플 실행 Script 및 육안 검증 Script를 확인했다.

현재 Warning은 기존 Starlette TestClient와 httpx 조합에서 발생하는 기술부채이며 Day 10 기능 실패가 아니다. OpenCV 추가 과정에서 기존 API Dependency를 무리하게 변경하지 않았다.

## 11. 한계와 해석 정책

- Contour는 Threshold·Morphology 파라미터에 반응한 후보 형태다.
- 조명, 그림자, 제품 외곽선과 정상 Texture도 Contour가 될 수 있다.
- Contour에는 결함 Class와 Confidence가 없다.
- Contour를 Pascal VOC XML 대신 Detection Target으로 사용하지 않는다.
- Casting과 NEU-DET Metric을 모델 성능 비교로 사용하지 않는다.
- Day 10에서는 Detection 학습, mAP 평가, Checkpoint, Detection API를 구현하지 않았다.

## 12. Day 11 연결

Day 11에서는 Day 9의 Pascal VOC XML과 Split Artifact를 사용해 Torchvision Detection Dataset을 구현한다. Day 10 OpenCV 결과는 이미지 입력 품질과 보조 시각화를 확인하는 용도로만 연결하며, Detection Ground Truth는 다음 좌표 변환 정책을 유지한다.

```python
(xmin - 1, ymin - 1, xmax, ymax)
```

OpenCV Contour를 Detection Label이나 Bounding Box로 자동 변환하지 않는다.
