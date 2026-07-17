# Day 13 — Detection FastAPI and Streamlit Integration

## 1. 목표

Day 12에서 선택을 끝낸 Faster R-CNN Best Checkpoint를 기존
Classification 시스템과 충돌하지 않는 Detection API로 연결하고,
Streamlit이 Checkpoint를 직접 로딩하지 않은 채 FastAPI Client를 통해
Class·Score·원본 이미지 좌표 Bounding Box를 표시하도록 구현했다.

Day 13은 최종 Portfolio·Interview·전체 Architecture 정리 단계가 아니다.
해당 작업은 Day 14 범위로 남긴다.

## 2. 최종 호출 흐름

```text
Browser
→ Streamlit Detection Page
→ DetectionDashboardApiClient
→ POST /api/v1/detection/predictions
→ FastAPI Upload Validation
→ DetectionInferenceService
→ Day 12 Best Checkpoint
→ Detection Response JSON
→ Prediction Overlay·Table
```

기존 Classification 흐름은 유지한다.

```text
POST /api/v1/predictions
→ ResNet18 Classification

POST /api/v1/detection/predictions
→ Faster R-CNN Object Detection
```

## 3. Detection Model Service

| 항목 | 결과 |
| --- | --- |
| Architecture | `fasterrcnn_mobilenet_v3_large_320_fpn` |
| Device | CPU |
| Checkpoint | `models/detection/day12_detection_best.pt` |
| Checkpoint Epoch | 3 |
| Best Validation mAP@0.50 | 0.677418 |
| Checkpoint Size | 152,015,461 bytes |
| Startup Policy | FastAPI Lifespan 1회 로딩 |
| Inference Context | `torch.inference_mode()` |
| CPU Forward Safety | Process 내부 Lock |
| Network Weight Download | 사용하지 않음 |

Checkpoint 내부 Epoch Index 2는 사람이 읽는 Epoch 3을 의미한다.
Test 결과를 사용해 Checkpoint를 다시 선택하지 않았다.

## 4. Detection API

```text
POST /api/v1/detection/predictions
```

정상 응답은 다음 정보를 제공한다.

```text
Class ID
Class Name
Score
Bounding Box (xmin, ymin, xmax, ymax)
Detection Count
Score Threshold
IoU Threshold
Model·Checkpoint Metadata
원본 이미지 Metadata
Inference Time
```

Threshold 정책:

| 항목 | 값 |
| --- | ---: |
| 기본 Score Threshold | 0.50 |
| 최소 Score Threshold | 0.05 |
| 최대 Score Threshold | 0.95 |
| IoU Threshold Metadata | 0.50 |

Dashboard Slider는 Prediction 탐색용이다. Day 12 공식 Test 지표는
Score Threshold 0.5 결과로 유지하며 `crazing` Recall을 이유로 기본값을
임의로 낮추지 않았다.

## 5. 입력·출력 방어

기존 업로드 이미지 검증을 Detection에서도 재사용했다.

```text
빈 파일
JPEG·PNG 확장자
MIME Type
확장자·MIME·실제 Decode 형식 일치
깨진 이미지
업로드 Byte 제한
Pixel 수 제한
RGB 변환
```

Detection Service는 모델 출력에 대해 다음을 추가 검증한다.

```text
boxes=[N,4]
labels=[N]
scores=[N]
개수 일치
NaN·inf 차단
Label 1~6
Class Mapping 일치
Score 0~1
Threshold 적용
xmin < xmax
ymin < ymax
원본 이미지 좌표 범위
Score 내림차순 정렬
```

Torchvision이 원본 입력 크기로 복원한 Box를 사용하며, API에서
NEU-DET의 200×200 좌표로 임의 변환하지 않는다.

## 6. 실제 Detection API Smoke Test

| 항목 | 결과 |
| --- | --- |
| HTTP Status | 200 |
| Endpoint | `/api/v1/detection/predictions` |
| Score Threshold | 0.50 |
| Detection Count | 0 |
| Inference Time | 321.35 ms |
| Checkpoint Epoch | 3 |
| Result | PASS |

검증 이미지는 `C:\Users\kflow\Downloads\manufacturing-vision-defect-analysis-system\data\raw\neu_det\NEU-DET\train\images\crazing\crazing_1.jpg`를 사용했다.

## 7. Streamlit Detection Page

기존 Classification `src/dashboard/app.py`를 유지하고,
Streamlit Multipage 구조에 Detection 전용 페이지를 추가했다.

```text
src/dashboard/pages/2_Detection.py
src/dashboard/detection_page.py
src/dashboard/detection_api_client.py
src/dashboard/detection_session_state.py
src/dashboard/detection_ui_helpers.py
```

주요 화면 기능:

```text
이미지 업로드
Score Threshold Slider
원본 이미지 Preview
Detection 실행
Detection Count
Inference Time
Checkpoint Epoch
원본·Prediction Overlay 좌우 표시
Prediction Table
Class·Score·Box 좌표
빈 Detection 상태
안전한 API 오류 메시지
```

Streamlit Detection Page는 `torch`, `torchvision`, `src.detection`,
Detection Checkpoint Loader를 직접 Import하지 않는다.

## 8. Overlay 정책

Box에는 `P1`, `P2` 같은 짧은 태그만 표시하고 Class·Score·좌표는
이미지 밖 Prediction Table에서 제공한다. 긴 Class 이름과 Box의 겹침을
방지하며 Score 상위 8개까지 Overlay에 표시한다.

자동 Dashboard Client 검증:

| 항목 | 결과 |
| --- | --- |
| Detection Count | 0 |
| Inference Time | 414.04 ms |
| Checkpoint Epoch | 3 |
| Overlay Figure | `reports/figures/day13_detection_dashboard_overlay.png` |
| API Client Only | PASS |
| Result | PASS |

검증 이미지는 `C:\Users\kflow\Downloads\manufacturing-vision-defect-analysis-system\data\raw\neu_det\NEU-DET\train\images\crazing\crazing_1.jpg`를 사용했다.

브라우저 수동 시각 검증 결과는 이 보고서에 별도로 기록하지 않았다. 완료 사실은 자동 HTTP Client·Overlay Artifact·정적 구조 검증 범위로 한정한다.

## 9. OpenCV와 Detection의 구분

```text
OpenCV
→ 이미지 명암·경계·형태 특성 보조 분석
→ Threshold·Morphology 기반 Contour 후보
→ Ground Truth나 Detection Prediction이 아님

Detection
→ 학습된 Faster R-CNN Prediction
→ Class·Score·Bounding Box 제공
→ Ground Truth가 아님
```

Detection 0개는 정상 제품 또는 Ground Truth 결함 없음이라는 뜻이 아니다.
현재 Score Threshold 이상으로 반환된 모델 Prediction이 없다는 뜻이다.

## 10. 오류 처리

API Client는 다음 오류를 안전한 Dashboard 메시지로 변환한다.

```text
Timeout
Connection Error
HTTP 4xx
HTTP 5xx
잘못된 JSON
Schema 누락
잘못된 Class Mapping
범위 밖 Box
NaN·inf
Detection Model Not Ready
```

서버 내부 경로·Checkpoint 경로·Stack Trace는 Dashboard에 노출하지 않는다.

## 11. 검증 결과

| 검증 | 결과 |
| --- | ---: |
| Day 13 Targeted Tests | 92 passed |
| Full Regression Tests | 1668 passed |
| Warnings | 1 |
| API Prerequisites | PASS |
| Best Checkpoint Inspection | PASS |
| Actual Detection API Smoke Test | PASS |
| Dashboard Static Inspection | PASS |
| Dashboard API Client·Overlay Validation | PASS |

기존 Warning이 있다면 Day 7부터 유지된 Starlette `TestClient`와
`httpx` 관련 기술부채이며 Day 13에서 Dependency를 무리하게 변경하지 않았다.

## 12. 생성 Artifact

```text
reports/artifacts/day13_integration_prerequisites.json
reports/artifacts/day13_detection_api_stage1_inspection.json
reports/artifacts/day13_detection_api_smoke_test.json
reports/artifacts/day13_detection_dashboard_stage2_inspection.json
reports/artifacts/day13_detection_dashboard_api_client_validation.json
reports/figures/day13_detection_dashboard_overlay.png
reports/artifacts/day13_detection_integration_summary.json
```

## 13. Day 14 연결 지점

Day 14에서는 다음을 진행한다.

```text
최종 기능 통합 점검
README 전체 구조 정리
최종 Architecture Diagram
Portfolio 문구
Interview 답변
최종 실행·회귀 검증
```

Day 13 보고서에서는 위 작업을 완료했다고 표현하지 않는다.
