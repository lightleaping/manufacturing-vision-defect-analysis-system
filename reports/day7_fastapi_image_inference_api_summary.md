# Day 7 — FastAPI Image Inference API

## 1. 프로젝트

```text
영문: Manufacturing Vision Defect Analysis System
한글: 제조 비전 결함 분석 시스템
```

Day 7의 목표는 Day 4에서 학습한 ResNet18 Best Checkpoint를 재학습하지 않고
FastAPI 서비스에서 재사용하여, 업로드한 제조 이미지를 `NORMAL` 또는
`DEFECT`로 분류하는 HTTP 추론 경로를 구현하는 것이다.

---

## 2. 구현 범위

```text
GET  /api/v1/health
POST /api/v1/predictions
GET  /docs
GET  /redoc
```

구현 내용:

- FastAPI Application Factory
- Lifespan 기반 모델 1회 로딩
- App State 기반 Inference Service 보관
- JPEG·JPG·PNG 업로드 검증
- 제한 크기 Chunk 읽기
- Pillow 실제 Decode 및 무결성 검증
- RGBA·Grayscale 입력의 RGB 변환
- Day 2 Test Transform 재사용
- ResNet18 Best Checkpoint 복원
- `torch.inference_mode()` 추론
- Raw Logit → Sigmoid → Threshold 처리
- 공통 오류 응답
- 단위 테스트·통합 테스트
- 실제 Uvicorn HTTP 검증

이번 기본 Prediction Endpoint에는 Grad-CAM을 포함하지 않았다.
일반 추론과 설명 가능성 계산의 책임을 분리하여 응답 시간과 연산 비용을
줄이고, 향후 별도 Explain Endpoint로 확장할 수 있도록 설계했다.

---

## 3. API 구조

```text
FastAPI 시작
→ Lifespan
→ create_production_inference_service()
→ restore_best_checkpoint()
→ create_test_transform()
→ ImageInferenceService
→ app.state.inference_service
```

Prediction 요청 흐름:

```text
multipart/form-data 이미지 업로드
→ 확장자·Content-Type 교차 검증
→ 10 MB 제한 읽기
→ Pillow 실제 Decode
→ JPEG·PNG 실제 형식 확인
→ 25,000,000 Pixel 제한 확인
→ 원본 Metadata 기록
→ RGB 변환
→ Resize 224 × 224
→ ToTensor
→ ImageNet Normalize
→ Tensor [1, 3, 224, 224]
→ torch.inference_mode()
→ ResNet18 Raw Logit
→ sigmoid
→ Threshold 0.5
→ NORMAL 또는 DEFECT JSON
```

---

## 4. 모델 재사용 정책

```text
Checkpoint:
models/checkpoints/resnet18_transfer_best.pt

Model:
ResNet18Transfer

Device:
cpu

Positive Class:
DEFECT

Threshold:
0.5
```

모델은 요청마다 다시 로딩하지 않는다.
FastAPI Lifespan에서 Checkpoint를 한 번 복원하고, 준비된 Service를
`app.state.inference_service`에 저장한다.

Production Loader는 기존 Day 4의 검증된
`restore_best_checkpoint(checkpoint_path, device)`를 그대로 재사용한다.
따라서 Checkpoint Schema를 API 계층에서 중복 구현하거나 추측하지 않는다.

전처리 역시 Day 2의 `create_test_transform()`을 그대로 사용한다.

---

## 5. 이미지 검증 정책

지원 입력:

```text
Extensions:
.jpg
.jpeg
.png

Content-Types:
image/jpeg
image/png

Decoded Formats:
JPEG
PNG
```

검증 기준:

- 파일 없음
- 빈 파일
- 지원하지 않는 확장자
- 지원하지 않는 Content-Type
- 확장자와 Content-Type 불일치
- Metadata와 실제 Decode 형식 불일치
- 손상 이미지
- Pillow가 읽을 수 없는 이미지
- 10 MB 초과 업로드
- 25,000,000 Pixel 초과 이미지
- 잘못된 Transform 출력
- NaN·Infinity 모델 출력
- 잘못된 모델 출력 Shape

이미지는 서버 디스크에 영구 저장하지 않는다.
원본 파일명을 서버 경로로 사용하지 않으며, 모든 Decode와 RGB 변환은
메모리 안에서 수행한다.

Mode 처리:

```text
RGB  → 그대로 사용
RGBA → RGB 변환
L    → RGB 변환
```

응답의 `image_mode`에는 RGB 변환 전 원본 Mode를 기록한다.

---

## 6. 추론 정책

```text
Model Output:
Raw Logit

defect_probability:
sigmoid(raw_logit)

normal_probability:
1 - defect_probability

prediction:
1 if defect_probability >= 0.5 else 0

0 = NORMAL
1 = DEFECT
```

모델 내부에는 Sigmoid를 추가하지 않았다.
기존 학습·평가 구조와 동일하게 모델은 Raw Logit만 반환하고,
API Service에서 확률 변환과 Threshold 판정을 수행한다.

---

## 7. 오류 응답

공통 형식:

```json
{
  "detail": {
    "code": "INVALID_IMAGE",
    "message": "업로드한 파일을 정상적인 이미지로 읽을 수 없습니다."
  }
}
```

오류 코드:

```text
MISSING_FILE
EMPTY_FILE
UNSUPPORTED_FILE_TYPE
FILE_TOO_LARGE
IMAGE_TOO_LARGE
INVALID_IMAGE
MODEL_NOT_READY
INVALID_MODEL_INPUT
INVALID_MODEL_OUTPUT
INFERENCE_FAILED
```

내부 Checkpoint 경로, Python Stack Trace, 원본 예외 문자열은 외부 응답에
노출하지 않는다.

---

## 8. 실제 HTTP 검증

Health 응답:

```text
Status       : ok
Service      : Manufacturing Vision Defect Analysis System
Model Loaded : True
Model Name   : ResNet18Transfer
Device       : cpu
```

### NORMAL 이미지

```text
Path              : data/raw/casting_product_images/casting_data/casting_data/test/ok_front/cast_ok_0_7631.jpeg
Prediction        : NORMAL
Prediction Index  : 0
P(DEFECT)         : 0.013476800174
P(NORMAL)         : 0.986523199826
Raw Logit         : -4.293217182159
Original Size     : 300 × 300
Original Mode     : RGB
Inference Time    : 196.59 ms
```

### DEFECT 이미지

```text
Path              : data/raw/casting_product_images/casting_data/casting_data/test/def_front/cast_def_0_1414.jpeg
Prediction        : DEFECT
Prediction Index  : 1
P(DEFECT)         : 0.999903678894
P(NORMAL)         : 0.000096321106
Raw Logit         : 9.247676849365
Original Size     : 300 × 300
Original Mode     : RGB
Inference Time    : 54.83 ms
```

전체 검증 Runtime:

```text
14.21 seconds
```

검증 Artifact:

```text
reports/artifacts/day7_fastapi_inference_validation.json
```

Day 6에서 선택한 동일 표본의 확률과 API 응답이 일치하므로,
Script 추론과 HTTP 추론이 같은 Checkpoint와 같은 Test Transform을
사용한다는 것을 교차 검증했다.

---

## 9. Dependency

실제 실행 환경:

```text
FastAPI         : 0.139.1
Uvicorn         : 0.51.0
python-multipart: 0.0.32
httpx           : 0.28.1
Pydantic        : 2.13.4
Pillow          : 12.3.0
PyTorch         : 2.12.0+cpu
Torchvision     : 0.27.0+cpu
```

---

## 10. 테스트

Day 7 대상 테스트:

```text
40 passed
```

전체 회귀 테스트:

```text
1255 passed
```

검증 범위:

- 이미지 업로드와 Decode
- RGB·RGBA·Grayscale 처리
- 확장자·Content-Type·실제 형식 교차 검증
- 파일 크기·Pixel 제한
- Raw Logit·Sigmoid·Threshold
- NORMAL·DEFECT 분류
- `torch.inference_mode()`
- NaN·Infinity 차단
- 모델 출력 Shape 검증
- Production Loader의 기존 함수 재사용
- FastAPI Health·Prediction Endpoint
- Model Not Ready
- Startup 실패 보호
- 내부 경로·Stack Trace 비노출
- 실제 ResNet18 HTTP 요청

테스트 실행 중 다음 호환성 경고가 1건 발생했다.

```text
StarletteDeprecationWarning:
Using httpx with starlette.testclient is deprecated;
install httpx2 instead.
```

현재 테스트와 실제 HTTP 검증은 모두 통과했다.
이 경고는 기능 실패가 아니며, 향후 FastAPI·Starlette 테스트 Client
Dependency 정책을 정리할 때 처리할 기술부채로 기록한다.

---

## 11. 실행 방법

API 실행:

```powershell
python -m uvicorn `
    src.api.app:app `
    --host 127.0.0.1 `
    --port 8000
```

문서:

```text
Swagger UI: http://127.0.0.1:8000/docs
ReDoc     : http://127.0.0.1:8000/redoc
```

실제 HTTP 자동 검증:

```powershell
python -m scripts.run_day7_api_validation
```

---

## 12. 현재 범위와 향후 확장

현재 구현:

- 단일 이미지 정상·불량 이진 분류
- CPU 추론
- JPEG·PNG 업로드
- 동기 단일 요청
- 모델 1회 로딩
- JSON 응답
- 안전한 입력·오류 검증

향후 확장 가능 범위:

- 별도 Grad-CAM Explain Endpoint
- Batch Prediction
- 요청 추적 ID
- 구조화 Logging
- Prometheus Metric
- Streamlit Dashboard 연결
- Model Version Registry
- 비동기 Queue 기반 대량 추론

이번 Day 7에서는 범위 확장을 위해 복잡도를 늘리지 않고,
포트폴리오에 필요한 안정적인 단일 이미지 추론 API를 우선 완성했다.

---

## 13. 실무 포인트

1. 모델 학습 코드와 HTTP 요청 처리를 분리했다.
2. Lifespan에서 모델을 한 번만 로딩하여 요청마다 Checkpoint를 읽지 않는다.
3. 실제 Decoder 결과까지 확인하여 확장자 위장 파일을 차단한다.
4. 업로드 크기와 Pixel 수를 모두 제한한다.
5. 모델은 Raw Logit을 유지하고 Service 계층에서 확률을 계산한다.
6. Dummy Service 주입으로 API 테스트가 실제 Checkpoint에 의존하지 않는다.
7. 실제 Uvicorn HTTP 검증에서만 Best Checkpoint와 Dataset 이미지를 사용한다.
8. 내부 경로와 Stack Trace를 외부 오류 응답에 노출하지 않는다.
