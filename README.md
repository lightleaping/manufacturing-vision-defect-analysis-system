# 제조 표면 결함 Vision AI

**Surface Defect Classification, Detection, Failure Analysis, and Inference API**

> 제조 이미지의 정상·불량 분류와 표면 결함 객체 탐지를 구현하고, 모델 평가와 실패 사례 분석을 FastAPI·Streamlit 사용자 흐름까지 연결한 Vision AI 프로젝트입니다.

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3561D8?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.12-21AFC4?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Vision-Classification%20%7C%20Detection-151F32?style=flat-square" alt="Vision Classification and Detection">
  <img src="https://img.shields.io/badge/FastAPI-Inference%20API-3561D8?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Tests-1737%20passed-21AFC4?style=flat-square" alt="Tests">
</p>

---

## Why This Project

제조 이미지 검사에서 필요한 판단은 한 가지가 아닙니다. 제품 전체가 정상인지 불량인지 빠르게 판정해야 하고, 불량이라면 어떤 결함이 어디에 있는지도 확인해야 합니다. 분류 모델은 제품 단위 판정에 적합하지만 위치를 제공하지 않으며, 객체 탐지 모델은 결함의 종류와 Bounding Box를 제공할 수 있습니다.

또한 높은 Accuracy 하나만으로는 실제 불량 누락, 불필요한 불량 판정, 위치 오류를 구분하기 어렵습니다. 따라서 이 프로젝트는 다음 목표로 구성했습니다.

1. 이미지 전체의 정상·불량을 분류한다.
2. 6개 표면 결함의 종류와 위치를 탐지한다.
3. False Negative, False Positive, 위치 오류와 Class 혼동을 따로 분석한다.
4. 모델 결과를 FastAPI와 Streamlit 입력·응답 흐름으로 연결한다.

## Project Overview

| 항목 | 내용 |
|---|---|
| **기간** | 2026.07 |
| **형태** | 개인 프로젝트 |
| **목표** | 이미지 전체의 정상·불량 상태와 개별 표면 결함의 종류·위치를 서로 다른 모델로 분석 |
| **범위** | 이미지 데이터 분석, 분류 모델, 객체 탐지 모델, OpenCV 보조 분석, 성능 평가, 실패 사례 분석, FastAPI, Streamlit, 테스트, 문서화 |
| **기술** | Python, PyTorch, torchvision, ResNet18, Faster R-CNN, OpenCV, scikit-learn, FastAPI, Streamlit, pytest |
| **대표 결과** | DEFECT-class F1 97.92%, Detection mAP@0.50 0.7077, FastAPI Endpoint 3/3 PASS |

---

## Problem → Implementation → Result

| Problem | Implementation | Result |
|---|---|---|
| 이미지 전체의 불량 여부만으로는 개별 결함의 종류와 위치를 알기 어려움 | ResNet18 분류와 Faster R-CNN 객체 탐지를 목적별 파이프라인으로 분리 | 정상·불량 분류와 6개 표면 결함 탐지를 각각 제공 |
| 전체 성능 수치만으로는 모델이 어떤 사례에서 실패하는지 확인하기 어려움 | 분류 오분류와 객체 탐지 실패 사례를 유형별로 수집하고 시각화 | 분류 오분류 19개, 객체 탐지 실패 유형 5종 정리 |
| 모델 결과가 학습 코드에만 머물면 실제 입력·응답 흐름을 확인하기 어려움 | FastAPI가 모델을 로드하고 Streamlit이 HTTP API로 결과를 요청하도록 구성 | 분류·검출 Endpoint, Schema, Dashboard, 통합 테스트 검증 |

---

## System Overview

<img src="./docs/assets/vision-system-overview.svg" alt="제조 표면 결함 Vision AI 시스템 구성도" width="100%">

| Flow | Output |
|---|---|
| **Classification** | NORMAL 또는 DEFECT, Defect Probability |
| **Object Detection** | 결함 Class, Confidence Score, Bounding Box |
| **Service** | FastAPI JSON Response와 Streamlit Result 또는 Overlay |


### Design Decisions

- **분류와 객체 탐지 분리**: 서로 다른 질문에 답하므로 모델과 Endpoint를 분리했습니다.
- **FastAPI에서 모델 로드**: 요청마다 Checkpoint를 다시 읽지 않도록 Lifespan에서 한 번만 로드했습니다.
- **Streamlit은 Client 역할**: Streamlit은 모델을 직접 로드하지 않고 FastAPI 응답을 표시합니다.
- **OpenCV는 보조 분석**: Contour Candidate는 이미지 상태 확인용이며 객체 탐지 결과로 사용하지 않았습니다.
- **Grad-CAM은 설명 보조**: 모델 반응 영역을 확인하는 시각화 도구로 사용했습니다.

---

## Practical Evaluation Criteria

제조 Vision 모델에도 모든 현장에 공통인 단일 합격 수치는 없습니다. 제품 단가, 결함 심각도, 검사 속도, 수동 재검사 절차에 따라 기준이 달라집니다. 다만 다음 평가 축은 실제 검사 의사결정과 직접 연결됩니다.

| 실무 관점 | 평가 기준 | 프로젝트에서 확인한 근거 |
|---|---|---|
| **불량 누락 감소** | DEFECT Recall, False Negative | Classification Recall **98.68%**, FN **6** |
| **불필요한 불량 판정 관리** | Precision, False Positive, Confusion Matrix | Classification Precision **97.17%**, FP **13** |
| **결함 탐지 범위** | Detection Recall, Missed Detection | Detection Recall **52.68%**, 누락 사례를 별도 Failure Type으로 분석 |
| **탐지 신뢰도** | Precision, mAP | Detection Precision **81.30%**, mAP@0.50 **0.7077** |
| **위치 정확성** | IoU, Localization Error | Mean Matched IoU **0.7523**, 위치 오류 사례 분리 |
| **실패 원인 확인** | FP, FN, Localization, Duplicate, Class Confusion | 객체 탐지 실패 유형 **5종** 정리 |
| **서비스 연결성** | 입력 검증, API Response, UI 표시, Regression Test | FastAPI Endpoint **3/3 PASS**, 1,737개 회귀 테스트 |

> Classification은 제품 단위 불량 판정에서 높은 Recall과 F1을 확인했고, Detection은 Precision과 위치 IoU를 확인하면서 결함 누락을 다음 개선 우선순위로 구체화했습니다. 실제 생산 적용 기준은 결함별 위험도, 처리 속도와 재검사 비용을 반영해 정해야 합니다.

## Key Results

### Classification

| Metric | CNN Baseline | ResNet18 Transfer |
|---|---:|---:|
| Accuracy | 76.92% | **97.34%** |
| Precision | 82.88% | **97.17%** |
| Recall | 80.13% | **98.68%** |
| F1 Score | 81.48% | **97.92%** |
| False Negative | 90 | **6** |
| Total Errors | 165 | **19** |

> Precision, Recall, F1은 `DEFECT` Class 기준입니다.

### Object Detection

| Metric | Test Result |
|---|---:|
| Precision | **0.812950** |
| Recall | 0.526807 |
| F1 | 0.639321 |
| mAP@0.50 | **0.707726** |
| Mean Matched IoU | **0.752338** |

### Validation

| Verification | Result |
|---|---:|
| FastAPI Endpoints | **3/3 PASS** |
| Full Regression Tests | **1,737 passed** |
| Test Runtime | 100.56 seconds |

---

## Visual Evidence

<table>
  <tr>
    <td width="50%" align="center">
      <b>Classification False Negative Analysis</b><br><br>
      <img src="./reports/figures/day5_resnet18_false_negatives.png" alt="ResNet18 False Negative 분석" width="100%">
    </td>
    <td width="50%" align="center">
      <b>Grad-CAM</b><br><br>
      <img src="./reports/figures/day6_resnet18_gradcam_overview.png" alt="ResNet18 Grad-CAM" width="100%">
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <b>Detection Predictions</b><br><br>
      <img src="./reports/figures/day12_detection_predictions.png" alt="객체 탐지 예측 결과" width="100%">
    </td>
    <td width="50%" align="center">
      <b>Detection Failure Analysis</b><br><br>
      <img src="./reports/figures/day12_detection_failure_analysis.png" alt="객체 탐지 실패 사례 분석" width="100%">
    </td>
  </tr>
</table>

> 전체 오분류는 False Positive 13개와 False Negative 6개로 집계했으며, 상단에는 제조 품질 판정에서 우선 확인할 필요가 있는 False Negative 사례를 표시했습니다.

---

## Technical Details

<details open>
<summary><b>01 | Classification</b></summary>

<br>

### Dataset

| 항목 | 내용 |
|---|---:|
| Dataset | Casting Product Image Data for Quality Inspection |
| Total | 7,348 images |
| Train | 5,306 |
| Validation | 1,327 |
| Test | 715 |
| Target | NORMAL / DEFECT |

### CNN Baseline

```text
Input
→ Conv(3→8) → ReLU → Pool
→ Conv(8→16) → ReLU → Pool
→ Conv(16→32) → ReLU → Pool
→ Adaptive Average Pool
→ Linear(32→1)
→ Raw Logit
```

- Parameters: 6,065
- 목적: Dataset, 학습 Loop, Checkpoint, 평가 흐름의 기준 모델 구성

### ResNet18 Transfer Learning

```text
ImageNet Pretrained ResNet18
→ Frozen Backbone
→ Linear(512→1)
→ Raw Logit
```

- Total Parameters: 11,177,025
- Trainable Parameters: 513
- Best Validation Accuracy: 97.06%
- Test Accuracy: 97.34%

</details>

<details>
<summary><b>02 | Object Detection</b></summary>

<br>

### Dataset and Model

| 항목 | 내용 |
|---|---|
| Dataset | NEU Surface Defect |
| Images | 1,800 |
| Classes | 6 |
| Model | Faster R-CNN MobileNetV3 Large 320 FPN |
| Output | Class, Score, Bounding Box |
| Best Checkpoint | Epoch 2 |

### Output

```text
Image
→ Faster R-CNN
→ Class
→ Confidence Score
→ Bounding Box
```

</details>

<details>
<summary><b>03 | Failure Analysis</b></summary>

<br>

### Classification

Test 715장 중 19개 오분류를 다음과 같이 구분했습니다.

- False Positive: 13
- False Negative: 6
- Error Rate: 2.66%

제조 품질 판정에서 불량을 정상으로 판단하는 False Negative를 별도로 확인하고, 예측 확률과 이미지 패턴을 함께 검토했습니다.

### Object Detection

| Failure Type | 의미 |
|---|---|
| Low Confidence | 결함 후보를 찾았지만 Score가 낮은 사례 |
| Missed Detection | 실제 결함을 탐지하지 못한 사례 |
| Localization Error | 결함을 찾았지만 Bounding Box 위치가 부정확한 사례 |
| Duplicate Detection | 하나의 결함을 여러 Box로 예측한 사례 |
| Class Confusion | 위치는 찾았지만 결함 Class를 다르게 예측한 사례 |

Precision보다 Recall이 낮아 현재 모델에서는 과도한 오탐보다 일부 결함 누락을 줄이는 개선이 우선이라고 판단했습니다.

</details>

<details>
<summary><b>04 | OpenCV and Grad-CAM</b></summary>

<br>

### OpenCV Auxiliary Analysis

- Brightness와 Histogram
- Edge Detection
- Threshold
- Morphology
- Contour Candidate

OpenCV 분석은 이미지 상태와 형태적 특징을 확인하는 보조 과정입니다. Contour Candidate를 객체 탐지의 정답 위치나 Faster R-CNN 결과로 사용하지 않았습니다.

### Grad-CAM

ResNet18의 반응 영역이 결함과 관련된 부분에 나타나는지, 배경이나 주변 패턴에 과도하게 반응하는지를 시각적으로 확인했습니다.

Grad-CAM은 결함 위치의 Ground Truth가 아니라 모델 반응 영역을 확인하기 위한 설명 보조 수단입니다.

</details>

<details>
<summary><b>05 | FastAPI and Streamlit</b></summary>

<br>

### Endpoints

| Method | Endpoint | 역할 |
|---|---|---|
| GET | `/api/v1/health` | 모델과 서비스 상태 확인 |
| POST | `/api/v1/predictions` | 정상·불량 분류 |
| POST | `/api/v1/detection/predictions` | 결함 Class, Score, Bounding Box 반환 |

### Integration Flow

```text
Image Upload
→ Streamlit API Client
→ FastAPI Validation
→ Loaded Model Inference
→ JSON Response
→ Prediction or Detection Overlay
```

</details>

<details>
<summary><b>06 | Validation</b></summary>

<br>

다음 범위를 pytest와 통합 실행으로 검증했습니다.

- Dataset Configuration, Split, Transform, DataLoader
- CNN, ResNet18, Faster R-CNN Model Flow
- Checkpoint Metadata and Loading
- Accuracy, Precision, Recall, F1, Confusion Matrix
- Detection mAP, IoU, Class Metrics
- Misclassification and Detection Failure Artifact
- Image Validation and API Schema
- FastAPI Endpoint and HTTP Integration
- Streamlit API Client and Overlay
- Final Regression Test

</details>

<details>
<summary><b>07 | Current Scope and Next Steps</b></summary>

<br>

### Current Scope

- Classification과 Detection은 목적과 데이터셋이 다른 독립 파이프라인입니다.
- Detection은 CPU 환경에서 3 Epoch 학습한 Checkpoint를 평가했습니다.
- Grad-CAM은 모델 반응 영역을 확인하는 보조 분석입니다.
- 공개 제조 이미지 데이터셋을 사용한 프로젝트입니다.

### Next Steps

1. Detection Epoch, Scheduler, Augmentation 비교
2. Class별 Recall과 누락 원인에 따른 Sampling 개선
3. COCOeval 기반 표준 AP 추가
4. 모델 Version, 추론 Log, Latency Monitoring
5. 실제 제조 데이터 환경을 고려한 Drift와 재학습 기준 설계

</details>

---

## API Response

### Classification

```json
{
  "prediction": "DEFECT",
  "defect_probability": 0.9999,
  "threshold": 0.5,
  "model": "resnet18_transfer"
}
```

### Object Detection

```json
{
  "detections": [
    {
      "class_name": "crazing",
      "score": 0.91,
      "box": [31.2, 48.5, 151.7, 176.9]
    }
  ]
}
```

---

## Run and Verify

### Environment

| 항목 | 내용 |
|---|---|
| OS | Windows |
| Python | 3.11.9 |
| PyTorch | 2.12.0+cpu |
| Torchvision | 0.27.0+cpu |
| CUDA | False |

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
python -m pip check
```

### FastAPI

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

### Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\src\dashboard\app.py
```

### Tests

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests -q
```

---

## Project Structure

```text
manufacturing-vision-defect-analysis-system/
├── data/
├── docs/
│   └── assets/
├── reports/
│   ├── artifacts/
│   ├── figures/
│   └── day*_summary.md
├── scripts/
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── data/
│   ├── detection/
│   ├── evaluation/
│   ├── explainability/
│   ├── models/
│   ├── opencv_analysis/
│   ├── services/
│   └── training/
├── tests/
├── README.md
└── requirements.txt
```

---

## What This Project Demonstrates

- 제조 이미지 데이터를 PyTorch 학습 파이프라인으로 구성한 경험
- CNN Baseline과 ResNet18 전이학습을 동일한 평가 기준으로 비교한 경험
- 이미지 분류와 객체 탐지의 역할을 구분해 구현한 경험
- Accuracy뿐 아니라 Recall, F1, mAP, IoU와 실제 실패 사례를 함께 분석한 경험
- PyTorch 모델을 FastAPI와 Streamlit 사용자 흐름으로 연결한 경험
- 모델, API, Dashboard를 테스트하고 결과를 문서화한 경험

---

## Contact

- Developer: 김수진
- GitHub: [github.com/lightleaping](https://github.com/lightleaping)
- Email: workingskyroad@gmail.com
