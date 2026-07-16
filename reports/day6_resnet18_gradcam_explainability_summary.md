# Day 6 - ResNet18 Grad-CAM Explainability Summary

## 1. 프로젝트와 Day 6 목표

프로젝트명:

```text
영문: Manufacturing Vision Defect Analysis System
한글: 제조 비전 결함 분석 시스템
```

Day 6의 목표는 Day 4에서 학습·평가한 ResNet18 전이학습 모델이
특정 이미지를 `NORMAL` 또는 `DEFECT`로 판단할 때 마지막 Convolution
Layer의 어느 공간 영역을 상대적으로 강하게 사용했는지 Grad-CAM으로
시각화하는 것이다.

Day 5에서 확인한 고확신 오분류와 결정 경계 오분류를 정분류 표본과 함께
비교하여 모델의 판단 근거를 분석할 수 있는 설명 가능성 Artifact를
추가했다.

---

## 2. Grad-CAM이 필요한 이유

분류 지표만으로는 모델이 왜 해당 결과를 출력했는지 확인할 수 없다.

Grad-CAM은 다음 질문을 검토하는 데 사용한다.

```text
모델이 실제 제품 영역을 보고 있는가?
결함으로 보이는 표면 패턴을 보고 있는가?
제품 가장자리·조명·배경을 잘못된 단서로 사용하고 있는가?
정분류와 오분류의 주목 영역에 차이가 있는가?
```

Grad-CAM은 모델 검증과 오류 분석을 보조하지만 실제 결함 위치의
정답 Mask, Bounding Box 또는 Detection 결과는 아니다.

---

## 3. 구현 구조

구현 파일:

```text
src/explainability/gradcam.py
src/explainability/gradcam_sample_selector.py
src/explainability/gradcam_visualization.py
src/explainability/gradcam_pipeline.py
scripts/run_day6_resnet18_gradcam.py
```

테스트 파일:

```text
tests/test_gradcam.py
tests/test_gradcam_sample_selector.py
tests/test_gradcam_visualization.py
tests/test_gradcam_pipeline.py
tests/test_day6_resnet18_gradcam.py
```

호출 흐름:

```text
Day 4 평가 JSON
→ Day 5 오분류 JSON 교차 검증
→ 대표 표본 7장 선택
→ ResNet18 Best Checkpoint 복원
→ Test Transform 적용
→ Batch Size 1 Forward
→ Target Score Backward
→ Activation·Gradient 저장
→ Channel Weight 계산
→ Weighted Activation 합산
→ ReLU 및 0~1 정규화
→ 원본·Heatmap·Overlay 생성
→ JSON·PNG Atomic 저장
```

---

## 4. Grad-CAM 고정 정책

Target Layer:

```text
resnet18.layer4.1.conv2
```

Target 정책:

```text
Prediction = DEFECT
→ target_score = raw_logit

Prediction = NORMAL
→ target_score = -raw_logit
```

설정:

```text
Batch Size             = 1
Classification Threshold = 0.5
Target Policy          = predicted_class
Channel Weight         = Spatial Mean of Gradients
Weighted Activation    = Channel Weighted Sum
ReLU                   = True
Normalization          = Min-Max 0~1
Input Size             = [224, 224]
Input Normalization    = ImageNet Mean / Standard Deviation
Color Map              = jet
Overlay Alpha          = 0.40
```

마지막 Convolution Layer는 Classification Head에 가까운 고수준 Feature를
가지면서도 공간 정보를 보존한다. Fully Connected Layer 이후에는 공간
정보가 사라지므로 `resnet18.layer4.1.conv2`를 사용했다.

---

## 5. Hook와 Grad-CAM 계산

Forward Hook는 Target Layer의 Activation Feature Map을 저장한다.

Forward 과정에서 얻은 Activation Tensor에 Gradient Hook를 등록하고,
Target Score를 Backward하여 다음 Gradient를 저장한다.

```text
∂target_score / ∂activation
```

Channel Weight:

```python
weights = gradients.mean(
    dim=(2, 3),
    keepdim=True,
)
```

Weighted Activation:

```python
cam = (
    weights * activations
).sum(
    dim=1,
    keepdim=True,
)
```

최종 처리:

```text
ReLU
→ 입력 Tensor 크기로 Bilinear Resize
→ Min-Max 0~1 정규화
→ 원본 이미지 크기로 Resize
→ Color Map
→ Alpha Blending
```

Frozen Backbone에서도 Activation Gradient를 생성할 수 있도록 Grad-CAM
계산용 입력 복사본에만 `requires_grad_(True)`를 적용했다. Optimizer나
학습 Parameter Update는 수행하지 않는다.

---

## 6. 대표 표본 선택 결과

총 7장을 자동 선택했다.

```text
고확신 정분류 NORMAL 1장
고확신 정분류 DEFECT 1장
고확신 False Positive 2장
고확신 False Negative 1장
결정 경계 False Positive 1장
결정 경계 False Negative 1장
```

| 유형 | Index | 파일 | Ground Truth | Prediction | P(DEFECT) | Target |
|---|---:|---|---|---|---:|---|
| 고확신 True Negative | 198 | `cast_ok_0_7631.jpeg` | NORMAL | NORMAL | 0.013477 | NORMAL |
| 고확신 True Positive | 313 | `cast_def_0_1414.jpeg` | DEFECT | DEFECT | 0.999904 | DEFECT |
| 고확신 False Positive 1 | 202 | `cast_ok_0_7839.jpeg` | NORMAL | DEFECT | 0.828241 | DEFECT |
| 고확신 False Positive 2 | 21 | `cast_ok_0_1121.jpeg` | NORMAL | DEFECT | 0.825668 | DEFECT |
| 고확신 False Negative | 342 | `cast_def_0_150.jpeg` | DEFECT | NORMAL | 0.256505 | NORMAL |
| 결정 경계 False Positive | 157 | `cast_ok_0_4497.jpeg` | NORMAL | DEFECT | 0.504139 | DEFECT |
| 결정 경계 False Negative | 384 | `cast_def_0_1591.jpeg` | DEFECT | NORMAL | 0.492131 | NORMAL |

정분류는 모델 확신도가 가장 높은 표본을 사용했고, 오분류는 잘못된 예측
확신도와 Threshold 0.5까지의 거리를 기준으로 선택했다. 동일 표본이
여러 기준에 걸릴 경우 먼저 선택한 표본을 유지하고 다음 후보를 선택하여
중복을 제거했다.

---

## 7. 실제 실행 결과

```text
Generated Samples          = 7
Correct Prediction Samples = 2
High-Confidence Errors     = 3
Boundary Errors            = 2
Runtime                    = 4.81 seconds
Target Layer               = resnet18.layer4.1.conv2
Device                     = CPU
```

Day 4 저장 결과와 Grad-CAM 재추론 결과도 표본별로 비교했다.

```text
Prediction Match           = 모든 표본 True
Maximum Raw Logit Error    = 0.000000000000
Maximum Probability Error  = 0.000000000000
```

이 검증은 잘못된 Checkpoint 복원, 다른 Transform 사용, 이미지 경로 오류,
모델 Architecture 불일치를 Artifact 생성 전에 차단하기 위한 것이다.

---

## 8. 생성 Artifact

| Artifact | 경로 | 크기 |
|---|---|---:|
| Metadata JSON | `reports/artifacts/day6_resnet18_gradcam_analysis.json` | 13,521 bytes |
| 전체 비교 Figure | `reports/figures/day6_resnet18_gradcam_overview.png` | 1,191,265 bytes |
| 고확신 오류 Figure | `reports/figures/day6_resnet18_gradcam_high_confidence_errors.png` | 544,234 bytes |
| 결정 경계 오류 Figure | `reports/figures/day6_resnet18_gradcam_boundary_errors.png` | 387,459 bytes |

Figure 구성:

```text
Overview
→ 정분류 2장과 오분류 5장 전체 비교

High-Confidence Errors
→ 고확신 False Positive 2장과 False Negative 1장

Boundary Errors
→ 결정 경계 False Positive 1장과 False Negative 1장
```

모든 JSON과 PNG는 임시 파일에 먼저 기록한 뒤 `os.replace()`를 사용해
최종 경로로 교체했다.

---

## 9. 테스트 및 검증

Day 6 단위·통합 테스트:

```text
40 passed
```

전체 회귀 테스트:

```text
1204 passed
```

PNG 육안 확인:

```text
이상 없음
```

검증 범위:

```text
Target Layer 탐색
Forward Hook 등록
Activation Gradient 저장
NORMAL·DEFECT Target Score 방향
CAM Shape와 0~1 범위
NaN·Infinity 차단
Zero CAM 차단
Batch Size 1 정책
Hook 해제
대표 표본 선택
Day 4·Day 5 교차 검증
Day 4 예측 재현
이미지 로딩과 손상 검사
Heatmap·Overlay 생성
Windows 환경 Figure 메모리 사용 개선
JSON·PNG Atomic 저장
실제 ResNet18 Best Checkpoint 실행
전체 회귀 테스트
```

---

## 10. 오류 처리와 자원 정리

다음 상황을 명시적인 예외로 처리한다.

```text
Target Layer가 존재하지 않음
Activation 또는 Gradient가 저장되지 않음
Activation·Gradient Shape 불일치
Batch Size가 1이 아님
입력 Tensor에 NaN·Infinity 포함
CAM에 NaN·Infinity 발생
CAM 전체 값이 0
이미지 파일 누락 또는 손상
Day 4·Day 5 Artifact 불일치
재추론 Logit·Probability·Prediction 불일치
빈 시각화 표본
Figure 저장 실패
```

Grad-CAM Context 종료 시 Hook를 반드시 해제한다. Figure도 성공과 실패에
관계없이 `clear()`와 `close()`를 수행해 Windows CPU 환경에서 자원이
누적되지 않도록 했다.

---

## 11. 실무 해석 원칙

Grad-CAM 결과는 다음과 같이 사용한다.

```text
정분류에서 제품의 결함 관련 영역을 일관되게 보는지 확인
False Positive에서 정상 패턴을 결함으로 오인한 영역 확인
False Negative에서 실제 결함을 놓치고 다른 영역을 본 가능성 확인
결정 경계 오류에서 Heatmap이 분산되거나 애매한지 확인
배경·가장자리·조명 편향 여부 확인
```

다만 Heatmap의 붉은 영역을 실제 결함 위치라고 단정하지 않는다. Grad-CAM은
모델의 상대적인 주목 영역을 보여주는 설명 보조 수단이며, 정밀한 결함
위치 검증에는 별도의 Annotation과 Detection·Segmentation 평가가 필요하다.

---

## 12. 면접 설명

### Q1. Binary Classification에서 Class별 Logit이 하나뿐인데 NORMAL Grad-CAM은 어떻게 계산했나요?

DEFECT는 Raw Logit이 증가하는 방향이므로 `raw_logit`을 Target Score로
사용했다. NORMAL은 Raw Logit이 감소하는 방향이므로 `-raw_logit`을
Target Score로 사용했다.

### Q2. 왜 마지막 Convolution Layer를 선택했나요?

Classification Head에 가장 가까워 결함 판별에 필요한 고수준 Feature를
포함하면서도 위치 정보를 보존하기 때문이다. Fully Connected Layer는
공간 정보가 없어 Heatmap 생성에 적합하지 않다.

### Q3. Frozen Backbone인데 Gradient를 계산할 수 있나요?

Parameter 학습 Gradient와 Grad-CAM 설명 Gradient는 목적이 다르다.
Optimizer나 Parameter Update 없이 Target Score를 Activation까지
Backward하면 설명용 Gradient를 구할 수 있다. Frozen Parameter와 입력이
모두 Gradient를 요구하지 않는 경우를 방지하기 위해 Grad-CAM 전용 입력
복사본에만 `requires_grad_(True)`를 적용했다.

### Q4. Grad-CAM으로 결함 위치를 검출했다고 말할 수 있나요?

말할 수 없다. Grad-CAM은 모델이 예측에 상대적으로 사용한 영역을
시각화할 뿐이며 실제 결함 위치의 정답 Mask나 Detection 결과가 아니다.

### Q5. 왜 전체 Test Dataset 715장에 Grad-CAM을 생성하지 않았나요?

Day 6의 목적은 대표 오류 분석과 포트폴리오 설명 가능성 확보다. CPU
환경에서 전체 표본을 처리하는 것보다 정분류 2장, 고확신 오류 3장,
결정 경계 오류 2장을 선택해 비교하는 것이 효율적이고 분석 목적에도
적합하다.

---

## 13. Day 6 결론

Day 6에서는 외부 Grad-CAM 라이브러리에 의존하지 않고 PyTorch Hook 기반
Grad-CAM 계산 과정을 직접 구현했다.

ResNet18 Best Checkpoint와 Day 4·Day 5 Artifact를 연결하고, 대표 표본
7장에 대해 모델이 실제로 선택한 예측 Class 관점의 Heatmap을 생성했다.
재추론 결과를 기존 평가 JSON과 다시 비교했으며, Metadata JSON과
세 종류의 PNG Figure를 Atomic 방식으로 저장했다.

최종적으로 1204 passed와 PNG 육안 확인 `이상 없음`을
통과하여 Day 6 ResNet18 Grad-CAM Explainability 구현과 검증을 완료했다.
