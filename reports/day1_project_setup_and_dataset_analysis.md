\# Day 1 — Project Setup and Dataset Analysis



\## 1. Day 1 목표



Day 1의 목표는 \*\*Manufacturing Vision Defect Analysis System\*\*의 독립 개발 환경과 기본 프로젝트 구조를 생성하고, 실제 제조 이미지 데이터셋을 확정한 뒤 모델 학습 전에 필요한 데이터 품질 정보를 분석하는 것이었다.



한국어 프로젝트명:



\*\*제조 비전 결함 분석 시스템\*\*



이번 프로젝트는 별도의 학습·연습 저장소를 만들지 않고, 실제 포트폴리오와 지원서 제출에 사용할 프로젝트 자체를 구현하면서 학습한다.



Day 1에서는 아직 CNN이나 ResNet18 모델을 구현하지 않았다.



모델을 먼저 만들기 전에 다음 정보를 확인하는 것을 우선했다.



\* 실제 이미지 수

\* 정상·불량 클래스 구성

\* Train·Test 구조

\* 클래스 비율

\* 이미지 확장자

\* 이미지 크기

\* 이미지 색상 모드

\* 이미지 채널 수

\* 손상 이미지

\* 미지원 파일

\* 샘플 이미지

\* 클래스 분포



\---



\## 2. 개발 환경



운영체제:



```text

Windows

```



터미널:



```text

PowerShell

```



Python:



```text

Python 3.11.9

```



프로젝트 경로:



```text

C:\\Users\\kflow\\Downloads\\manufacturing-vision-defect-analysis-system

```



가상환경:



```text

C:\\Users\\kflow\\Downloads\\

manufacturing-vision-defect-analysis-system\\

.venv

```



가상환경 활성화 명령:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\---



\## 3. 프로젝트 구조



Day 1에 생성한 기본 구조:



```text

manufacturing-vision-defect-analysis-system/

│

├── data/

│   ├── processed/

│   └── raw/

│

├── reports/

│   ├── artifacts/

│   └── day1\_project\_setup\_and\_dataset\_analysis.md

│

├── scripts/

│

├── src/

│   ├── \_\_init\_\_.py

│   │

│   ├── api/

│   │   └── \_\_init\_\_.py

│   │

│   ├── dashboard/

│   │   └── \_\_init\_\_.py

│   │

│   ├── data/

│   │   ├── \_\_init\_\_.py

│   │   ├── dataset\_analysis.py

│   │   ├── dataset\_config.py

│   │   └── dataset\_visualization.py

│   │

│   ├── inference/

│   │   └── \_\_init\_\_.py

│   │

│   ├── models/

│   │   └── \_\_init\_\_.py

│   │

│   └── training/

│       └── \_\_init\_\_.py

│

├── tests/

│   ├── \_\_init\_\_.py

│   ├── test\_dataset\_config\_and\_analysis.py

│   └── test\_dataset\_summary\_and\_visualization.py

│

├── .gitignore

├── README.md

└── requirements.txt

```



기능별 책임을 미리 분리하여 이후 데이터 처리, 모델, 학습, 추론, FastAPI, Streamlit 코드가 하나의 파일에 섞이지 않도록 구성했다.



\---



\## 4. 설치 라이브러리



Day 1에서 설치하고 검증한 라이브러리:



| Library    | Version | 역할                  |

| ---------- | ------: | ------------------- |

| NumPy      |   2.4.6 | 수치 데이터 및 시각화 배열 처리  |

| Pillow     |  12.3.0 | 이미지 열기·검증·크기·채널 분석  |

| Matplotlib |  3.11.0 | 클래스 분포 및 샘플 이미지 시각화 |

| pandas     |   3.0.3 | 이미지별 분석 결과와 통계 관리   |

| pytest     |   9.1.1 | 자동 테스트              |



라이브러리 import 검증 결과:



```text

\[PASS] Day 1 packages imported successfully

```



\---



\## 5. 데이터셋



사용 데이터셋:



```text

Casting Product Image Data for Quality Inspection

```



프로젝트 데이터 루트:



```text

data/raw/

casting\_product\_images/

casting\_data/

casting\_data

```



실제 구조:



```text

casting\_data/

├── train/

│   ├── def\_front/

│   └── ok\_front/

│

└── test/

&#x20;   ├── def\_front/

&#x20;   └── ok\_front/

```



프로젝트 클래스 정의:



```text

ok\_front



→ Label 0



→ NORMAL



→ 정상

```



```text

def\_front



→ Label 1



→ DEFECT



→ 불량

```



불량을 Label `1`로 지정했다.



이후 Precision, Recall, F1 Score를 계산할 때 불량을 Positive Class로 해석하며, 제조 품질 검사에서 중요한 False Negative를 분석하기 위한 기준이다.



\---



\## 6. 데이터셋 선택 이유



이번 프로젝트의 목표는 이상 탐지가 아니라 \*\*정상·불량 지도 이진 이미지 분류\*\*다.



학습 구조:



```text

정상 이미지



\+



불량 이미지



↓



CNN Baseline



↓



ResNet18



↓



정상·불량 확률



↓



NORMAL 또는 DEFECT

```



이번 프로젝트에서 모델이 답하는 질문:



> 입력된 주조 제품 이미지는 정상인가, 불량인가?



Casting Product 데이터셋은 정상과 불량 Label이 이미 제공되므로 다음 구현 범위와 직접 연결된다.



\* CNN Baseline

\* ResNet18 전이학습

\* Accuracy

\* Precision

\* Recall

\* F1 Score

\* Confusion Matrix

\* False Positive

\* False Negative

\* 오분류 이미지 분석

\* Grad-CAM

\* 이미지 추론

\* FastAPI

\* Streamlit



\---



\## 7. 사용하지 않는 데이터



압축 파일에는 다음 데이터도 포함되어 있었다.



```text

casting\_512x512/

├── def\_front/

└── ok\_front/

```



이미지 수:



```text

DEFECT



668장

```



```text

NORMAL



519장

```



총:



```text

1,187장

```



현재 프로젝트에서는 `casting\_512x512`를 학습 데이터에 포함하지 않는다.



이유:



\* 기존 `casting\_data`에 충분한 Train·Test 데이터가 존재한다.

\* 동일 데이터셋의 다른 크기 또는 가공 버전일 가능성이 있다.

\* 함께 사용하면 같은 원본에서 생성된 유사 이미지가 Train과 Test에 분리될 가능성이 있다.

\* 데이터 중복 또는 데이터 누수 위험을 불필요하게 증가시킬 수 있다.



현재는 두 데이터 구조의 이미지 대응 관계를 완전히 검증하지 않았으므로 동일 이미지라고 단정하지 않는다.



다만 불필요한 중복 가능성을 줄이기 위해 명확한 Train·Test 구조가 있는 `casting\_data`만 사용한다.



\---



\## 8. 데이터 분석 결과



전체 파일 수:



```text

7,348

```



유효 이미지:



```text

7,348

```



손상 이미지:



```text

0

```



지원하지 않는 파일:



```text

0

```



\---



\## 9. Split·Class별 이미지 수



| Split | Class  | Label | Image Count |

| ----- | ------ | ----: | ----------: |

| Train | NORMAL |     0 |       2,875 |

| Train | DEFECT |     1 |       3,758 |

| Test  | NORMAL |     0 |         262 |

| Test  | DEFECT |     1 |         453 |



전체 클래스 수:



| Class  | Image Count |

| ------ | ----------: |

| NORMAL |       3,137 |

| DEFECT |       4,211 |

| Total  |       7,348 |



전체 클래스 비율:



```text

NORMAL



3,137 / 7,348



약 42.69%

```



```text

DEFECT



4,211 / 7,348



약 57.31%

```



불량 이미지가 더 많지만 한 클래스가 매우 부족한 극단적인 불균형 상태는 아니다.



현재 단계에서는 클래스 가중치 또는 WeightedRandomSampler를 바로 적용하지 않는다.



Day 4 모델 학습 후 다음 결과를 확인하여 필요성을 판단한다.



\* 클래스별 Precision

\* 클래스별 Recall

\* F1 Score

\* False Positive

\* False Negative

\* Confusion Matrix



\---



\## 10. 이미지 속성 분석



전체 이미지 확장자:



```text

.jpeg

```



이미지 수:



```text

7,348

```



모든 이미지 크기:



```text

300 × 300

```



이미지 모드:



```text

RGB

```



채널 수:



```text

3

```



분석 결과:



```text

7,348장의 이미지가 모두



300 × 300



RGB



3채널



JPEG

```



\---



\## 11. 이미지 크기와 모델 입력



원본 이미지는 모두 `300 × 300`으로 동일하다.



Day 2에서는 CNN Baseline과 ResNet18의 입력 조건을 통일하기 위해 `224 × 224` Resize를 우선 검토한다.



예상 변환:



```text

원본 이미지



300 × 300 × 3



↓



Resize



224 × 224 × 3



↓



ToTensor



3 × 224 × 224

```



이미지가 이미 동일한 크기여도 Resize를 명시적으로 적용할 예정이다.



이유:



\* 모델 입력 Shape 고정

\* CNN과 ResNet18 입력 조건 통일

\* 추론 파이프라인과 학습 파이프라인 일치

\* 외부 이미지 입력에도 동일한 전처리 적용



\---



\## 12. RGB 처리



현재 모든 이미지는 RGB 3채널이다.



Day 2 Dataset에서는 다음 변환을 명시적으로 적용할 예정이다.



```python

image = image.convert("RGB")

```



현재 데이터에서는 변환 결과가 동일하지만, 이후 다른 이미지가 추가되더라도 항상 3채널 입력을 보장할 수 있다.



\---



\## 13. 손상 이미지



실제 분석 결과:



```text

손상 이미지



0장

```



현재 데이터셋에서 삭제하거나 제외할 파일은 없다.



하지만 Dataset과 FastAPI에서는 앞으로도 이미지 로딩 예외를 처리한다.



현재 데이터가 정상이라고 해서 이후 사용자 업로드 이미지 또는 새로운 데이터도 항상 정상이라고 가정할 수 없기 때문이다.



\---



\## 14. 데이터 분석 코드



\### `src/data/dataset\_config.py`



책임:



\* 프로젝트 루트 경로

\* 데이터셋 경로

\* Train·Test 경로

\* 클래스 Label

\* 지원 이미지 확장자



입력:



```text

현재 Python 파일 위치

```



출력:



```text

PROJECT\_ROOT



DATASET\_ROOT



TRAIN\_ROOT



TEST\_ROOT



CLASS\_TO\_INDEX



INDEX\_TO\_CLASS\_NAME



SUPPORTED\_IMAGE\_EXTENSIONS

```



\---



\### `src/data/dataset\_analysis.py`



책임:



\* 데이터 폴더 구조 검증

\* 이미지 파일 탐색

\* 확장자 확인

\* 이미지 손상 검사

\* 이미지 Width·Height 분석

\* 이미지 Mode 분석

\* 이미지 채널 분석

\* 분석 DataFrame 생성

\* CSV 저장

\* JSON 저장



주요 호출 흐름:



```text

main()



↓



analyze\_dataset()



↓



validate\_dataset\_structure()



↓



analyze\_split()



↓



analyze\_file()



↓



Image.verify()



↓



Image.load()



↓



create\_dataset\_summary()



↓



save\_analysis\_results()

```



\---



\### `src/data/dataset\_visualization.py`



책임:



\* 분석 CSV 로드

\* Train·Test 클래스 분포 시각화

\* NORMAL·DEFECT 샘플 선택

\* 샘플 이미지 Grid 생성

\* 결과 PNG 저장



호출 흐름:



```text

main()



↓



load\_analysis\_dataframe()



↓



create\_class\_distribution\_chart()



↓



create\_sample\_image\_grid()



↓



PNG 저장

```



\---



\## 15. 분석 결과 파일



생성 파일:



```text

reports/artifacts/

day1\_dataset\_analysis.csv

```



역할:



이미지 한 장마다 다음 정보를 저장한다.



\* Split

\* 원본 클래스

\* 숫자 Label

\* 프로젝트 클래스

\* 상대 파일 경로

\* 파일명

\* 확장자

\* 파일 크기

\* 지원 확장자 여부

\* 유효 이미지 여부

\* 손상 여부

\* Width

\* Height

\* 이미지 Mode

\* 채널 수

\* 처리 상태

\* 오류 메시지



\---



생성 파일:



```text

reports/artifacts/

day1\_dataset\_summary.json

```



역할:



전체 데이터 분석 결과를 JSON으로 저장한다.



포함 정보:



\* 데이터셋 이름

\* 클래스 정의

\* Train·Test 경로

\* 전체 파일 수

\* 유효 이미지 수

\* 손상 이미지 수

\* 미지원 파일 수

\* Split·Class별 이미지 수

\* 확장자별 이미지 수

\* 이미지 크기

\* 이미지 Mode

\* 채널 수

\* 손상 파일 목록

\* 미지원 파일 목록



\---



생성 파일:



```text

reports/artifacts/

day1\_class\_distribution.png

```



역할:



Train·Test의 NORMAL·DEFECT 이미지 수를 막대그래프로 비교한다.



\---



생성 파일:



```text

reports/artifacts/

day1\_sample\_images.png

```



역할:



Train 데이터의 NORMAL·DEFECT 샘플 이미지를 비교한다.



구성:



```text

첫 번째 행



NORMAL 이미지 4장

```



```text

두 번째 행



DEFECT 이미지 4장

```



샘플 선택 Seed:



```text

42

```



같은 데이터로 다시 실행하면 같은 샘플을 선택할 수 있도록 재현성을 확보했다.



\---



\## 16. 기존 프로젝트 참고



\### 기존 코드 참고



Manufacturing AI Quality Agent의 다음 구조 원칙을 참고했다.



```text

설정값 분리



→ 처리 기능 분리



→ 결과 저장



→ 예외 처리



→ 테스트

```



\### 그대로 재사용



기존 코드에서 직접 복사한 Vision 분석 코드는 없다.



\### 수정하여 재사용



기존 프로젝트의 경로·설정 분리 원칙을 유지했지만 입력 대상을 센서 데이터에서 이미지 데이터로 변경했다.



\### 신규 구현



기존 프로젝트에 없던 다음 기능을 새로 구현했다.



\* 이미지 파일 탐색

\* 이미지 확장자 검사

\* Pillow 이미지 검증

\* 이미지 Width·Height 분석

\* 이미지 Mode 분석

\* 이미지 채널 분석

\* 손상 이미지 검사

\* 데이터 분석 CSV

\* 데이터 분석 JSON

\* 클래스 분포 시각화

\* 정상·불량 샘플 시각화



\---



\## 17. 테스트



Day 1 테스트 파일:



```text

tests/

test\_dataset\_config\_and\_analysis.py

```



테스트 수:



```text

7 passed

```



검증 내용:



\* 데이터 루트 존재

\* Train·Test 폴더 존재

\* 클래스 폴더 구조

\* NORMAL=0

\* DEFECT=1

\* 지원 이미지 확장자

\* 정상 RGB 이미지 분석

\* 손상 JPEG 처리

\* 미지원 TXT 처리



\---



Day 1 테스트 파일:



```text

tests/

test\_dataset\_summary\_and\_visualization.py

```



테스트 수:



```text

8 passed

```



검증 내용:



\* 전체·유효·손상 파일 수

\* 이미지 크기 요약

\* RGB·3채널 요약

\* Boolean 유효 이미지 처리

\* 문자열 Boolean 처리

\* Seed 기반 샘플 재현성

\* 클래스 분포 PNG 생성

\* 샘플 Grid PNG 생성

\* 분석 CSV 누락 예외



\---



전체 테스트:



```text

15 passed

```



실행 명령:



```powershell

python -m pytest `

&#x20;   .\\tests `

&#x20;   -v

```



\---



\## 18. 발생한 오류와 해결



\### 문제



처음 테스트 실행 시 다음 오류가 발생했다.



```text

pytest 명령을 찾을 수 없음

```



이후 기존 프로젝트인 다음 경로에서 가상환경을 활성화했다.



```text

C:\\Users\\kflow\\Downloads\\

manufacturing-mcp-agent

```



그 결과 다음 오류가 발생했다.



```text

ModuleNotFoundError:



No module named 'PIL'

```



\### 원인



새 Vision 프로젝트가 아닌 기존 MCP 프로젝트 폴더와 가상환경에서 테스트를 실행했다.



Vision 프로젝트의 Pillow와 pytest는 다음 환경에 설치되어 있었다.



```text

manufacturing-vision-defect-analysis-system/

.venv

```



\### 해결



기존 가상환경 종료:



```powershell

deactivate

```



Vision 프로젝트 이동:



```powershell

cd `

C:\\Users\\kflow\\Downloads\\

manufacturing-vision-defect-analysis-system

```



Vision 가상환경 활성화:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



현재 Python 확인:



```powershell

python -c "import sys; print(sys.executable)"

```



이후 테스트는 다음 형식으로 실행하도록 고정했다.



```powershell

python -m pytest `

&#x20;   .\\tests `

&#x20;   -v

```



`python -m pytest`는 현재 선택된 Python 환경에 설치된 pytest를 실행하므로 가상환경 혼동 가능성을 줄일 수 있다.



\---



\### 문제



테스트 코드에 잘못된 import 문법이 포함되었다.



잘못된 코드:



```python

import src.data.dataset\_visualization as (

&#x20;   dataset\_visualization

)

```



오류:



```text

SyntaxError: invalid syntax

```



\### 원인



Python의 `import ... as ...` 문법에서 별칭을 괄호로 감쌀 수 없다.



\### 해결



다음 한 줄로 수정했다.



```python

import src.data.dataset\_visualization as dataset\_visualization

```



수정 후:



```text

8 passed



15 passed

```



\---



\## 19. 실무 포인트



\### 데이터 분석을 모델보다 먼저 수행하는 이유



모델을 먼저 구현하면 다음 문제를 늦게 발견할 수 있다.



\* 이미지 크기 불일치

\* 채널 불일치

\* 손상 이미지

\* 잘못된 클래스 폴더

\* 클래스 불균형

\* 잘못된 Label

\* Train·Test 구성 오류



따라서 다음 순서를 사용했다.



```text

데이터 구조



→ 데이터 품질



→ 클래스 분포



→ 이미지 속성



→ Dataset



→ DataLoader



→ 모델

```



\---



\### Accuracy만 사용하지 않는 이유



현재 클래스 비율은 다음과 같다.



```text

DEFECT



57.31%

```



```text

NORMAL



42.69%

```



극단적인 불균형은 아니지만 Accuracy만 사용하면 클래스별 오류를 충분히 알 수 없다.



Day 4부터 다음 지표를 함께 확인한다.



\* Accuracy

\* Precision

\* Recall

\* F1 Score

\* Confusion Matrix



\---



\### 제조 불량 검사에서 False Negative



False Negative:



```text

실제 DEFECT



→ 모델 예측 NORMAL

```



실제 불량을 정상으로 통과시키는 오류다.



제조 현장에서는 고객 불량, 재작업, 품질 비용으로 연결될 수 있으므로 중요한 오류다.



이 프로젝트에서는 불량 Recall과 False Negative 수를 별도로 분석한다.



\---



\### 데이터 누수



현재 공개 데이터셋의 Train·Test 폴더를 사용한다.



하지만 실제 생산 배치 정보 또는 동일 제품 식별 정보는 현재 분석에 포함되어 있지 않다.



따라서 다음 사항은 완전히 검증하지 못했다.



\* 동일 제품의 Train·Test 중복

\* 동일 생산 배치의 Train·Test 중복

\* 증강 원본과 파생 이미지의 분리 여부



README에서 현재 구현 결과와 데이터셋 한계를 구분하여 작성한다.



\---



\## 20. 면접 질문·답변



\### Q1. 왜 모델부터 만들지 않고 이미지 데이터를 먼저 분석했나요?



모델 입력 조건과 데이터 품질을 확인하지 않고 학습을 시작하면 이미지 크기, 채널, 손상 파일, 클래스 불균형 등의 문제를 학습 이후에 발견할 수 있습니다.



그래서 데이터 구조, 클래스 분포, 이미지 크기, 채널, 손상 여부를 먼저 확인한 뒤 Dataset과 모델 입력 조건을 결정했습니다.



\---



\### Q2. 정상은 왜 0이고 불량은 왜 1인가요?



불량을 Positive Class로 해석하기 위해 `DEFECT=1`로 설정했습니다.



이 기준을 사용하면 Precision, Recall, F1 Score와 False Negative를 불량 검출 관점에서 해석할 수 있습니다.



\---



\### Q3. 현재 클래스 불균형은 어떻게 처리했나요?



전체 데이터는 NORMAL 약 42.69%, DEFECT 약 57.31%였습니다.



현재는 극단적인 불균형으로 판단하지 않았기 때문에 클래스 가중치나 Sampler를 바로 적용하지 않았습니다.



먼저 모델의 클래스별 Precision, Recall, F1과 Confusion Matrix를 확인한 뒤 필요하면 학습 전략을 변경할 계획입니다.



\---



\### Q4. 이미지가 모두 같은 크기인데 Resize가 필요한가요?



현재 데이터는 모두 `300 × 300`이지만 모델 입력 규칙을 명시적으로 고정하기 위해 Resize를 적용할 예정입니다.



CNN Baseline과 ResNet18을 동일한 입력 조건으로 비교하고, 이후 외부 이미지 추론에서도 같은 전처리를 재사용할 수 있습니다.



\---



\### Q5. 손상 이미지가 0장인데 왜 손상 이미지 예외 처리를 테스트했나요?



현재 데이터셋이 정상이라고 해서 이후 추가되는 데이터나 API 업로드 이미지도 항상 정상이라고 보장할 수 없습니다.



그래서 정상 데이터만 확인하는 것이 아니라 인위적으로 손상된 JPEG를 생성하여 예외 처리를 테스트했습니다.



\---



\### Q6. 왜 `casting\_512x512` 데이터를 함께 사용하지 않았나요?



기존 `casting\_data`에 이미 충분한 Train·Test 데이터가 있고, `casting\_512x512`가 동일 데이터의 별도 크기 또는 가공 버전일 가능성이 있기 때문입니다.



두 데이터를 무조건 합치면 중복 또는 데이터 누수 위험이 커질 수 있으므로 현재는 명확한 Train·Test 구조가 있는 데이터만 사용했습니다.



\---



\## 21. Day 1 최종 결론



Day 1에서 다음을 완료했다.



```text

프로젝트 생성



→ 완료





Python 3.11.9 가상환경



→ 완료





기본 프로젝트 구조



→ 완료





.gitignore



→ 완료





requirements.txt



→ 완료





데이터셋 확정



→ 완료





Train·Test 구조 확인



→ 완료





정상·불량 이미지 수



→ 완료





클래스 비율



→ 완료





이미지 확장자



→ 완료





이미지 크기



→ 완료





RGB·채널



→ 완료





손상 이미지



→ 완료





분석 CSV



→ 완료





분석 JSON



→ 완료





클래스 분포 그래프



→ 완료





샘플 이미지 Grid



→ 완료





자동 테스트



→ 15 passed

```



Day 2에서는 다음 기능을 구현한다.



```text

Train·Validation 분리



→ Dataset



→ \_\_len\_\_



→ \_\_getitem\_\_



→ Train Transform



→ Validation Transform



→ Test Transform



→ Resize



→ ToTensor



→ Normalize



→ DataLoader



→ Batch 확인



→ Tensor Shape 확인

```



