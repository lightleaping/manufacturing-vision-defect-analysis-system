"""
Manufacturing Vision Defect Analysis System

Dataset configuration module.

이 파일은 Casting Product 이미지 데이터셋의 경로와
정상·불량 클래스 규칙을 한 곳에서 관리한다.

주요 역할
---------
1. 프로젝트 루트 경로 계산
2. 원본 데이터셋 경로 정의
3. Train·Test 경로 정의
4. 원본 폴더 이름과 숫자 Label 연결
5. 프로젝트에서 사용할 클래스 이름 정의

이 파일을 별도로 두는 이유
-------------------------
여러 파일에서 데이터 경로 문자열을 직접 작성하면
경로가 변경될 때 모든 파일을 수정해야 한다.

경로와 클래스 규칙을 이 파일에서 한 번만 정의하면
Dataset, 데이터 분석, 학습, 평가 코드가
동일한 기준을 재사용할 수 있다.
"""

from pathlib import Path


# -----------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------

# 현재 파일 위치:
#
# manufacturing-vision-defect-analysis-system/
# └── src/
#     └── data/
#         └── dataset_config.py
#
# parents[0] -> src/data
# parents[1] -> src
# parents[2] -> 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Kaggle에서 다운로드한 전체 압축 해제 폴더이다.
RAW_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "casting_product_images"
)


# 실제 모델 학습과 평가에 사용할 데이터 루트이다.
#
# 실제 구조:
#
# casting_data/
# └── casting_data/
#     ├── train/
#     └── test/
DATASET_ROOT = (
    RAW_DATA_ROOT
    / "casting_data"
    / "casting_data"
)


# 모델 학습용 원본 폴더이다.
TRAIN_ROOT = DATASET_ROOT / "train"


# 모델 최종 평가용 원본 폴더이다.
TEST_ROOT = DATASET_ROOT / "test"


# -----------------------------------------------------------------
# Class configuration
# -----------------------------------------------------------------

# 원본 데이터셋 폴더 이름을 숫자 Label로 변환한다.
#
# 정상:
# ok_front -> 0
#
# 불량:
# def_front -> 1
#
# 불량을 1로 설정하면 Precision, Recall, F1 계산 시
# 불량 클래스를 Positive Class로 해석할 수 있다.
CLASS_TO_INDEX = {
    "ok_front": 0,
    "def_front": 1,
}


# 숫자 Label을 프로젝트 사용자용 클래스 이름으로 변환한다.
INDEX_TO_CLASS_NAME = {
    0: "NORMAL",
    1: "DEFECT",
}


# -----------------------------------------------------------------
# Supported image formats
# -----------------------------------------------------------------

# 이미지 분석과 Dataset에서 허용할 파일 확장자이다.
#
# 현재 데이터셋에는 주로 JPEG 이미지가 있지만,
# 이후 입력 검증과 코드 재사용을 고려하여
# 일반적으로 사용하는 이미지 확장자를 함께 정의한다.
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}