"""
CNN Baseline model for binary casting-product defect classification.

이 모듈의 역할
---------------
Casting Product Image Data for Quality Inspection 데이터셋의 RGB 이미지를
입력받아 NORMAL 또는 DEFECT 분류에 사용할 Binary Raw Logit을 출력한다.

현재 프로젝트의 클래스 정의
--------------------------
0 = NORMAL
1 = DEFECT

따라서 모델이 출력한 Logit에 Sigmoid를 적용하면
DEFECT 클래스일 확률을 얻을 수 있다.

입력
----
images:
    Shape: [batch_size, 3, height, width]

현재 공식 이미지 Pipeline에서는 다음 Shape를 사용한다.

    [batch_size, 3, 224, 224]

출력
----
logits:
    Shape: [batch_size]

각 이미지마다 Binary Raw Logit 하나를 반환한다.

중요
----
이 모델 내부에서는 Sigmoid를 적용하지 않는다.

향후 학습 단계에서는 다음 구조를 사용한다.

    Raw Logit
    -> BCEWithLogitsLoss

향후 추론 단계에서 확률이 필요할 때만 다음 계산을 수행한다.

    DEFECT Probability
    = torch.sigmoid(logit)
"""

from torch import Tensor, nn


class CNNBaseline(nn.Module):
    """
    정상·불량 이미지 이진 분류를 위한 경량 CNN Baseline 모델.

    왜 필요한가
    -----------
    이후 구현할 ResNet18 전이학습 모델과 성능을 비교하기 전에,
    구조가 단순하고 계산량이 작은 기준 모델이 필요하다.

    이 모델은 다음 목적을 가진다.

    1. Dataset과 DataLoader가 정상적으로 동작하는지 확인한다.
    2. 이미지에 정상·불량을 구분할 학습 신호가 있는지 확인한다.
    3. Loss가 정상적으로 감소하는지 확인한다.
    4. Validation 성능의 기준값을 만든다.
    5. 이후 ResNet18의 성능 향상 정도를 비교한다.

    입력
    ----
    images:
        RGB 이미지 Batch Tensor

        공식 입력 Shape:
            [batch_size, 3, 224, 224]

    처리 과정
    ---------
    Input
    -> Conv1
    -> ReLU1
    -> MaxPool1
    -> Conv2
    -> ReLU2
    -> MaxPool2
    -> Conv3
    -> ReLU3
    -> MaxPool3
    -> Adaptive Average Pooling
    -> Flatten
    -> Linear Classifier

    출력
    ----
    logits:
        Shape:
            [batch_size]

        값의 의미:
            Binary Raw Logit

        주의:
            아직 확률이 아니다.

    호출 관계
    ---------
    향후 Training Pipeline
    -> CNNBaseline.forward()
    -> Raw Logit
    -> BCEWithLogitsLoss

    향후 Inference Pipeline
    -> CNNBaseline.forward()
    -> Raw Logit
    -> Sigmoid
    -> DEFECT Probability
    -> Threshold
    -> NORMAL 또는 DEFECT

    설계 이유
    ---------
    Channel 수:
        3 -> 8 -> 16 -> 32

        CPU 전용 환경에서 계산량을 제한하면서
        단계적으로 더 다양한 이미지 특징을 학습하기 위한 구조다.

    공간 크기:
        224 -> 112 -> 56 -> 28

        Max Pooling으로 공간 크기를 줄여
        후속 Convolution 계산량과 메모리 사용량을 감소시킨다.

    Adaptive Average Pooling:
        각 Channel의 공간 정보를 대표값 하나로 압축한다.

        대형 Fully Connected Layer를 만들지 않으므로
        Parameter 수와 과적합 위험을 줄일 수 있다.

    Binary Logit:
        마지막 Linear Layer는 이미지 한 장마다
        하나의 Raw Logit을 출력한다.

        모델 내부에 Sigmoid를 넣지 않아
        향후 BCEWithLogitsLoss를 안정적으로 사용할 수 있다.
    """

    def __init__(self) -> None:
        """
        CNN Baseline을 구성하는 Layer를 생성한다.

        입력
        ----
        없음

        처리 과정
        ---------
        1. 첫 번째 Convolution Layer 생성
        2. 첫 번째 ReLU와 Max Pooling 생성
        3. 두 번째 Convolution Layer 생성
        4. 두 번째 ReLU와 Max Pooling 생성
        5. 세 번째 Convolution Layer 생성
        6. 세 번째 ReLU와 Max Pooling 생성
        7. Adaptive Average Pooling 생성
        8. Binary Classifier 생성

        출력
        ----
        직접 반환값은 없다.

        생성한 Layer는 CNNBaseline 인스턴스 내부에 저장된다.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Convolution Block 1
        # ------------------------------------------------------------------
        #
        # 입력 Shape:
        #     [B, 3, 224, 224]
        #
        # 출력 Shape:
        #     [B, 8, 224, 224]
        #
        # RGB 이미지이므로 입력 Channel은 3이다.
        #
        # 첫 번째 Layer에서는 Channel을 8개로 확장하여
        # 경계, 밝기 변화, 단순한 표면 질감과 같은
        # 기본적인 이미지 특징을 학습하도록 한다.
        #
        # kernel_size=3:
        #     3×3 크기의 지역 영역을 관찰한다.
        #
        # stride=1:
        #     Kernel을 한 Pixel씩 이동한다.
        #
        # padding=1:
        #     Convolution 전후의 가로·세로 크기를 유지한다.
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # ReLU는 음수 값을 0으로 변경하고
        # 모델이 비선형 패턴을 학습할 수 있도록 한다.
        #
        # inplace=False를 명시한다.
        #
        # 현재 프로젝트에는 향후 Grad-CAM 구현 계획이 있다.
        # In-place 연산은 일부 Hook 또는 역전파 기반 해석 과정에서
        # 원인 파악이 어려운 문제를 만들 수 있으므로 사용하지 않는다.
        self.relu1 = nn.ReLU(
            inplace=False,
        )

        # 입력 Shape:
        #     [B, 8, 224, 224]
        #
        # 출력 Shape:
        #     [B, 8, 112, 112]
        #
        # 2×2 영역마다 가장 큰 특징 값을 유지하면서
        # 가로와 세로 크기를 각각 절반으로 감소시킨다.
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # ------------------------------------------------------------------
        # Convolution Block 2
        # ------------------------------------------------------------------
        #
        # 입력 Shape:
        #     [B, 8, 112, 112]
        #
        # 출력 Shape:
        #     [B, 16, 112, 112]
        #
        # Channel 수를 8개에서 16개로 확장한다.
        #
        # 첫 번째 Block에서 추출한 단순 특징을 조합하여
        # 더 복잡한 표면 패턴과 결함 형태를 학습하도록 한다.
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.relu2 = nn.ReLU(
            inplace=False,
        )

        # 입력 Shape:
        #     [B, 16, 112, 112]
        #
        # 출력 Shape:
        #     [B, 16, 56, 56]
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # ------------------------------------------------------------------
        # Convolution Block 3
        # ------------------------------------------------------------------
        #
        # 입력 Shape:
        #     [B, 16, 56, 56]
        #
        # 출력 Shape:
        #     [B, 32, 56, 56]
        #
        # Channel 수를 16개에서 32개로 확장한다.
        #
        # 앞 Layer에서 추출한 특징을 더 복합적으로 조합하여
        # 정상 표면과 불량 표면을 구분하는 고수준 특징을
        # 학습하도록 한다.
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.relu3 = nn.ReLU(
            inplace=False,
        )

        # 입력 Shape:
        #     [B, 32, 56, 56]
        #
        # 출력 Shape:
        #     [B, 32, 28, 28]
        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # ------------------------------------------------------------------
        # Global Feature Aggregation
        # ------------------------------------------------------------------
        #
        # 공식 입력 크기 224×224를 사용하면
        # 세 번의 Max Pooling 후 Shape는 다음과 같다.
        #
        #     [B, 32, 28, 28]
        #
        # Adaptive Average Pooling은 각 Channel의
        # 28×28 특징을 평균값 하나로 압축한다.
        #
        # 출력 Shape:
        #     [B, 32, 1, 1]
        #
        # 이 구조를 사용하면 32×28×28 전체를
        # 대형 Fully Connected Layer에 직접 연결할 필요가 없다.
        self.global_average_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1),
        )

        # ------------------------------------------------------------------
        # Binary Classifier
        # ------------------------------------------------------------------
        #
        # Adaptive Average Pooling 후 Flatten하면
        # 이미지 한 장은 32개의 대표 특징으로 표현된다.
        #
        # 입력 Shape:
        #     [B, 32]
        #
        # 출력 Shape:
        #     [B, 1]
        #
        # 정상·불량 이진 분류이므로 이미지 한 장마다
        # Raw Logit 하나만 출력한다.
        self.classifier = nn.Linear(
            in_features=32,
            out_features=1,
        )

    def forward(self, images: Tensor) -> Tensor:
        """
        이미지 Batch를 입력받아 Binary Raw Logit을 반환한다.

        입력
        ----
        images:
            RGB 이미지 Tensor

            공식 Shape:
                [batch_size, 3, 224, 224]

            dtype:
                Floating Point Tensor

            현재 Dataset Pipeline의 실제 dtype:
                torch.float32

        처리 과정
        ---------
        1. 입력 Tensor의 기본 형식을 검증한다.
        2. 첫 번째 Convolution Block을 통과한다.
        3. 두 번째 Convolution Block을 통과한다.
        4. 세 번째 Convolution Block을 통과한다.
        5. Adaptive Average Pooling으로 특징을 압축한다.
        6. Feature Tensor를 2차원으로 변환한다.
        7. Linear Layer에서 Binary Raw Logit을 생성한다.
        8. 마지막 크기 1인 차원만 제거한다.

        출력
        ----
        logits:
            Shape:
                [batch_size]

            의미:
                DEFECT 클래스에 대한 Binary Raw Logit

            주의:
                Sigmoid Probability가 아니다.

        예외 처리
        ---------
        Tensor가 아닌 입력:
            TypeError

        4차원이 아닌 입력:
            ValueError

        RGB 3 Channel이 아닌 입력:
            ValueError

        Floating Point가 아닌 입력:
            TypeError

        빈 Batch:
            ValueError

        너무 작은 공간 크기:
            ValueError
        """
        self._validate_input(images=images)

        # --------------------------------------------------------------
        # Convolution Block 1
        # --------------------------------------------------------------
        #
        # [B, 3, 224, 224]
        # -> [B, 8, 224, 224]
        images = self.conv1(images)

        # Shape 유지:
        # [B, 8, 224, 224]
        images = self.relu1(images)

        # [B, 8, 224, 224]
        # -> [B, 8, 112, 112]
        images = self.pool1(images)

        # --------------------------------------------------------------
        # Convolution Block 2
        # --------------------------------------------------------------
        #
        # [B, 8, 112, 112]
        # -> [B, 16, 112, 112]
        images = self.conv2(images)

        # Shape 유지:
        # [B, 16, 112, 112]
        images = self.relu2(images)

        # [B, 16, 112, 112]
        # -> [B, 16, 56, 56]
        images = self.pool2(images)

        # --------------------------------------------------------------
        # Convolution Block 3
        # --------------------------------------------------------------
        #
        # [B, 16, 56, 56]
        # -> [B, 32, 56, 56]
        images = self.conv3(images)

        # Shape 유지:
        # [B, 32, 56, 56]
        images = self.relu3(images)

        # [B, 32, 56, 56]
        # -> [B, 32, 28, 28]
        images = self.pool3(images)

        # --------------------------------------------------------------
        # Adaptive Average Pooling
        # --------------------------------------------------------------
        #
        # [B, 32, 28, 28]
        # -> [B, 32, 1, 1]
        features = self.global_average_pool(images)

        # --------------------------------------------------------------
        # Flatten
        # --------------------------------------------------------------
        #
        # start_dim=1을 사용하여 Batch 차원은 유지하고
        # Channel과 공간 차원만 하나의 Feature 차원으로 합친다.
        #
        # [B, 32, 1, 1]
        # -> [B, 32]
        features = features.flatten(
            start_dim=1,
        )

        # --------------------------------------------------------------
        # Binary Classifier
        # --------------------------------------------------------------
        #
        # [B, 32]
        # -> [B, 1]
        logits = self.classifier(features)

        # --------------------------------------------------------------
        # Output Shape Normalization
        # --------------------------------------------------------------
        #
        # [B, 1]
        # -> [B]
        #
        # squeeze()가 아니라 squeeze(dim=1)을 사용한다.
        #
        # 일반 squeeze()는 Batch Size가 1일 때
        # Batch 차원까지 제거할 수 있다.
        #
        # squeeze(dim=1)은 두 번째 차원만 제거하므로
        # Batch Size가 1이어도 결과 Shape [1]을 유지한다.
        logits = logits.squeeze(
            dim=1,
        )

        return logits

    @staticmethod
    def _validate_input(images: Tensor) -> None:
        """
        Forward 입력 Tensor의 기본 형식을 검증한다.

        왜 필요한가
        -----------
        잘못된 입력이 Convolution Layer 내부까지 전달되면
        초보자가 원인을 이해하기 어려운 PyTorch 오류가 발생할 수 있다.

        모델 진입 시점에 입력 형식을 확인하면
        오류 원인을 더 빠르고 명확하게 찾을 수 있다.

        입력
        ----
        images:
            검증할 이미지 Batch

        처리 과정
        ---------
        1. Tensor 여부 확인
        2. 4차원 여부 확인
        3. 빈 Batch 여부 확인
        4. RGB 3 Channel 여부 확인
        5. Floating Point dtype 여부 확인
        6. 세 번의 Pooling이 가능한 공간 크기인지 확인

        출력
        ----
        정상 입력:
            반환값 없음

        잘못된 입력:
            TypeError 또는 ValueError 발생

        실무 포인트
        -----------
        NaN과 inf 전체 검사는 매 Forward마다 수행하면
        학습 속도에 영향을 줄 수 있으므로 모델 내부에서 반복하지 않는다.

        현재 프로젝트에서는 Dataset, DataLoader 검증과
        학습 Pipeline 검증 단계에서 별도로 확인한다.
        """
        if not isinstance(images, Tensor):
            raise TypeError(
                "images must be a torch.Tensor. "
                f"Received type: {type(images).__name__}."
            )

        if images.ndim != 4:
            raise ValueError(
                "images must have 4 dimensions in "
                "[batch_size, channels, height, width] format. "
                f"Received shape: {tuple(images.shape)}."
            )

        batch_size = images.shape[0]

        if batch_size == 0:
            raise ValueError(
                "images must contain at least one image. "
                f"Received shape: {tuple(images.shape)}."
            )

        channel_count = images.shape[1]

        if channel_count != 3:
            raise ValueError(
                "images must contain 3 RGB channels. "
                f"Received channel count: {channel_count}. "
                f"Received shape: {tuple(images.shape)}."
            )

        if not images.is_floating_point():
            raise TypeError(
                "images must use a floating-point dtype because "
                "the Dataset Transform returns normalized floating-point tensors. "
                f"Received dtype: {images.dtype}."
            )

        image_height = images.shape[2]
        image_width = images.shape[3]

        # MaxPool2d를 세 번 적용하면 공간 크기가 세 번 절반으로 감소한다.
        #
        # 최소 8×8이면 다음 흐름이 가능하다.
        #
        # 8
        # -> 4
        # -> 2
        # -> 1
        #
        # 공식 Pipeline은 224×224를 사용하므로
        # 실제 학습에서는 이 제한보다 충분히 크다.
        minimum_spatial_size = 8

        if (
            image_height < minimum_spatial_size
            or image_width < minimum_spatial_size
        ):
            raise ValueError(
                "image height and width must both be at least "
                f"{minimum_spatial_size} pixels because the model applies "
                "MaxPool2d three times. "
                f"Received height: {image_height}, "
                f"width: {image_width}."
            )