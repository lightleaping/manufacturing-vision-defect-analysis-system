"""
CNN Baseline model unit tests.

테스트 대상
----------
src/models/cnn_baseline.py

테스트 목적
----------
CNNBaseline이 설계한 Layer 구조와 Forward 흐름을 정확히 구현하는지
자동으로 검증한다.

현재 클래스 정의
---------------
0 = NORMAL
1 = DEFECT

Positive Class:
    DEFECT

모델 출력
---------
Binary Raw Logit

공식 입력 Shape
---------------
[batch_size, 3, 224, 224]

공식 출력 Shape
---------------
[batch_size]

주의
----
현재 단계에서는 Loss Function, Optimizer, Training Loop를 테스트하지 않는다.

이번 테스트 범위는 CNN 모델 자체로 제한한다.
"""

import pytest
import torch
from torch import Tensor, nn

from src.models.cnn_baseline import CNNBaseline


# =============================================================================
# Model Creation
# =============================================================================


@pytest.fixture
def model() -> CNNBaseline:
    """
    각 테스트에서 사용할 새로운 CNNBaseline 인스턴스를 생성한다.

    왜 필요한가
    -----------
    같은 모델 객체를 여러 테스트가 공유하면 한 테스트에서 변경한
    train/eval 상태나 Gradient가 다른 테스트에 영향을 줄 수 있다.

    각 테스트마다 새로운 모델을 생성하면 테스트 간 독립성을 유지할 수 있다.

    입력
    ----
    없음

    출력
    ----
    CNNBaseline:
        새롭게 생성한 CNN Baseline 모델
    """
    return CNNBaseline()


# =============================================================================
# Model Type and Default State
# =============================================================================


def test_cnn_baseline_is_torch_module(
    model: CNNBaseline,
) -> None:
    """
    CNNBaseline이 PyTorch nn.Module을 상속하는지 확인한다.
    """
    assert isinstance(
        model,
        nn.Module,
    )


def test_cnn_baseline_starts_in_training_mode(
    model: CNNBaseline,
) -> None:
    """
    PyTorch 모델 생성 직후 기본 상태가 Training Mode인지 확인한다.
    """
    assert model.training is True


def test_eval_changes_model_to_evaluation_mode(
    model: CNNBaseline,
) -> None:
    """
    model.eval() 호출 후 Evaluation Mode로 변경되는지 확인한다.
    """
    model.eval()

    assert model.training is False


# =============================================================================
# Convolution Layer Structure
# =============================================================================


def test_first_convolution_layer_configuration(
    model: CNNBaseline,
) -> None:
    """
    첫 번째 Convolution Layer 설정을 확인한다.

    예상 구조
    ---------
    RGB 3 Channel
    -> Feature 8 Channel
    """
    assert isinstance(
        model.conv1,
        nn.Conv2d,
    )

    assert model.conv1.in_channels == 3
    assert model.conv1.out_channels == 8

    assert model.conv1.kernel_size == (
        3,
        3,
    )

    assert model.conv1.stride == (
        1,
        1,
    )

    assert model.conv1.padding == (
        1,
        1,
    )


def test_second_convolution_layer_configuration(
    model: CNNBaseline,
) -> None:
    """
    두 번째 Convolution Layer 설정을 확인한다.

    예상 구조
    ---------
    Feature 8 Channel
    -> Feature 16 Channel
    """
    assert isinstance(
        model.conv2,
        nn.Conv2d,
    )

    assert model.conv2.in_channels == 8
    assert model.conv2.out_channels == 16

    assert model.conv2.kernel_size == (
        3,
        3,
    )

    assert model.conv2.stride == (
        1,
        1,
    )

    assert model.conv2.padding == (
        1,
        1,
    )


def test_third_convolution_layer_configuration(
    model: CNNBaseline,
) -> None:
    """
    세 번째 Convolution Layer 설정을 확인한다.

    예상 구조
    ---------
    Feature 16 Channel
    -> Feature 32 Channel
    """
    assert isinstance(
        model.conv3,
        nn.Conv2d,
    )

    assert model.conv3.in_channels == 16
    assert model.conv3.out_channels == 32

    assert model.conv3.kernel_size == (
        3,
        3,
    )

    assert model.conv3.stride == (
        1,
        1,
    )

    assert model.conv3.padding == (
        1,
        1,
    )


# =============================================================================
# ReLU Structure
# =============================================================================


@pytest.mark.parametrize(
    "layer_name",
    [
        "relu1",
        "relu2",
        "relu3",
    ],
)
def test_relu_layers_do_not_use_inplace_operation(
    model: CNNBaseline,
    layer_name: str,
) -> None:
    """
    모든 ReLU Layer가 inplace=False인지 확인한다.

    이유
    ----
    향후 Grad-CAM Hook과 역전파 기반 해석에서
    In-place 연산으로 인한 문제 가능성을 줄이기 위함이다.
    """
    relu_layer = getattr(
        model,
        layer_name,
    )

    assert isinstance(
        relu_layer,
        nn.ReLU,
    )

    assert relu_layer.inplace is False


# =============================================================================
# Max Pooling Structure
# =============================================================================


@pytest.mark.parametrize(
    "layer_name",
    [
        "pool1",
        "pool2",
        "pool3",
    ],
)
def test_max_pooling_layer_configuration(
    model: CNNBaseline,
    layer_name: str,
) -> None:
    """
    모든 Max Pooling Layer가 공간 크기를 절반으로 줄이는 설정인지 확인한다.
    """
    pooling_layer = getattr(
        model,
        layer_name,
    )

    assert isinstance(
        pooling_layer,
        nn.MaxPool2d,
    )

    assert pooling_layer.kernel_size == 2
    assert pooling_layer.stride == 2


# =============================================================================
# Global Average Pooling and Classifier
# =============================================================================


def test_global_average_pool_configuration(
    model: CNNBaseline,
) -> None:
    """
    Adaptive Average Pooling 출력 크기가 1×1인지 확인한다.
    """
    assert isinstance(
        model.global_average_pool,
        nn.AdaptiveAvgPool2d,
    )

    assert model.global_average_pool.output_size == (
        1,
        1,
    )


def test_binary_classifier_configuration(
    model: CNNBaseline,
) -> None:
    """
    마지막 Classifier가 32개 특징을 Binary Logit 하나로 변환하는지 확인한다.
    """
    assert isinstance(
        model.classifier,
        nn.Linear,
    )

    assert model.classifier.in_features == 32
    assert model.classifier.out_features == 1

    assert model.classifier.bias is not None


def test_model_does_not_contain_sigmoid_layer(
    model: CNNBaseline,
) -> None:
    """
    모델 내부에 Sigmoid Layer가 없는지 확인한다.

    이유
    ----
    향후 BCEWithLogitsLoss를 사용할 예정이므로
    모델은 Sigmoid Probability가 아니라 Raw Logit을 반환해야 한다.
    """
    sigmoid_layers = [
        module
        for module in model.modules()
        if isinstance(
            module,
            nn.Sigmoid,
        )
    ]

    assert sigmoid_layers == []


# =============================================================================
# Parameter Count
# =============================================================================


def test_total_parameter_count(
    model: CNNBaseline,
) -> None:
    """
    CNN Baseline 전체 Parameter 수가 설계값과 같은지 확인한다.
    """
    total_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert total_parameter_count == 6_065


def test_all_parameters_are_trainable(
    model: CNNBaseline,
) -> None:
    """
    현재 CNN Baseline의 모든 Parameter가 학습 가능한 상태인지 확인한다.
    """
    total_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert trainable_parameter_count == total_parameter_count

    assert trainable_parameter_count == 6_065


# =============================================================================
# Official Input and Output Shape
# =============================================================================


def test_forward_returns_one_logit_per_image_for_official_input(
    model: CNNBaseline,
) -> None:
    """
    공식 입력 Shape를 사용했을 때 이미지마다 Logit 하나를 반환하는지 확인한다.

    입력
    ----
    [32, 3, 224, 224]

    예상 출력
    ---------
    [32]
    """
    images = torch.randn(
        32,
        3,
        224,
        224,
        dtype=torch.float32,
    )

    logits = model(images)

    assert isinstance(
        logits,
        Tensor,
    )

    assert logits.shape == (
        32,
    )


def test_forward_preserves_single_image_batch_dimension(
    model: CNNBaseline,
) -> None:
    """
    Batch Size가 1이어도 Batch 차원이 유지되는지 확인한다.

    향후 FastAPI 단일 이미지 추론에서 중요하다.
    """
    single_image = torch.randn(
        1,
        3,
        224,
        224,
        dtype=torch.float32,
    )

    single_logit = model(single_image)

    assert single_logit.shape == (
        1,
    )


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        7,
    ],
)
def test_forward_supports_different_batch_sizes(
    model: CNNBaseline,
    batch_size: int,
) -> None:
    """
    여러 Batch Size에서 이미지 수와 Logit 수가 일치하는지 확인한다.
    """
    images = torch.randn(
        batch_size,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    logits = model(images)

    assert logits.shape == (
        batch_size,
    )


def test_forward_supports_valid_rectangular_image_size(
    model: CNNBaseline,
) -> None:
    """
    Adaptive Average Pooling 구조가 유효한 직사각형 입력도 처리하는지 확인한다.

    공식 학습 Pipeline은 224×224를 사용하지만,
    모델 자체는 특정 공간 크기에 불필요하게 고정하지 않는다.
    """
    images = torch.randn(
        2,
        3,
        64,
        80,
        dtype=torch.float32,
    )

    logits = model(images)

    assert logits.shape == (
        2,
    )


def test_forward_supports_minimum_valid_spatial_size(
    model: CNNBaseline,
) -> None:
    """
    세 번의 Max Pooling이 가능한 최소 8×8 입력을 처리하는지 확인한다.

    공간 크기 흐름
    ------------
    8
    -> 4
    -> 2
    -> 1
    """
    images = torch.randn(
        2,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    logits = model(images)

    assert logits.shape == (
        2,
    )


# =============================================================================
# Output Value Validation
# =============================================================================


def test_forward_output_dtype_is_float32(
    model: CNNBaseline,
) -> None:
    """
    float32 이미지 입력의 Logit도 float32인지 확인한다.
    """
    images = torch.randn(
        4,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    logits = model(images)

    assert logits.dtype == torch.float32


def test_forward_output_contains_only_finite_values(
    model: CNNBaseline,
) -> None:
    """
    Forward 결과에 NaN 또는 inf가 없는지 확인한다.
    """
    images = torch.randn(
        4,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    logits = model(images)

    assert torch.isfinite(
        logits,
    ).all()


def test_sigmoid_probabilities_are_between_zero_and_one(
    model: CNNBaseline,
) -> None:
    """
    Raw Logit에 Sigmoid를 적용한 결과가 확률 범위에 있는지 확인한다.
    """
    images = torch.randn(
        4,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    logits = model(images)

    probabilities = torch.sigmoid(
        logits,
    )

    assert probabilities.shape == (
        4,
    )

    assert torch.all(
        probabilities >= 0.0,
    )

    assert torch.all(
        probabilities <= 1.0,
    )


def test_forward_does_not_modify_original_input_tensor(
    model: CNNBaseline,
) -> None:
    """
    Forward 실행이 원본 입력 Tensor를 직접 변경하지 않는지 확인한다.
    """
    images = torch.randn(
        2,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    original_images = images.clone()

    _ = model(images)

    assert torch.equal(
        images,
        original_images,
    )


def test_repeated_evaluation_forward_is_deterministic(
    model: CNNBaseline,
) -> None:
    """
    같은 Weight와 같은 입력에서 Evaluation Forward 결과가 같은지 확인한다.

    현재 모델에는 Dropout과 Random Layer가 없으므로
    같은 입력은 같은 출력을 반환해야 한다.
    """
    model.eval()

    images = torch.randn(
        2,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        first_logits = model(images)
        second_logits = model(images)

    assert torch.equal(
        first_logits,
        second_logits,
    )


# =============================================================================
# Gradient Flow
# =============================================================================


def test_backward_creates_finite_gradients_for_trainable_parameters(
    model: CNNBaseline,
) -> None:
    """
    모델 출력에서 역전파가 가능한지 확인한다.

    현재는 실제 Loss Function을 구현하지 않았으므로,
    Logit 합계를 임시 Scalar 값으로 사용해 Gradient 흐름만 검증한다.
    """
    images = torch.randn(
        2,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    logits = model(images)

    temporary_scalar = logits.sum()

    temporary_scalar.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None

        assert torch.isfinite(
            parameter.grad,
        ).all()


# =============================================================================
# Invalid Input Validation
# =============================================================================


def test_forward_rejects_non_tensor_input(
    model: CNNBaseline,
) -> None:
    """
    Tensor가 아닌 입력을 거부하는지 확인한다.
    """
    invalid_images = [
        0.0,
    ]

    with pytest.raises(
        TypeError,
        match="images must be a torch.Tensor",
    ):
        model(invalid_images)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_images",
    [
        torch.randn(
            3,
            224,
            224,
        ),
        torch.randn(
            1,
            1,
            3,
            224,
            224,
        ),
    ],
)
def test_forward_rejects_input_that_is_not_four_dimensional(
    model: CNNBaseline,
    invalid_images: Tensor,
) -> None:
    """
    [B, C, H, W] 형식이 아닌 Tensor를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="images must have 4 dimensions",
    ):
        model(invalid_images)


def test_forward_rejects_empty_batch(
    model: CNNBaseline,
) -> None:
    """
    이미지가 한 장도 없는 빈 Batch를 거부하는지 확인한다.
    """
    empty_images = torch.empty(
        0,
        3,
        224,
        224,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="at least one image",
    ):
        model(empty_images)


@pytest.mark.parametrize(
    "channel_count",
    [
        1,
        2,
        4,
    ],
)
def test_forward_rejects_non_rgb_channel_count(
    model: CNNBaseline,
    channel_count: int,
) -> None:
    """
    RGB 3 Channel이 아닌 이미지 Tensor를 거부하는지 확인한다.
    """
    invalid_images = torch.randn(
        2,
        channel_count,
        32,
        32,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="3 RGB channels",
    ):
        model(invalid_images)


def test_forward_rejects_integer_image_tensor(
    model: CNNBaseline,
) -> None:
    """
    정수 dtype 이미지 Tensor를 거부하는지 확인한다.

    현재 Dataset Transform은 Normalize된 float32 Tensor를 반환한다.
    """
    integer_images = torch.zeros(
        2,
        3,
        32,
        32,
        dtype=torch.int64,
    )

    with pytest.raises(
        TypeError,
        match="floating-point dtype",
    ):
        model(integer_images)


def test_forward_rejects_image_height_smaller_than_eight(
    model: CNNBaseline,
) -> None:
    """
    세 번의 Max Pooling이 불가능한 높이를 거부하는지 확인한다.
    """
    invalid_images = torch.randn(
        2,
        3,
        7,
        8,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="at least 8 pixels",
    ):
        model(invalid_images)


def test_forward_rejects_image_width_smaller_than_eight(
    model: CNNBaseline,
) -> None:
    """
    세 번의 Max Pooling이 불가능한 너비를 거부하는지 확인한다.
    """
    invalid_images = torch.randn(
        2,
        3,
        8,
        7,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="at least 8 pixels",
    ):
        model(invalid_images)