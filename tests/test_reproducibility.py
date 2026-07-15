"""
프로젝트 전역 Random Seed·재현성 설정 테스트.

이 테스트 파일의 책임
---------------------
1. 프로젝트 기본 Random Seed 설정을 검증한다.
2. Random Seed 타입과 허용 범위를 검증한다.
3. deterministic_algorithms 설정값을 검증한다.
4. 현재 환경에 맞는 기본 Device 선택을 검증한다.
5. Python·NumPy·PyTorch 난수가 재현되는지 검증한다.
6. 다른 Seed를 사용하면 다른 난수 흐름이 생성되는지 검증한다.
7. PyTorch 결정적 알고리즘 설정을 검증한다.
8. CUDA 사용 가능 환경의 Seed 설정 분기를 검증한다.
9. 재현성 설정 출력 결과를 검증한다.
10. 설정 결과 객체가 불변인지 검증한다.
11. 모듈 main() 실행 결과를 검증한다.

중요
----
Random Seed 고정은 같은 환경에서 재현 가능성을 높인다.

하지만 운영체제·PyTorch 버전·CPU·GPU·CUDA·사용 연산이 달라지면
완전히 동일한 결과를 보장하지는 않는다.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import (
    FrozenInstanceError,
)

import numpy as np
import pytest
import torch

from src.reproducibility import (
    DEFAULT_RANDOM_SEED,
    MAX_RANDOM_SEED,
    ReproducibilitySettings,
    get_default_device_name,
    main,
    print_reproducibility_settings,
    set_global_random_seed,
    validate_deterministic_option,
    validate_random_seed,
)


@pytest.fixture(
    autouse=True,
)
def restore_deterministic_algorithm_state() -> (
    Iterator[None]
):
    """
    각 테스트가 끝난 뒤 PyTorch 결정적 알고리즘 상태를 복원한다.

    필요한 이유
    -----------
    torch.use_deterministic_algorithms()는 전역 설정이다.

    하나의 테스트에서 True로 변경한 상태가 다음 테스트에 남으면
    테스트 실행 순서에 따라 결과가 달라질 수 있다.

    따라서 테스트 시작 시 기존 상태를 저장하고,
    테스트 종료 후 원래 값으로 복원한다.
    """

    original_state = (
        torch
        .are_deterministic_algorithms_enabled()
    )

    yield

    torch.use_deterministic_algorithms(
        original_state
    )


def test_reproducibility_constants_are_expected() -> None:
    """
    프로젝트 기본 Random Seed와 최대 허용값을 검증한다.
    """

    assert (
        DEFAULT_RANDOM_SEED
        == 42
    )

    assert (
        MAX_RANDOM_SEED
        == 2**32 - 1
    )


@pytest.mark.parametrize(
    "valid_seed",
    [
        0,
        MAX_RANDOM_SEED,
    ],
)
def test_validate_random_seed_accepts_boundary_values(
    valid_seed: int,
) -> None:
    """
    Random Seed 허용 범위의 최소·최대 경계값을 허용하는지 검증한다.

    허용 범위:

        0

        ~

        2**32 - 1
    """

    validate_random_seed(
        seed=valid_seed,
    )


@pytest.mark.parametrize(
    "invalid_seed",
    [
        True,
        42.0,
        "42",
    ],
)
def test_validate_random_seed_rejects_non_integer_values(
    invalid_seed: object,
) -> None:
    """
    실제 int가 아닌 Random Seed를 거부하는지 검증한다.

    bool은 Python에서 int의 하위 타입이지만
    명시적인 Random Seed로 허용하지 않는다.
    """

    with pytest.raises(
        TypeError,
        match=(
            "seed는 "
            "int여야 합니다"
        ),
    ):
        validate_random_seed(
            seed=invalid_seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_seed",
    [
        -1,
        MAX_RANDOM_SEED + 1,
    ],
)
def test_validate_random_seed_rejects_out_of_range_values(
    invalid_seed: int,
) -> None:
    """
    Random Seed 허용 범위를 벗어난 값을 거부하는지 검증한다.
    """

    with pytest.raises(
        ValueError,
        match=(
            "seed는 0 이상"
        ),
    ):
        validate_random_seed(
            seed=invalid_seed,
        )


@pytest.mark.parametrize(
    "valid_option",
    [
        True,
        False,
    ],
)
def test_validate_deterministic_option_accepts_boolean_values(
    valid_option: bool,
) -> None:
    """
    deterministic_algorithms에 실제 bool 값을 허용하는지 검증한다.
    """

    validate_deterministic_option(
        deterministic_algorithms=(
            valid_option
        ),
    )


@pytest.mark.parametrize(
    "invalid_option",
    [
        1,
        0,
        "False",
    ],
)
def test_validate_deterministic_option_rejects_non_boolean_values(
    invalid_option: object,
) -> None:
    """
    deterministic_algorithms에 bool이 아닌 값을 거부하는지 검증한다.
    """

    with pytest.raises(
        TypeError,
        match=(
            "deterministic_algorithms는 "
            "bool이어야 합니다"
        ),
    ):
        validate_deterministic_option(
            deterministic_algorithms=(
                invalid_option
            ),  # type: ignore[arg-type]
        )


def test_get_default_device_name_matches_current_environment() -> None:
    """
    현재 CUDA 사용 가능 여부에 맞는 Device 이름을 반환하는지 검증한다.
    """

    expected_device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    assert (
        get_default_device_name()
        == expected_device
    )


def test_set_global_random_seed_returns_expected_settings() -> None:
    """
    전역 Seed 설정 함수가 실제 환경 정보를 포함한 설정 객체를 반환하는지 검증한다.
    """

    settings = (
        set_global_random_seed(
            seed=42,
            deterministic_algorithms=False,
        )
    )

    cuda_available = (
        torch.cuda.is_available()
    )

    expected_device = (
        "cuda"
        if cuda_available
        else "cpu"
    )

    assert isinstance(
        settings,
        ReproducibilitySettings,
    )

    assert settings.seed == 42

    assert (
        settings
        .deterministic_algorithms
        is False
    )

    assert (
        settings.cuda_available
        is cuda_available
    )

    assert (
        settings.cuda_seed_applied
        is cuda_available
    )

    assert (
        settings.device
        == expected_device
    )


def test_same_seed_reproduces_python_numpy_and_torch_random_values() -> None:
    """
    같은 Seed를 다시 설정하면 Python·NumPy·PyTorch 난수가 재현되는지 검증한다.
    """

    set_global_random_seed(
        seed=42,
    )

    first_python_value = (
        random.random()
    )

    first_numpy_value = (
        np.random.random()
    )

    first_torch_value = (
        torch.rand(
            5
        )
    )

    set_global_random_seed(
        seed=42,
    )

    second_python_value = (
        random.random()
    )

    second_numpy_value = (
        np.random.random()
    )

    second_torch_value = (
        torch.rand(
            5
        )
    )

    assert (
        first_python_value
        == second_python_value
    )

    assert (
        first_numpy_value
        == second_numpy_value
    )

    assert torch.equal(
        first_torch_value,
        second_torch_value,
    )


def test_different_seed_changes_random_values() -> None:
    """
    서로 다른 Seed를 사용하면 난수 흐름이 달라지는지 검증한다.
    """

    set_global_random_seed(
        seed=42,
    )

    first_python_value = (
        random.random()
    )

    first_numpy_value = (
        np.random.random()
    )

    first_torch_value = (
        torch.rand(
            3
        )
    )

    set_global_random_seed(
        seed=43,
    )

    second_python_value = (
        random.random()
    )

    second_numpy_value = (
        np.random.random()
    )

    second_torch_value = (
        torch.rand(
            3
        )
    )

    assert (
        first_python_value
        != second_python_value
    )

    assert (
        first_numpy_value
        != second_numpy_value
    )

    assert not torch.equal(
        first_torch_value,
        second_torch_value,
    )


def test_set_global_random_seed_enables_deterministic_algorithms() -> None:
    """
    deterministic_algorithms=True가 PyTorch 전역 설정에 적용되는지 검증한다.
    """

    settings = (
        set_global_random_seed(
            seed=42,
            deterministic_algorithms=True,
        )
    )

    assert (
        settings
        .deterministic_algorithms
        is True
    )

    assert (
        torch
        .are_deterministic_algorithms_enabled()
        is True
    )


def test_set_global_random_seed_disables_deterministic_algorithms() -> None:
    """
    deterministic_algorithms=False가 기존 True 상태를 해제하는지 검증한다.
    """

    torch.use_deterministic_algorithms(
        True
    )

    assert (
        torch
        .are_deterministic_algorithms_enabled()
        is True
    )

    settings = (
        set_global_random_seed(
            seed=42,
            deterministic_algorithms=False,
        )
    )

    assert (
        settings
        .deterministic_algorithms
        is False
    )

    assert (
        torch
        .are_deterministic_algorithms_enabled()
        is False
    )


def test_set_global_random_seed_applies_cuda_seed_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    CUDA 사용 가능 환경에서 CUDA Seed 설정 분기가 실행되는지 검증한다.

    현재 개발 컴퓨터에는 CUDA GPU가 없다.

    따라서 실제 CUDA Device 없이 monkeypatch를 사용하여
    CUDA 사용 가능 환경을 모의한다.
    """

    cpu_seed_calls: list[
        int
    ] = []

    cuda_seed_calls: list[
        int
    ] = []

    def fake_torch_manual_seed(
        seed: int,
    ) -> None:
        """
        torch.manual_seed() 호출값을 기록한다.
        """

        cpu_seed_calls.append(
            seed
        )

    def fake_cuda_manual_seed_all(
        seed: int,
    ) -> None:
        """
        torch.cuda.manual_seed_all() 호출값을 기록한다.
        """

        cuda_seed_calls.append(
            seed
        )

    monkeypatch.setattr(
        torch,
        "manual_seed",
        fake_torch_manual_seed,
    )

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )

    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        fake_cuda_manual_seed_all,
    )

    settings = (
        set_global_random_seed(
            seed=123,
            deterministic_algorithms=False,
        )
    )

    assert cpu_seed_calls == [
        123,
    ]

    assert cuda_seed_calls == [
        123,
    ]

    assert (
        settings.cuda_available
        is True
    )

    assert (
        settings.cuda_seed_applied
        is True
    )

    assert (
        settings.device
        == "cuda"
    )


def test_print_reproducibility_settings_contains_expected_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    재현성 설정 출력에 주요 정보가 포함되는지 검증한다.
    """

    settings = (
        ReproducibilitySettings(
            seed=42,
            deterministic_algorithms=False,
            cuda_available=False,
            cuda_seed_applied=False,
            device="cpu",
        )
    )

    print_reproducibility_settings(
        settings=settings,
    )

    captured_output = (
        capsys
        .readouterr()
        .out
    )

    assert (
        "PROJECT REPRODUCIBILITY SETTINGS"
        in captured_output
    )

    assert (
        "random seed"
        in captured_output
    )

    assert (
        "42"
        in captured_output
    )

    assert (
        "deterministic algorithms"
        in captured_output
    )

    assert (
        "CUDA available"
        in captured_output
    )

    assert (
        "CUDA seed applied"
        in captured_output
    )

    assert (
        "default device"
        in captured_output
    )

    assert (
        "cpu"
        in captured_output
    )


def test_reproducibility_settings_is_immutable() -> None:
    """
    ReproducibilitySettings가 생성 후 수정되지 않는 불변 객체인지 검증한다.
    """

    settings = (
        ReproducibilitySettings(
            seed=42,
            deterministic_algorithms=False,
            cuda_available=False,
            cuda_seed_applied=False,
            device="cpu",
        )
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        settings.seed = 100  # type: ignore[misc]


def test_main_applies_default_settings_and_prints_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    reproducibility 모듈의 main()이 기본 설정을 적용하고 결과를 출력하는지 검증한다.
    """

    main()

    captured_output = (
        capsys
        .readouterr()
        .out
    )

    assert (
        "PROJECT REPRODUCIBILITY SETTINGS"
        in captured_output
    )

    assert (
        "random seed"
        in captured_output
    )

    assert (
        str(
            DEFAULT_RANDOM_SEED
        )
        in captured_output
    )

    assert (
        "default device"
        in captured_output
    )