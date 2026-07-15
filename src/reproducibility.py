"""
프로젝트 전역 Random Seed·재현성 설정 모듈.

이 모듈의 책임
----------------
1. Python random의 Seed를 설정한다.
2. NumPy Random Seed를 설정한다.
3. PyTorch CPU Random Seed를 설정한다.
4. CUDA 사용 가능 환경에서는 모든 CUDA Device의 Seed를 설정한다.
5. 선택적으로 PyTorch 결정적 알고리즘을 활성화한다.
6. 실제 적용된 재현성 설정을 불변 데이터 구조로 반환한다.

사용 예
------
일반 학습:

    settings = set_global_random_seed(
        seed=42,
    )

결정적 알고리즘을 요청하는 실험:

    settings = set_global_random_seed(
        seed=42,
        deterministic_algorithms=True,
    )

중요
----
Random Seed를 고정하면 같은 환경에서 실험 재현 가능성을 높일 수 있다.

하지만 다음 조건이 달라지면 완전히 동일한 결과를 보장할 수는 없다.

    운영체제

    Python 버전

    PyTorch 버전

    CPU·GPU 종류

    CUDA 버전

    병렬 연산 방식

    사용 연산의 구현
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


# 프로젝트 전반에서 사용할 기본 Random Seed다.
#
# Day 1:
#
#     샘플 이미지 선택
#
# Day 2:
#
#     Train·Validation Split
#
#     DataLoader Shuffle
#
# 이후:
#
#     CNN Weight 초기화
#
#     ResNet18 학습
#
#     Dropout
#
#     Random Data Augmentation
DEFAULT_RANDOM_SEED: int = 42


# NumPy의 기존 RandomState Seed가 허용하는 최대값이다.
#
# NumPy random.seed()는 다음 범위의 정수를 사용한다.
#
#     0
#
#     ~
#
#     2**32 - 1
MAX_RANDOM_SEED: int = (
    2**32
    - 1
)


@dataclass(frozen=True)
class ReproducibilitySettings:
    """
    실제 적용된 프로젝트 재현성 설정.

    Attributes
    ----------
    seed:
        Python·NumPy·PyTorch에 적용한 Random Seed다.

    deterministic_algorithms:
        PyTorch 결정적 알고리즘 사용 요청 여부다.

    cuda_available:
        현재 PyTorch 환경에서 CUDA를 사용할 수 있는지 나타낸다.

    cuda_seed_applied:
        CUDA Device Seed 설정을 실제로 실행했는지 나타낸다.

    device:
        현재 환경에서 선택되는 기본 연산 장치 이름이다.

        현재 CPU 환경:

            "cpu"

        CUDA 환경:

            "cuda"

    frozen=True
    -----------
    설정 결과를 생성한 뒤 값이 실수로 변경되지 않도록
    불변 객체로 관리한다.
    """

    seed: int

    deterministic_algorithms: bool

    cuda_available: bool

    cuda_seed_applied: bool

    device: str


def validate_random_seed(
    seed: int,
) -> None:
    """
    Random Seed 값이 프로젝트에서 사용할 수 있는 범위인지 검증한다.

    Parameters
    ----------
    seed:
        Python·NumPy·PyTorch에 적용할 정수 Seed다.

    Raises
    ------
    TypeError
        Seed가 실제 int가 아닐 때 발생한다.

    ValueError
        Seed가 유효 범위를 벗어날 때 발생한다.

    bool 처리
    ---------
    Python에서 bool은 int의 하위 타입이다.

        isinstance(
            True,
            int,
        )

        → True

    그러나 True·False는 명시적인 Random Seed로 사용하지 않는다.

    따라서:

        type(seed) is int

    기준으로 검증한다.
    """

    if type(seed) is not int:
        raise TypeError(
            "seed는 int여야 합니다: "
            f"{type(seed).__name__}"
        )

    if not 0 <= seed <= MAX_RANDOM_SEED:
        raise ValueError(
            "seed는 0 이상 "
            f"{MAX_RANDOM_SEED} 이하여야 합니다: "
            f"{seed}"
        )


def validate_deterministic_option(
    deterministic_algorithms: bool,
) -> None:
    """
    결정적 알고리즘 설정값이 실제 bool인지 검증한다.

    Parameters
    ----------
    deterministic_algorithms:
        PyTorch 결정적 알고리즘 활성화 여부다.

    Raises
    ------
    TypeError
        bool이 아닌 값이 전달될 때 발생한다.
    """

    if type(
        deterministic_algorithms
    ) is not bool:
        raise TypeError(
            "deterministic_algorithms는 "
            "bool이어야 합니다: "
            f"{type(deterministic_algorithms).__name__}"
        )


def get_default_device_name() -> str:
    """
    현재 PyTorch 환경에서 사용할 기본 Device 이름을 반환한다.

    Returns
    -------
    str
        CUDA 사용 가능:

            "cuda"

        CUDA 사용 불가:

            "cpu"

    현재 프로젝트 환경
    ------------------
    현재 컴퓨터:

        Intel Core i5-1035G7

        Intel Iris Plus Graphics

        NVIDIA CUDA GPU 없음

    따라서 현재 결과:

        "cpu"
    """

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def set_global_random_seed(
    seed: int = DEFAULT_RANDOM_SEED,
    deterministic_algorithms: bool = False,
) -> ReproducibilitySettings:
    """
    Python·NumPy·PyTorch 전역 Random Seed를 설정한다.

    Parameters
    ----------
    seed:
        전체 난수 생성기에 적용할 Seed다.

        기본값:

            42

    deterministic_algorithms:
        PyTorch에 결정적 알고리즘 사용을 요청할지 결정한다.

        기본값:

            False

        True:

            가능한 경우 결정적 알고리즘을 사용한다.

            일부 연산 속도가 느려질 수 있다.

            결정적 구현이 없는 연산에서는 오류가 발생할 수 있다.

    Returns
    -------
    ReproducibilitySettings
        실제 적용된 Seed·CUDA·Device 설정 정보다.

    처리 순서
    ---------
    Seed 검증

    → Python random.seed()

    → NumPy random.seed()

    → torch.manual_seed()

    → CUDA 사용 가능 시 torch.cuda.manual_seed_all()

    → 결정적 알고리즘 옵션 적용

    → 설정 결과 반환

    향후 호출 위치
    -------------
    모델과 DataLoader를 생성하기 전에 호출한다.

    예:

        set_global_random_seed(
            seed=42,
        )

        data_loaders = (
            create_vision_data_loaders()
        )

        model = (
            create_cnn_model()
        )

    호출 순서가 중요한 이유
    ---------------------
    모델 Weight 초기화는 모델 객체를 생성할 때 난수를 사용할 수 있다.

    따라서 모델 생성 후 Seed를 설정하는 것보다
    모델 생성 전에 Seed를 설정해야 초기 Weight 재현성을 높일 수 있다.
    """

    validate_random_seed(
        seed=seed,
    )

    validate_deterministic_option(
        deterministic_algorithms=(
            deterministic_algorithms
        ),
    )

    # Python 표준 Library의 난수 생성기를 설정한다.
    random.seed(
        seed
    )

    # NumPy 난수 생성기를 설정한다.
    #
    # 향후 데이터 분석·평가·샘플 선택 과정에서
    # NumPy 난수를 사용하는 경우 같은 Seed를 공유한다.
    np.random.seed(
        seed
    )

    # PyTorch CPU 난수 생성기를 설정한다.
    #
    # 향후 다음 작업에 영향을 줄 수 있다.
    #
    #     모델 Weight 초기화
    #
    #     Dropout
    #
    #     Random Transform
    torch.manual_seed(
        seed
    )

    cuda_available = (
        torch.cuda.is_available()
    )

    cuda_seed_applied = False

    # 현재 컴퓨터에서는 실행되지 않는다.
    #
    # 향후 CUDA 환경에서 같은 프로젝트를 실행할 때
    # 모든 CUDA Device의 Seed를 설정한다.
    if cuda_available:
        torch.cuda.manual_seed_all(
            seed
        )

        cuda_seed_applied = True

    # True이면 PyTorch가 가능한 경우 결정적 알고리즘을 사용하도록
    # 요청한다.
    #
    # False이면 일반 실행 모드를 사용한다.
    torch.use_deterministic_algorithms(
        deterministic_algorithms
    )

    return ReproducibilitySettings(
        seed=seed,
        deterministic_algorithms=(
            deterministic_algorithms
        ),
        cuda_available=(
            cuda_available
        ),
        cuda_seed_applied=(
            cuda_seed_applied
        ),
        device=(
            get_default_device_name()
        ),
    )


def print_reproducibility_settings(
    settings: ReproducibilitySettings,
) -> None:
    """
    적용된 재현성 설정을 사람이 확인할 수 있도록 출력한다.

    Parameters
    ----------
    settings:
        set_global_random_seed()가 반환한 설정 객체다.
    """

    print(
        "=" * 80
    )

    print(
        "PROJECT REPRODUCIBILITY SETTINGS"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"random seed             : "
        f"{settings.seed}"
    )

    print(
        "deterministic algorithms: "
        f"{settings.deterministic_algorithms}"
    )

    print(
        f"CUDA available          : "
        f"{settings.cuda_available}"
    )

    print(
        f"CUDA seed applied       : "
        f"{settings.cuda_seed_applied}"
    )

    print(
        f"default device           : "
        f"{settings.device}"
    )


def main() -> None:
    """
    프로젝트 기본 Seed 설정을 적용하고 결과를 출력한다.

    실행 명령
    ---------
    프로젝트 Root에서:

        python -m src.reproducibility
    """

    settings = (
        set_global_random_seed(
            seed=(
                DEFAULT_RANDOM_SEED
            ),
            deterministic_algorithms=(
                False
            ),
        )
    )

    print_reproducibility_settings(
        settings=settings,
    )


if __name__ == "__main__":
    main()