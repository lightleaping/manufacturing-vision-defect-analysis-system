"""Day 12 Detection Checkpoint 저장·복원 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from src.detection.checkpoint import (
    REQUIRED_CHECKPOINT_KEYS,
    build_detection_checkpoint_payload,
    load_detection_checkpoint_payload,
    restore_detection_checkpoint,
    save_detection_checkpoint,
)


CLASS_MAPPING = {
    "BACKGROUND": 0,
    "crazing": 1,
    "inclusion": 2,
}


def create_components():
    model = nn.Linear(4, 2)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    return model, optimizer, scheduler


def test_checkpoint_round_trip_restores_model_optimizer_scheduler(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = create_components()

    inputs = torch.ones((1, 4))
    loss = model(inputs).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    expected_parameters = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    payload = build_detection_checkpoint_payload(
        epoch=2,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config={"epochs": 3, "batch_size": 1},
        class_mapping=CLASS_MAPPING,
        best_metric=0.75,
        history=[{"epoch": 1, "map_50": 0.5}],
    )
    latest = tmp_path / "day12_detection_latest.pt"
    best = tmp_path / "day12_detection_best.pt"
    saved_latest, saved_best = save_detection_checkpoint(
        payload=payload,
        latest_path=latest,
        best_path=best,
        is_best=True,
    )

    assert saved_latest == latest
    assert saved_best == best
    assert latest.is_file()
    assert best.is_file()

    for parameter in model.parameters():
        parameter.data.zero_()

    state = restore_detection_checkpoint(
        path=best,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_class_mapping=CLASS_MAPPING,
        map_location="cpu",
    )

    assert state.epoch == 2
    assert state.next_epoch == 3
    assert state.best_metric == pytest.approx(0.75)
    assert state.history == [{"epoch": 1, "map_50": 0.5}]
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected_parameters[name])


def test_latest_only_does_not_create_best(tmp_path: Path) -> None:
    model, optimizer, _ = create_components()
    payload = build_detection_checkpoint_payload(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        training_config={"epochs": 1},
        class_mapping=CLASS_MAPPING,
        best_metric=0.0,
        history=[],
    )
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    _, saved_best = save_detection_checkpoint(
        payload=payload,
        latest_path=latest,
        best_path=best,
        is_best=False,
    )
    assert latest.is_file()
    assert saved_best is None
    assert not best.exists()


def test_load_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    torch.save({"epoch": 0}, path)

    with pytest.raises(KeyError, match="missing keys"):
        load_detection_checkpoint_payload(path)


def test_restore_rejects_class_mapping_mismatch(tmp_path: Path) -> None:
    model, optimizer, _ = create_components()
    payload = build_detection_checkpoint_payload(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        training_config={},
        class_mapping=CLASS_MAPPING,
        best_metric=0.0,
        history=[],
    )
    path = tmp_path / "latest.pt"
    save_detection_checkpoint(payload=payload, latest_path=path)

    with pytest.raises(ValueError, match="class_mapping"):
        restore_detection_checkpoint(
            path=path,
            model=model,
            expected_class_mapping={
                "BACKGROUND": 0,
                "scratches": 1,
            },
        )


def test_payload_contains_required_keys() -> None:
    model, optimizer, _ = create_components()
    payload = build_detection_checkpoint_payload(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        training_config={},
        class_mapping=CLASS_MAPPING,
        best_metric=0.0,
        history=[],
    )
    assert REQUIRED_CHECKPOINT_KEYS <= set(payload)
