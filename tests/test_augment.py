from typing import Any

import torch
import pytest
from torch import nn

from fitstream import augment, validation_loss


def test_augment_merges_dicts() -> None:
    events = [{"step": 1, "train_loss": 0.5}, {"step": 2, "train_loss": 0.25}]

    def add_double_step(event: dict[str, Any]) -> dict[str, Any]:
        return {"double_step": event["step"] * 2}

    augmented = list(augment(events, add_double_step))

    assert augmented[0]["double_step"] == 2
    assert augmented[1]["double_step"] == 4
    assert events[0] == {"step": 1, "train_loss": 0.5}


def test_validation_loss_uses_event_model() -> None:
    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[2.0], [4.0]])
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    augment_fn = validation_loss((x, y), nn.MSELoss())
    result = augment_fn({"model": model})

    assert result["val_loss"] == 0.0


def test_validation_loss_requires_labels() -> None:
    x = torch.ones(2, 1)
    with pytest.raises(ValueError):
        validation_loss((x,), nn.MSELoss())


def test_validation_loss_requires_model_key() -> None:
    x = torch.ones(2, 1)
    y = torch.ones(2, 1)
    augment_fn = validation_loss((x, y), nn.MSELoss())
    with pytest.raises(KeyError):
        augment_fn({})


def test_validation_loss_rejects_non_module_model() -> None:
    x = torch.ones(2, 1)
    y = torch.ones(2, 1)
    augment_fn = validation_loss((x, y), nn.MSELoss())
    with pytest.raises(TypeError):
        augment_fn({"model": object()})


def test_validation_loss_allows_last_label_false_with_single_tensor() -> None:
    x = torch.ones(2, 1)
    model = nn.Identity()
    augment_fn = validation_loss((x,), lambda pred: pred.mean(), last_label=False)

    result = augment_fn({"model": model})

    assert result["val_loss"] == pytest.approx(x.mean().item())


def test_validation_loss_last_label_false_uses_all_inputs() -> None:
    class AddModel(nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left + right

    x = torch.ones(2, 1)
    y = torch.full((2, 1), 2.0)
    model = AddModel()
    augment_fn = validation_loss((x, y), lambda pred: pred.mean(), last_label=False)

    result = augment_fn({"model": model})

    assert result["val_loss"] == pytest.approx((x + y).mean().item())
