from typing import Any

import torch
import pytest
from torch import nn

from fitstream import augment, evaluate, validation_loss


def test_augment_merges_dicts() -> None:
    events = [{"step": 1, "train_loss": 0.5}, {"step": 2, "train_loss": 0.25}]

    def add_double_step(event: dict[str, Any]) -> dict[str, Any]:
        return {"double_step": event["step"] * 2}

    augmented = list(augment(add_double_step)(events))

    assert augmented[0]["double_step"] == 2
    assert augmented[1]["double_step"] == 4
    assert events[0] == {"step": 1, "train_loss": 0.5}


@pytest.mark.parametrize("training", [True, False])
def test_evaluate_shares_one_forward_pass_and_returns_floats(training: bool) -> None:
    class CountingModel(nn.Module):
        calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            assert not self.training
            assert not torch.is_grad_enabled()
            return 2 * x

    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    y = torch.tensor([[0.0], [1.0]])
    model = CountingModel().train(training)
    predictions: list[torch.Tensor] = []

    def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        assert target is y
        predictions.append(pred)
        return (pred - target).square().mean()

    def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        assert not torch.is_grad_enabled()
        assert target is y
        predictions.append(pred)
        return (pred - target).abs().mean().item()

    event = {"model": model, "step": 1}
    events = augment(evaluate((x, y), metrics={"val_loss": mse, "val_mae": mae}))([event])
    assert model.calls == 0
    result = list(events)[0]

    assert model.calls == 1
    assert predictions[0] is predictions[1]
    assert not predictions[0].requires_grad
    assert result == {**event, "val_loss": 6.5, "val_mae": 2.5}
    assert isinstance(result["val_loss"], float) and isinstance(result["val_mae"], float)
    assert model.training is training
    assert event == {"model": model, "step": 1}


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
