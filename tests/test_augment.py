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


@pytest.mark.parametrize("key", ["val_loss", "custom_loss"])
def test_validation_loss_uses_event_model(key: str) -> None:
    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[2.0], [4.0]])
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    augment_fn = validation_loss((x, y), nn.MSELoss(), key=key)
    result = augment_fn({"model": model})

    assert result == {key: 0.0}


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


@pytest.mark.parametrize("last_label", [True, False])
def test_evaluate_accepts_multiple_inputs_with_or_without_labels(last_label: bool) -> None:
    class AddModel(nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left + right

    x = torch.ones(2, 1)
    data = (x, 2 * x, x) if last_label else (x, 2 * x)
    metrics = {
        "mean": lambda pred, target=0: (pred - target).mean(),
        "sum": lambda pred, target=0: (pred - target).sum(),
    }
    result = evaluate(data, metrics=metrics, last_label=last_label)({"model": AddModel()})
    expected = 2.0 if last_label else 3.0
    assert result == {"mean": expected, "sum": 2 * expected}


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("failure", [None, "forward", "metric"])
def test_evaluate_restores_mixed_modes_and_grad_state(training: bool, failure: str | None) -> None:
    model = nn.Sequential(nn.Linear(1, 1), nn.Dropout()).train(training)
    model[0].train(not training)
    modes = [module.training for module in model.modules()]
    grad_enabled = torch.is_grad_enabled()

    def check_forward(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        assert all(not child.training for child in module.modules())
        assert not torch.is_grad_enabled()
        if failure == "forward":
            raise RuntimeError("forward failure")

    def metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert all(not module.training for module in model.modules())
        assert not torch.is_grad_enabled()
        if failure == "metric":
            raise RuntimeError("metric failure")
        return (pred - target).square().mean()

    model.register_forward_pre_hook(check_forward)
    x = torch.ones(2, 1)
    compute = evaluate((x, x), metrics={"loss": metric})
    if failure is None:
        assert isinstance(compute({"model": model})["loss"], float)
    else:
        with pytest.raises(RuntimeError, match=failure):
            compute({"model": model})
    assert [module.training for module in model.modules()] == modes
    assert torch.is_grad_enabled() is grad_enabled


def test_evaluate_rejects_empty_metrics() -> None:
    x = torch.ones(2, 1)
    with pytest.raises(ValueError, match="metrics"):
        evaluate((x, x), metrics={})


def test_evaluate_rejects_nonscalar_metric_and_restores_mode() -> None:
    x = torch.ones(2, 1)
    model = nn.Identity().train()
    compute = evaluate((x, x), metrics={"invalid": lambda pred, target: pred})
    with pytest.raises(RuntimeError):
        compute({"model": model})
    assert model.training
