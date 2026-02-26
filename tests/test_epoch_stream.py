import torch
import pytest
from torch import nn

from fitstream import epoch_stream


def test_epoch_stream_raises_on_missing_label() -> None:
    data = (torch.arange(3.0).view(3, 1),)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    stream = epoch_stream(data, model, optimizer, nn.MSELoss(), last_label=True)

    with pytest.raises(ValueError):
        next(stream)


def test_epoch_stream_aggregates_loss_over_samples() -> None:
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    stream = epoch_stream((x, y), model, optimizer, nn.MSELoss(), batch_size=3, shuffle=False)
    event = next(stream)

    expected = (y**2).mean().item()
    assert event["train_loss"] == pytest.approx(expected)


def test_epoch_stream_last_label_false_uses_all_inputs() -> None:
    class AddModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return self.scale * (left + right)

    x = torch.ones(5, 1)
    y = torch.full((5, 1), 2.0)
    model = AddModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    def loss_fn(pred: torch.Tensor) -> torch.Tensor:
        return pred.mean()

    stream = epoch_stream((x, y), model, optimizer, loss_fn, batch_size=2, shuffle=False, last_label=False)
    event = next(stream)

    expected = (x + y).mean().item()
    assert event["train_loss"] == pytest.approx(expected)


def test_epoch_stream_appends_extra_fields_to_each_event() -> None:
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    extra = {"run_id": "test-run", "seed": 123}
    stream = epoch_stream((x, y), model, optimizer, nn.MSELoss(), batch_size=2, shuffle=False, extra=extra)

    event1 = next(stream)
    event2 = next(stream)

    assert event1["run_id"] == "test-run"
    assert event1["seed"] == 123
    assert event2["run_id"] == "test-run"
    assert event2["seed"] == 123
