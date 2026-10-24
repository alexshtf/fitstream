from collections import OrderedDict

import pytest
import torch
from torch import nn

import fitstream as fts
from fitstream.tracking import _snapshot_state


@pytest.mark.parametrize("device", [None, "cpu", torch.device("cpu")])
def test_snapshot_owns_parameters_and_buffers(device) -> None:
    model = nn.BatchNorm1d(2, dtype=torch.float64)
    state = _snapshot_state(model, device)

    assert state["weight"].dtype == torch.float64
    assert state["num_batches_tracked"].dtype == torch.int64
    assert all(value.device == torch.device("cpu") for value in state.values())
    assert all(not value.requires_grad for value in state.values())
    assert model.running_mean is not None
    assert model.num_batches_tracked is not None
    with torch.no_grad():
        model.weight.fill_(2)
        model.running_mean.fill_(3)
        model.num_batches_tracked.add_(1)

    torch.testing.assert_close(state["weight"], torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(state["running_mean"], torch.zeros(2, dtype=torch.float64))
    assert state["num_batches_tracked"].item() == 0
    state["bias"].fill_(4)
    torch.testing.assert_close(model.bias, torch.zeros(2, dtype=torch.float64))


class _ExtraStateModel(nn.Module):
    _version = 3

    def __init__(self) -> None:
        super().__init__()
        self.info = {"labels": ["original"]}

    def get_extra_state(self):
        return self.info

    def set_extra_state(self, state) -> None:
        self.info = state


def test_snapshot_preserves_metadata_and_copies_extra_state() -> None:
    child = _ExtraStateModel()
    model = nn.Sequential(child)
    state = _snapshot_state(model)
    child.info["labels"].append("changed")

    assert isinstance(state, OrderedDict)
    assert state["0._extra_state"] == {"labels": ["original"]}
    restored_child = _ExtraStateModel()
    loaded_versions: list[int | None] = []

    def record_version(_module, _state_dict, _prefix, local_metadata, *_) -> None:
        loaded_versions.append(local_metadata.get("version"))

    restored_child.register_load_state_dict_pre_hook(record_version)
    nn.Sequential(restored_child).load_state_dict(state)
    assert restored_child.info == {"labels": ["original"]}
    assert loaded_versions == [3]


def test_best_model_tracks_and_restores_through_tap() -> None:
    model = nn.Linear(1, 1, bias=False)
    best = fts.best_model(metric_key="val_loss", direction="min")

    def events():
        for step, loss in enumerate([3.0, 1.0, 2.0], start=1):
            with torch.no_grad():
                model.weight.fill_(step)
            yield {"model": model, "step": step, "val_loss": loss}

    history = fts.collect(fts.pipe(events(), fts.tap(best)))
    assert [event["val_loss"] for event in history] == [3.0, 1.0, 2.0]
    assert model.weight.item() == 3.0

    parameter = model.weight
    best.restore(model)
    assert model.weight is parameter
    assert model.weight.item() == 2.0
