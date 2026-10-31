import gc
import io
import json
from collections import OrderedDict
from weakref import ref

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
        self.weight = nn.Parameter(torch.ones(1))
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
    target_child = _ExtraStateModel()
    loaded_versions: list[int | None] = []

    def record_version(_module, _state_dict, _prefix, local_metadata, *_) -> None:
        loaded_versions.append(local_metadata.get("version"))

    target_child.register_load_state_dict_pre_hook(record_version)
    nn.Sequential(target_child).load_state_dict(state)
    assert target_child.info == {"labels": ["original"]}
    assert loaded_versions == [3]


def _capture_jsonl(events, **kwargs):
    with io.StringIO() as output:
        reduction = fts.collect_jsonl(events, output, **kwargs)
        history = [json.loads(line) for line in output.getvalue().splitlines()]
    return history, reduction


@pytest.mark.parametrize("collect_history", [fts.collect, _capture_jsonl], ids=["list", "jsonl"])
def test_best_model_augments_stream_and_reduces_to_loadable_state(collect_history) -> None:
    model = nn.Linear(1, 1, bias=False)
    best = fts.best_model(metric_key="val_loss", direction="min")
    snapshots = []

    def events():
        for step, loss in enumerate([3.0, 1.0, 2.0], start=1):
            with torch.no_grad():
                model.weight.fill_(step)
            yield {"model": model, "step": step, "val_loss": loss}

    stream = fts.pipe(
        events(), fts.augment(best), fts.tap(lambda event: snapshots.append(ref(event["best_state"]["weight"])))
    )
    assert snapshots == []
    history, reduction = collect_history(stream, exclude=["best_state"], reducers={"best_state": fts.last_value})
    assert history == [{"step": step, "val_loss": loss} for step, loss in enumerate([3.0, 1.0, 2.0], start=1)]
    assert model.weight.item() == 3.0
    # Ensure cleanup happens before checking that obsolete snapshots were released.
    gc.collect()
    assert snapshots[0]() is None  # Excluded history must not keep obsolete snapshots alive.
    assert snapshots[1]() is snapshots[2]() is reduction["best_state"]["weight"]

    parameter = model.weight
    model.load_state_dict(reduction["best_state"])
    assert model.weight is parameter
    assert model.weight.item() == 2.0


@pytest.mark.parametrize(
    "direction,metrics",
    [
        (None, [3.0, 1.0, 2.0, 0.0, 0.0]),
        ("min", [3.0, 1.0, 2.0, 0.0, 0.0]),
        ("max", [1.0, 3.0, 2.0, 4.0, 4.0]),
    ],
)
def test_best_model_selects_strict_improvements(direction, metrics) -> None:
    model = nn.Linear(1, 1, bias=False)
    best = fts.best_model("score") if direction is None else fts.best_model("score", direction=direction)
    results = []

    for step, (metric, expected_step) in enumerate(zip(metrics, [1, 2, 2, 4, 4], strict=True), start=1):
        with torch.no_grad():
            model.weight.fill_(step)
        result = best({"model": model, "score": metric})
        assert set(result) == {"best_state"}
        assert result["best_state"]["weight"].item() == expected_step
        if results:
            assert result is not results[-1]
            assert (result["best_state"] is results[-1]["best_state"]) == (step != expected_step)
        results.append(result)

    assert [result["best_state"]["weight"].item() for result in results] == [1, 2, 2, 4, 4]


def test_best_model_instances_and_captured_extra_state_are_independent() -> None:
    model = _ExtraStateModel()
    first = fts.best_model("score")
    state = first({"model": model, "score": 1.0})["best_state"]
    with torch.no_grad():
        model.weight.fill_(2)
    model.info["labels"].append("source change")

    second = fts.best_model("score")
    event = {"model": model, "score": 2.0}
    assert first(event)["best_state"] is state
    assert second(event)["best_state"]["weight"].item() == 2.0
    assert state["weight"].item() == 1.0
    assert state["_extra_state"] == {"labels": ["original"]}


@pytest.mark.parametrize("direction", ["minimum", "MAX", ""])
def test_best_model_rejects_invalid_direction(direction) -> None:
    with pytest.raises(ValueError, match="direction"):
        fts.best_model("score", direction=direction)


def test_best_model_load_into_requires_a_snapshot() -> None:
    best = fts.best_model("score")
    with pytest.raises(RuntimeError, match="No model state has been captured"):
        best.load_into(nn.Linear(1, 1))


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("has_snapshot", [False, True])
def test_best_model_rejects_nonfinite_metrics_without_changing_snapshot(metric, has_snapshot) -> None:
    model = nn.Linear(1, 1, bias=False)
    best = fts.best_model("score")
    with torch.no_grad():
        model.weight.fill_(1)
    if has_snapshot:
        best({"model": model, "score": 1.0})
    with torch.no_grad():
        model.weight.fill_(2)

    with pytest.raises(ValueError, match="must be finite"):
        best({"model": model, "score": metric})

    if has_snapshot:
        best.load_into(model)
        assert model.weight.item() == 1.0
    else:
        with pytest.raises(RuntimeError, match="No model state has been captured"):
            best.load_into(model)


@pytest.mark.parametrize("snapshot_device", [None, "cpu"])
@pytest.mark.parametrize(
    "source_device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_best_model_snapshot_device_and_loading_preserve_target(source_device, snapshot_device) -> None:
    model = nn.BatchNorm1d(2, device=source_device)
    assert model.running_mean is not None
    with torch.no_grad():
        model.weight.fill_(3)
        model.running_mean.fill_(4)
    best = fts.best_model("score", device=snapshot_device)
    best({"model": model, "score": 1.0})

    target = nn.BatchNorm1d(2, device=source_device, dtype=torch.float64).eval()
    parameter = target.weight
    snapshot_devices: set[str] = set()

    def record_devices(_module, state_dict, *_) -> None:
        snapshot_devices.update(value.device.type for value in state_dict.values())

    target.register_load_state_dict_pre_hook(record_devices)
    best.load_into(target)

    assert snapshot_devices == {snapshot_device or source_device}
    assert target.weight is parameter
    assert target.weight.device.type == source_device
    assert target.weight.dtype == torch.float64
    assert not target.training
    assert target.running_mean is not None
    torch.testing.assert_close(target.weight, torch.full_like(target.weight, 3))
    torch.testing.assert_close(target.running_mean, torch.full_like(target.running_mean, 4))


@pytest.mark.parametrize("device", [None, "cpu"])
def test_best_model_repeated_loading_keeps_weights_and_extra_state_independent(device) -> None:
    model = _ExtraStateModel()
    best = fts.best_model("score", device=device)
    best({"model": model, "score": 1.0})
    with torch.no_grad():
        model.weight.fill_(2)
    model.info["labels"].append("source change")

    target = _ExtraStateModel()
    for _ in range(2):
        with torch.no_grad():
            target.weight.fill_(9)
        target.info["labels"].append("target change")
        best.load_into(target)
        assert target.weight.item() == 1.0
        assert target.info == {"labels": ["original"]}
