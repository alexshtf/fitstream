import io
import json
from operator import add, truediv

import pytest

from fitstream import collect, collect_jsonl, collect_pd, last_value


def test_collect_drops_model_by_default() -> None:
    events = [{"step": 1, "model": object(), "loss": 0.5}]

    result = collect(events)

    assert result == [{"step": 1, "loss": 0.5}]


def test_collect_include_excludes_model_even_if_listed() -> None:
    events = [{"model": object(), "a": 1, "b": 2}]

    result = collect(events, include=["model", "a"])

    assert result == [{"a": 1}]


def test_collect_exclude_removes_requested_keys() -> None:
    events = [{"model": object(), "a": 1, "b": 2}]

    result = collect(events, exclude=["a"])

    assert result == [{"b": 2}]


def test_collect_rejects_include_and_exclude() -> None:
    with pytest.raises(ValueError):
        collect([{"a": 1}], include=["a"], exclude=["b"])


@pytest.mark.parametrize("filtering", [{"include": ["loss"]}, {"exclude": ["step", "payload"]}])
def test_collect_reduces_raw_fields_while_filtering_history(filtering) -> None:
    payloads = [object(), object(), object()]
    events = (
        {"loss": loss, "step": step, "payload": payload}
        for step, (loss, payload) in enumerate(zip([3, 1, 2], payloads, strict=True), start=1)
    )
    history, reduction = collect(events, **filtering, reducers={"loss": min, "step": add, "payload": last_value})
    assert history == [{"loss": 3}, {"loss": 1}, {"loss": 2}]
    assert reduction == {"loss": 1, "step": 6, "payload": payloads[-1]}
    assert reduction["payload"] is payloads[-1]


def test_collect_seeds_then_reduces_in_stream_order() -> None:
    calls = []

    def subtract(previous, current):
        calls.append((previous, current))
        return previous - current

    def events():
        for index, value in enumerate([10, 3, 2]):
            # Reduce previous events before pulling another; the first value only seeds the accumulator.
            assert len(calls) == max(0, index - 1)
            yield {"x": value}

    assert collect(events(), reducers={"x": subtract})[1] == {"x": 5}
    assert calls == [(10, 3), (7, 2)]


@pytest.mark.parametrize("events", [[], [{"x": 1}]])
@pytest.mark.parametrize("kwargs", [{}, {"reducers": None}, {"reducers": {}}])
def test_collect_return_forms(events, kwargs) -> None:
    expected = events if kwargs.get("reducers") is None else (events, {})
    assert collect(iter(events), **kwargs) == expected


def test_collect_empty_stream_with_reducers() -> None:
    assert collect([], reducers={"x": min}) == ([], {})


@pytest.mark.parametrize("values", [[None, object()], [object(), None]])
def test_collect_reduces_none_values(values) -> None:
    calls = []

    def keep_last(previous, current):
        calls.append(previous)
        return last_value(previous, current)

    assert collect(({"x": value} for value in values), reducers={"x": keep_last})[1]["x"] is values[-1]
    assert calls == [values[0]]


@pytest.mark.parametrize(
    "events,reducer,error", [([{}], min, KeyError), ([{"x": 1}, {"x": 0}], truediv, ZeroDivisionError)]
)
def test_collect_propagates_reduction_errors(events, reducer, error) -> None:
    with pytest.raises(error):
        collect(iter(events), reducers={"x": reducer})


def test_collect_validates_reducers_before_consuming() -> None:
    def events():
        pytest.fail("Input was consumed before validating reducers.")
        yield {}

    with pytest.raises(TypeError, match="'x'.*callable"):
        collect(events(), reducers={"x": None})


def test_collect_jsonl_accepts_file_object() -> None:
    events = [{"step": 1, "model": object(), "loss": 0.5}]
    buffer = io.StringIO()

    collect_jsonl(events, buffer)

    buffer.seek(0)
    record = json.loads(buffer.read().strip())
    assert record == {"step": 1, "loss": 0.5}


def test_collect_jsonl_writes_path(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    events = [{"step": 1, "model": object(), "loss": 0.5}]

    collect_jsonl(events, path)

    lines = path.read_text().splitlines()
    assert json.loads(lines[0]) == {"step": 1, "loss": 0.5}


def test_collect_pd_handles_missing_dependency(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pandas is required for collect_pd"):
        collect_pd([{"step": 1, "loss": 0.5}])


def test_collect_pd_creates_dataframe() -> None:
    df = collect_pd([{"step": 1, "loss": 0.5, "model": object()}])
    assert list(df.columns) == ["step", "loss"]
