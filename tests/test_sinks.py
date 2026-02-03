from __future__ import annotations

import io
import json

import pytest

from fitstream import collect, collect_jsonl, collect_pd


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
