import pytest

from fitstream import tick


def test_tick_calls_fn_and_yields_events() -> None:
    events = [{"step": 1}, {"step": 2}]
    calls: list[str] = []

    def record() -> None:
        calls.append("tick")

    result = list(tick(record)(events))

    assert calls == ["tick", "tick"]
    assert result == events
    assert result[0] is events[0]


def test_tick_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        tick(None)  # type: ignore[arg-type]
