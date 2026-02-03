import pytest

from fitstream import tap


def test_tap_calls_fn_and_yields_events() -> None:
    events = [{"step": 1}, {"step": 2}]
    seen: list[int] = []

    def record(event: dict) -> None:
        seen.append(event["step"])

    result = list(tap(record)(events))

    assert seen == [1, 2]
    assert result == events
    assert result[0] is events[0]


def test_tap_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        tap(None)  # type: ignore[arg-type]
