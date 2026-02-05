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


def test_tap_every_applies_callback_on_cadence() -> None:
    events = [{"step": 1}, {"step": 2}, {"step": 3}, {"step": 4}, {"step": 5}]
    seen: list[int] = []

    def record(event: dict) -> None:
        seen.append(event["step"])

    result = list(tap(record, every=2)(events))

    assert seen == [1, 3, 5]
    assert result == events


def test_tap_start_offsets_cadence() -> None:
    events = [{"step": 1}, {"step": 2}, {"step": 3}, {"step": 4}, {"step": 5}]
    seen: list[int] = []

    def record(event: dict) -> None:
        seen.append(event["step"])

    result = list(tap(record, every=2, start=2)(events))

    assert seen == [2, 4]
    assert result == events


def test_tap_rejects_invalid_every() -> None:
    with pytest.raises(ValueError):
        tap(lambda _: None, every=0)


def test_tap_rejects_invalid_start() -> None:
    with pytest.raises(ValueError):
        tap(lambda _: None, start=0)
