import pytest

from fitstream import early_stop, pipe


def test_early_stop_stops_after_patience() -> None:
    events = [{"val_loss": loss} for loss in [5.0, 4.0, 4.0, 4.0, 3.0]]

    result = list(early_stop(events, key="val_loss", patience=2))

    assert [event["val_loss"] for event in result] == [5.0, 4.0, 4.0, 4.0]


def test_early_stop_rejects_invalid_patience() -> None:
    with pytest.raises(ValueError):
        list(early_stop([{"val_loss": 1.0}], key="val_loss", patience=0))


def test_early_stop_is_pipe_stage() -> None:
    events = [{"val_loss": loss} for loss in [5.0, 4.0, 4.0, 4.0, 3.0]]

    result = list(pipe(events, early_stop(key="val_loss", patience=2)))

    assert [event["val_loss"] for event in result] == [5.0, 4.0, 4.0, 4.0]
