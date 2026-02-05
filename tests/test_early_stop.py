import pytest

from fitstream import early_stop, pipe


def test_early_stop_stops_after_patience() -> None:
    events = [{"val_loss": loss} for loss in [5.0, 4.0, 4.0, 4.0, 3.0]]

    result = list(early_stop(key="val_loss", patience=2)(events))

    assert [event["val_loss"] for event in result] == [5.0, 4.0, 4.0, 4.0]


def test_early_stop_rejects_invalid_patience() -> None:
    with pytest.raises(ValueError):
        early_stop(key="val_loss", patience=0)


def test_early_stop_explicit_min_mode_matches_default() -> None:
    events = [{"val_loss": loss} for loss in [5.0, 4.0, 4.0, 4.0, 3.0]]

    result_default = list(early_stop(key="val_loss", patience=2)(events))
    result_explicit = list(early_stop(key="val_loss", patience=2, mode="min", min_delta=0.0)(events))

    assert result_explicit == result_default


def test_early_stop_supports_max_mode() -> None:
    events = [{"val_acc": value} for value in [0.50, 0.60, 0.60, 0.59, 0.70]]

    result = list(early_stop(key="val_acc", patience=2, mode="max")(events))

    assert [event["val_acc"] for event in result] == [0.50, 0.60, 0.60, 0.59]


def test_early_stop_min_delta_for_min_mode() -> None:
    events = [{"val_loss": value} for value in [1.00, 0.95, 0.92, 0.85]]

    result = list(early_stop(key="val_loss", patience=2, min_delta=0.1)(events))

    assert [event["val_loss"] for event in result] == [1.00, 0.95, 0.92]


def test_early_stop_min_delta_for_max_mode() -> None:
    events = [{"val_acc": value} for value in [0.50, 0.53, 0.56, 0.57, 0.58]]

    result = list(early_stop(key="val_acc", patience=2, mode="max", min_delta=0.05)(events))

    assert [event["val_acc"] for event in result] == [0.50, 0.53, 0.56, 0.57, 0.58]


def test_early_stop_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        early_stop(key="val_loss", patience=2, mode="lower")  # type: ignore[arg-type]


def test_early_stop_rejects_negative_min_delta() -> None:
    with pytest.raises(ValueError):
        early_stop(key="val_loss", patience=2, min_delta=-0.1)


def test_early_stop_is_pipe_stage() -> None:
    events = [{"val_loss": loss} for loss in [5.0, 4.0, 4.0, 4.0, 3.0]]

    result = list(pipe(events, early_stop(key="val_loss", patience=2)))

    assert [event["val_loss"] for event in result] == [5.0, 4.0, 4.0, 4.0]
