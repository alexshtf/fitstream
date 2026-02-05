import pytest

from fitstream import ema, pipe


def test_ema_bias_correction_with_decay() -> None:
    events = [{"loss": 10.0}, {"loss": 20.0}, {"loss": 30.0}]

    result = list(ema("loss", decay=0.5)(events))

    observed = [event["loss_ema"] for event in result]
    expected = [10.0, 16.6666666667, 24.2857142857]
    assert observed == pytest.approx(expected)


def test_ema_can_disable_bias_correction() -> None:
    events = [{"loss": 10.0}, {"loss": 20.0}, {"loss": 30.0}]

    result = list(ema("loss", decay=0.5, bias_correction=False)(events))

    observed = [event["loss_ema"] for event in result]
    expected = [5.0, 12.5, 21.25]
    assert observed == pytest.approx(expected)


def test_ema_half_life_matches_equivalent_decay() -> None:
    events = [{"loss": 10.0}, {"loss": 20.0}, {"loss": 30.0}]

    from_decay = list(ema("loss", decay=0.5)(events))
    from_half_life = list(ema("loss", half_life=1.0)(events))

    observed_decay = [event["loss_ema"] for event in from_decay]
    observed_half_life = [event["loss_ema"] for event in from_half_life]
    assert observed_half_life == pytest.approx(observed_decay)


def test_ema_supports_custom_output_key() -> None:
    events = [{"loss": 10.0}]

    result = list(ema("loss", decay=0.5, out_key="smooth_loss")(events))

    assert "smooth_loss" in result[0]
    assert "loss_ema" not in result[0]


def test_ema_rejects_missing_decay_configuration() -> None:
    with pytest.raises(ValueError):
        ema("loss")


def test_ema_rejects_conflicting_decay_configuration() -> None:
    with pytest.raises(ValueError):
        ema("loss", decay=0.9, half_life=10.0)


@pytest.mark.parametrize("decay", [-0.1, 0.0, 1.0, 1.1])
def test_ema_rejects_invalid_decay(decay: float) -> None:
    with pytest.raises(ValueError):
        ema("loss", decay=decay)


@pytest.mark.parametrize("half_life", [0.0, -1.0])
def test_ema_rejects_invalid_half_life(half_life: float) -> None:
    with pytest.raises(ValueError):
        ema("loss", half_life=half_life)


def test_ema_requires_metric_key() -> None:
    with pytest.raises(KeyError):
        list(ema("loss", decay=0.5)([{"other": 1.0}]))


def test_ema_rejects_non_numeric_value() -> None:
    with pytest.raises(ValueError):
        list(ema("loss", decay=0.5)([{"loss": "abc"}]))


def test_ema_is_pipe_stage() -> None:
    events = [{"loss": 10.0}, {"loss": 20.0}]

    result = list(pipe(events, ema("loss", decay=0.5)))

    assert "loss_ema" in result[0]
    assert result[0]["loss"] == 10.0
