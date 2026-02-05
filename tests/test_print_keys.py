import pytest

from fitstream import pipe, print_keys, tap


def test_print_keys_includes_step_and_formats_values(capsys: pytest.CaptureFixture[str]) -> None:
    callback = print_keys("train_loss", "val_loss")

    callback({"step": 5, "train_loss": 0.12345})

    assert capsys.readouterr().out.strip() == "step=0005 train_loss=0.1235 val_loss=NA"


def test_print_keys_include_step_false(capsys: pytest.CaptureFixture[str]) -> None:
    callback = print_keys("train_loss", include_step=False, precision=2)

    callback({"step": 7, "train_loss": 1.0})

    assert capsys.readouterr().out.strip() == "train_loss=1.00"


def test_print_keys_without_step_key_present(capsys: pytest.CaptureFixture[str]) -> None:
    callback = print_keys("train_loss")

    callback({"train_loss": 0.5})

    assert capsys.readouterr().out.strip() == "train_loss=0.5000"


def test_print_keys_formats_non_numeric_value(capsys: pytest.CaptureFixture[str]) -> None:
    callback = print_keys("status")

    callback({"step": 1, "status": "ok"})

    assert capsys.readouterr().out.strip() == "step=0001 status=ok"


def test_print_keys_rejects_invalid_precision() -> None:
    with pytest.raises(ValueError):
        print_keys("train_loss", precision=-1)


def test_print_keys_requires_key_when_step_disabled() -> None:
    with pytest.raises(ValueError):
        print_keys(include_step=False)


def test_print_keys_composes_with_tap_every(capsys: pytest.CaptureFixture[str]) -> None:
    events = [{"step": 1, "train_loss": 1.0}, {"step": 2, "train_loss": 2.0}, {"step": 3, "train_loss": 3.0}]

    result = list(pipe(events, tap(print_keys("train_loss"), every=2)))

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["step=0001 train_loss=1.0000", "step=0003 train_loss=3.0000"]
    assert result == events
