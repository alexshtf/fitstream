from typing import Any

from fitstream import augment


def test_augment_merges_dicts() -> None:
    events = [{"step": 1, "train_loss": 0.5}, {"step": 2, "train_loss": 0.25}]

    def add_double_step(event: dict[str, Any]) -> dict[str, Any]:
        return {"double_step": event["step"] * 2}

    augmented = list(augment(events, add_double_step))

    assert augmented[0]["double_step"] == 2
    assert augmented[1]["double_step"] == 4
    assert events[0] == {"step": 1, "train_loss": 0.5}
