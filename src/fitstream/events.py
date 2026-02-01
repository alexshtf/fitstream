from typing import TypedDict

from torch import nn

class Event(TypedDict):
    """Per-epoch event emitted by fit/stream utilities.

    Keys:
        model: Live model reference updated each epoch.
        step: 1-based epoch index.
        train_loss: Mean training loss for the epoch.
        train_time_sec: Wall-clock seconds spent in the epoch.
    """
    model: nn.Module
    step: int
    train_loss: float
    train_time_sec: float

