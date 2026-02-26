from typing import Any, Literal, overload

from torch import nn


class Event(dict[str, Any]):
    """Per-epoch event emitted by fit/stream utilities.

    Keys:
        model: Live model reference updated each epoch.
        step: 1-based epoch index.
        train_loss: Mean training loss for the epoch.
        train_time_sec: Wall-clock seconds spent in the epoch.
    """

    @overload
    def __getitem__(self, key: Literal["model"]) -> nn.Module: ...

    @overload
    def __getitem__(self, key: Literal["step"]) -> int: ...

    @overload
    def __getitem__(self, key: Literal["train_loss"]) -> float: ...

    @overload
    def __getitem__(self, key: Literal["train_time_sec"]) -> float: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)
