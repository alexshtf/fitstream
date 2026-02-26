from typing import Any, Literal, overload

from torch import nn


class Event(dict[str, Any]):
    """Per-epoch event emitted by fit/stream utilities.

    This is a regular dict with some overloads which exists solely to
    as a hint for users and IDE autocomplete engines what to expect.
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
