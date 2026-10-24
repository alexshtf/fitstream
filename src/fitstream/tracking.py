from copy import deepcopy
from math import isfinite
from typing import Any, Literal

import torch
from torch import nn


def _snapshot_state(model: nn.Module, device: str | torch.device | None = None) -> dict[str, Any]:
    """Copy model state independently of subsequent model updates.

    Tensor entries are copied to ``device`` (or retain their device when ``None``).
    Opaque extra state and state-dictionary metadata are deep-copied.
    """
    state = model.state_dict()
    memo: dict[int, Any] = {}

    for value in state.values():
        if isinstance(value, torch.Tensor) and id(value) not in memo:
            memo[id(value)] = value.detach().to(device=device, copy=True)

    return deepcopy(state, memo)


class BestModel:
    """Callable that retains the model state with the best observed metric."""

    def __init__(self, metric_key: str, direction: Literal["min", "max"], device: str | torch.device | None) -> None:
        if direction not in {"min", "max"}:
            raise ValueError("direction must be one of {'min', 'max'}.")
        self._metric_key = metric_key
        self._direction = direction
        self._device = device
        self._best: float | None = None
        self._state: dict[str, Any] | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        value = float(event[self._metric_key])
        if not isfinite(value):
            raise ValueError(f"Metric {self._metric_key!r} must be finite.")
        improved = self._best is None or (value < self._best if self._direction == "min" else value > self._best)
        if improved:
            self._state = _snapshot_state(event["model"], self._device)
            self._best = value

    def restore(self, model: nn.Module) -> None:
        """Restore the saved state without sharing mutable extra state with the model."""
        if self._state is None:
            raise RuntimeError("No model state has been captured yet.")
        model.load_state_dict(deepcopy(self._state))


def best_model(
    metric_key: str,
    *,
    direction: Literal["min", "max"] = "min",
    device: str | torch.device | None = None,
) -> BestModel:
    """Create a best-model callback for ``tap`` with explicit ``restore(model)``."""
    return BestModel(metric_key, direction, device)
