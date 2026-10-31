from collections.abc import Callable
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


def best_model(
    metric_key: str,
    *,
    direction: Literal["min", "max"] = "min",
    device: str | torch.device | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create an augmenter that adds the best observed model state to each event.

    Args:
        metric_key: Event key containing a finite metric convertible to ``float``.
        direction: ``"min"`` selects lower values; ``"max"`` selects higher values.
        device: Storage device for snapshot tensor entries. ``None`` preserves each
            tensor's device; ``"cpu"`` copies directly to CPU without an intermediate
            GPU copy. Dtypes are preserved. Other extra state is deep-copied; the
            device option does not recursively move tensors nested inside it.

    Returns:
        A stateful callable for ``augment`` returning ``{"best_state": state_dict}``.
        The snapshot contains parameters, buffers, and custom extra state.

    Raises:
        ValueError: ``direction`` is neither ``"min"`` nor ``"max"``. The returned
            callable also raises if the metric, converted to ``float``, is not finite.

    Notes:
        Keeps one snapshot, replacing it only on strict improvement. Ties and worse
        metrics reuse the current snapshot; earlier snapshots are never updated in place.
        Place ``augment(best_model(...))`` after computing the metric. Only events
        reaching the augmenter are considered. Create a new augmenter for each run.

        Treat emitted states as read-only. When collecting metric history, use
        ``exclude=["best_state"]`` and ``reducers={"best_state": last_value}`` to
        retain only the final best state. Load it with ``model.load_state_dict(state)``.
        PyTorch may attach mutable custom extra state by reference when loading;
        copy the state first if isolation after loading is needed. Optimizer and
        scheduler state are not included.
    """
    if direction not in {"min", "max"}:
        raise ValueError("direction must be one of {'min', 'max'}.")
    best: float | None = None
    state: dict[str, Any] | None = None

    def track(event: dict[str, Any]) -> dict[str, Any]:
        nonlocal best, state
        value = float(event[metric_key])
        if not isfinite(value):
            raise ValueError(f"Metric {metric_key!r} must be finite.")
        improved = best is None or (value < best if direction == "min" else value > best)
        if improved:
            state = _snapshot_state(event["model"], device)
            best = value
        return {"best_state": state}

    return track
