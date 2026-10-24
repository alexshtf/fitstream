from copy import deepcopy
from typing import Any

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
