from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn


def validation_loss(
    val_data: Sequence[torch.Tensor],
    loss_fn: Callable[..., torch.Tensor],
    *,
    key: str = "val_loss",
    last_label: bool = True,
) -> Callable[[dict[str, Any]], dict[str, float]]:
    """Create an augmentation that computes validation loss.

    Args:
        val_data: Tuple of tensors with batch dimension first. When ``last_label=True``,
            the last tensor is treated as the label tensor and all preceding tensors are
            passed to the model.
        loss_fn: Loss function called as ``loss_fn(pred, labels)`` when ``last_label=True``,
            otherwise as ``loss_fn(pred)``.
        key: Name of the key to store the computed loss under.
        last_label: Whether the last tensor in ``val_data`` is the label tensor.

    Notes:
        Assumes the model and validation tensors are already on the same device.
        Validation is computed on the full validation set (no batching).
    """
    if not val_data:
        raise ValueError("val_data must contain at least one tensor.")
    if last_label and len(val_data) < 2:
        raise ValueError("last_label=True requires at least two tensors (inputs and labels).")

    if last_label:
        *inputs, labels = val_data
    else:
        inputs = list(val_data)
        labels = None

    def compute(event: dict[str, Any]) -> dict[str, float]:
        model = event["model"]
        if not isinstance(model, nn.Module):
            raise TypeError("validation_loss expects an event containing a 'model' key.")

        was_training = model.training
        model.eval()
        with torch.no_grad():
            preds = model(*inputs)
            loss = loss_fn(preds, labels) if last_label else loss_fn(preds)
        if was_training:
            model.train()
        return {key: loss.detach().cpu().item()}

    return compute
