from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn


def evaluate(
    val_data: Sequence[torch.Tensor],
    *,
    metrics: Mapping[str, Callable[..., torch.Tensor | float]],
    last_label: bool = True,
) -> Callable[[dict[str, Any]], dict[str, float]]:
    """Create an augmenter that evaluates named metrics from shared predictions.

    Args:
        val_data: Tensors with batch dimension first, passed as model inputs. When
            ``last_label=True``, the last tensor is used as labels instead.
        metrics: Mapping from event keys to metric functions. Each receives
            ``(predictions, labels)``, or just ``predictions`` when ``last_label=False``,
            and returns a single-element tensor or a number.
        last_label: Whether the last tensor in ``val_data`` contains labels.

    Returns:
        A callable that reads ``event["model"]`` and returns metric values as Python floats.

    Notes:
        Runs one forward pass on the full dataset per call, without batching.
        The model and data stay on their existing device, which must be the same.
        Runs the model in evaluation mode and disables gradients for inference and
        metrics. Restores each module's prior mode even if evaluation raises.
        Metric functions must not mutate the shared predictions or labels.
    """
    if not val_data:
        raise ValueError("val_data must contain at least one tensor.")
    if last_label and len(val_data) < 2:
        raise ValueError("last_label=True requires at least two tensors (inputs and labels).")
    metrics = dict(metrics)
    if not metrics:
        raise ValueError("metrics must contain at least one metric.")
    inputs = list(val_data)
    labels = inputs.pop() if last_label else None

    def compute(event: dict[str, Any]) -> dict[str, float]:
        model = event["model"]
        if not isinstance(model, nn.Module):
            raise TypeError("evaluate expects an event containing a 'model' key with a torch.nn.Module.")
        modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            with torch.no_grad():
                preds = model(*inputs)
                result: dict[str, float] = {}
                for key, metric in metrics.items():
                    value = metric(preds, labels) if last_label else metric(preds)
                    result[key] = float(value.detach().item() if isinstance(value, torch.Tensor) else value)
                return result
        finally:
            for module, training in modes:
                module.training = training

    return compute


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
    return evaluate(val_data, metrics={key: loss_fn}, last_label=last_label)
