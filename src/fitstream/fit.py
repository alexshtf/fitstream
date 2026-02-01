from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any
import time

import torch
from torch import nn

from .batching import iter_batches
from .events import Event


def augment(
    events: Iterable[dict[str, Any]],
    fn: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> Iterable[dict[str, Any]]:
    """Return a stream with extra keys merged into each event.

    Args:
        events: Iterable of event dictionaries.
        fn: Function called for each event. The returned mapping is shallow-merged
            into the event. Returning ``None`` adds nothing.

    Yields:
        New event dictionaries with the additional keys.
    """
    for event in events:
        extra = fn(event) or {}
        if not isinstance(extra, dict):
            raise TypeError("augment function must return a dict or None.")
        yield event | extra


def epoch_stream(
    train_data: Sequence[torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[..., torch.Tensor],
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    last_label: bool = True,
    generator: torch.Generator | None = None,
) -> Iterable[Event]:
    """Yield per-epoch training events from in-memory tensors.

    Args:
        train_data: Tuple of tensors with batch dimension first. When ``last_label=True``,
            the last tensor is treated as the label tensor and all preceding tensors are
            passed to the model.
        model: PyTorch model to train.
        optimizer: Optimizer instance constructed with the model parameters.
        loss_fn: Loss function. Called as ``loss_fn(pred, labels)`` when ``last_label=True``,
            otherwise as ``loss_fn(pred)``.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples before batching.
        last_label: Whether the last tensor in ``train_data`` is the label tensor.
        generator: Optional torch.Generator forwarded to ``iter_batches`` for reproducible
            shuffling.

    Notes:
        This function assumes the model and all tensors are already on the same device.
        It does not copy tensors or take snapshots of model weights.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not train_data:
        raise ValueError("train_data must contain at least one tensor.")
    if last_label and len(train_data) < 2:
        raise ValueError("last_label=True requires at least two tensors (inputs and labels).")

    step = 0
    while True:
        model.train()
        epoch_start = time.perf_counter()
        total_loss = None
        total_samples = 0

        for batch in iter_batches(*train_data, batch_size=batch_size, shuffle=shuffle, generator=generator):
            if last_label:
                *inputs, labels = batch
            else:
                inputs = list(batch)
                labels = None

            preds = model(*inputs)
            loss = loss_fn(preds, labels) if last_label else loss_fn(preds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_samples = inputs[0].shape[0]
            if total_loss is None:
                total_loss = loss.detach() * batch_samples
            else:
                total_loss += loss.detach() * batch_samples
            total_samples += int(batch_samples)

        step += 1
        total_loss = total_loss.cpu().item()
        yield Event(
            model=model, step=step, train_loss=total_loss / total_samples,
            train_time_sec=time.perf_counter() - epoch_start
        )
