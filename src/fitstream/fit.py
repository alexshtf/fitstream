from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any
import time

import torch
from torch import nn

from .batching import iter_batches
from .events import Event

Transform = Callable[[Iterable[dict[str, Any]]], Iterable[dict[str, Any]]]


def augment(
    fn: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> Transform:
    """Create a transform that merges extra keys into each event.

    Args:
        fn: Function called for each event. The returned mapping is shallow-merged
            into the event. Returning ``None`` adds nothing.

    Returns:
        A transform that accepts an event stream and yields augmented events.
    """

    def transform(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for event in events:
            extra = fn(event) or {}
            if not isinstance(extra, dict):
                raise TypeError("augment function must return a dict or None.")
            yield event | extra

    return transform


def pipe(stream: Iterable[dict[str, Any]], *stages: Transform) -> Iterable[dict[str, Any]]:
    """Compose stream transforms left-to-right.

    Args:
        stream: Input event stream.
        stages: Transform functions applied in order.

    Returns:
        The transformed event stream.
    """
    for stage in stages:
        if not callable(stage):
            raise TypeError("pipe stages must be callable.")
        stream = stage(stream)
    return stream


def take(n: int) -> Transform:
    """Limit an event stream to the first ``n`` events.

    Can be used directly on a stream or as a pipe stage:

    - ``take(10)(events)``
    - ``pipe(events, take(10))``
    """
    if n < 0:
        raise ValueError("n must be >= 0.")

    def stage(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        count = 0
        for event in events:
            if count >= n:
                break
            yield event
            count += 1

    return stage


def tap(
    fn: Callable[[dict[str, Any]], Any],
) -> Transform:
    """Create a stage that performs side effects and yields events unchanged."""
    if not callable(fn):
        raise TypeError("tap requires a callable.")

    def stage(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for event in events:
            fn(event)
            yield event

    return stage


def tick(
    fn: Callable[[], Any],
) -> Transform:
    """Create a stage that runs a no-arg callback per event and yields events unchanged."""
    if not callable(fn):
        raise TypeError("tick requires a callable.")

    def stage(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for event in events:
            fn()
            yield event

    return stage


def early_stop(
    key: str,
    patience: int,
) -> Transform:
    """Yield events until the metric stops improving for `patience` steps.

    Use as a pipe stage:

    - ``pipe(events, early_stop(key="val_loss", patience=10))``
    """
    if patience < 1:
        raise ValueError("patience must be >= 1.")

    def apply(stream: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        best = float("inf")
        bad = 0
        for event in stream:
            value = float(event[key])
            if value < best:
                best = value
                bad = 0
            else:
                bad += 1
            yield event
            if bad >= patience:
                break

    return apply


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
) -> Iterator[Event]:
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
        total_loss = torch.zeros((), device=train_data[0].device)
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
            total_loss += loss.detach() * batch_samples
            total_samples += int(batch_samples)

        step += 1
        epoch_loss = (total_loss / total_samples).item()
        yield Event(
            model=model,
            step=step,
            train_loss=epoch_loss,
            train_time_sec=time.perf_counter() - epoch_start,
        )
