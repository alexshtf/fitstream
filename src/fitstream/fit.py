from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Literal
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
    *,
    every: int = 1,
    start: int = 1,
) -> Transform:
    """Create a stage that performs side effects and yields events unchanged.

    Args:
        fn: Callback applied to selected events.
        every: Call ``fn`` every N events (event-count based).
        start: 1-based event index at which callback scheduling starts.
    """
    if not callable(fn):
        raise TypeError("tap requires a callable.")
    if every < 1:
        raise ValueError("every must be >= 1.")
    if start < 1:
        raise ValueError("start must be >= 1.")

    def stage(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for index, event in enumerate(events, start=1):
            if index >= start and (index - start) % every == 0:
                fn(event)
            yield event

    return stage


def print_keys(
    *keys: str,
    precision: int = 4,
    include_step: bool = True,
    step_key: str = "step",
) -> Callable[[dict[str, Any]], None]:
    """Create an event callback that prints selected keys in one compact line.

    Args:
        *keys: Event keys to print.
        precision: Number of digits after the decimal for numeric values.
        include_step: Whether to include ``step_key`` first when present.
        step_key: Event key used for the step prefix.
    """
    if precision < 0:
        raise ValueError("precision must be >= 0.")
    if not keys and not include_step:
        raise ValueError("Provide at least one key when include_step=False.")

    def format_value(value: Any) -> str:
        match value:
            case torch.Tensor() as tensor if tensor.numel() == 1:
                return f"{float(tensor.detach().cpu().item()):.{precision}f}"
            case bool() as boolean:
                return str(boolean)
            case int() | float() as number:
                return f"{float(number):.{precision}f}"
            case _:
                return str(value)

    def callback(event: dict[str, Any]) -> None:
        parts: list[str] = []
        if include_step and step_key in event:
            try:
                parts.append(f"{step_key}={int(event[step_key]):04d}")
            except Exception:
                parts.append(f"{step_key}={event[step_key]}")
        for key in keys:
            if key in event:
                parts.append(f"{key}={format_value(event[key])}")
            else:
                parts.append(f"{key}=NA")
        print(" ".join(parts))

    return callback


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


def ema(
    key: str,
    *,
    decay: float | None = None,
    half_life: float | None = None,
    out_key: str | None = None,
    bias_correction: bool = True,
) -> Transform:
    """Create a stage that adds an exponential moving average of ``key``.

    Exactly one of ``decay`` or ``half_life`` must be provided.
    The update rule is ``m = decay * m + (1 - decay) * x`` with ``m`` initialized to 0.
    """
    if (decay is None) == (half_life is None):
        raise ValueError("Provide exactly one of decay or half_life.")
    if half_life is not None:
        if half_life <= 0.0:
            raise ValueError("half_life must be > 0.")
        decay = 2.0 ** (-1.0 / half_life)
    assert decay is not None
    if not (0.0 < decay < 1.0):
        raise ValueError("decay must be in (0, 1).")

    output_key = out_key or f"{key}_ema"

    def stage(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        aggregate = 0.0
        t = 0
        for event in events:
            t += 1
            value = float(event[key])
            aggregate = decay * aggregate + (1.0 - decay) * value
            if bias_correction:
                smoothed = aggregate / (1.0 - (decay**t))
            else:
                smoothed = aggregate
            yield event | {output_key: smoothed}

    return stage


def early_stop(
    key: str,
    patience: int,
    *,
    mode: Literal["min", "max"] = "min",
    min_delta: float = 0.0,
) -> Transform:
    """Yield events until the metric stops improving for `patience` steps.

    Args:
        key: Event key containing the monitored metric.
        patience: Number of consecutive non-improving events tolerated before stopping.
        mode: Improvement direction. ``"min"`` means lower is better, ``"max"`` means
            higher is better.
        min_delta: Minimum absolute change required to count as an improvement.

    Use as a pipe stage:

    - ``pipe(events, early_stop(key="val_loss", patience=10))``
    """
    if patience < 1:
        raise ValueError("patience must be >= 1.")
    if mode not in {"min", "max"}:
        raise ValueError("mode must be one of {'min', 'max'}.")
    if min_delta < 0.0:
        raise ValueError("min_delta must be >= 0.")

    def apply(stream: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        best: float | None = None
        bad = 0
        for event in stream:
            value = float(event[key])
            if best is None:
                improved = True
            elif mode == "min":
                improved = value < (best - min_delta)
            else:
                improved = value > (best + min_delta)

            if improved:
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
