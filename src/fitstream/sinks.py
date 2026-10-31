import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO, overload

_Reducers = Mapping[str, Callable[[Any, Any], Any]]


def last_value(previous: Any, current: Any) -> Any:
    """Return the current value unchanged, for use as a reducer."""
    return current


def _filter_event(
    event: dict[str, Any],
    *,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
) -> dict[str, Any]:
    if include is not None and exclude is not None:
        raise ValueError("Provide only one of include or exclude.")
    if include is None:
        excluded = {"model"}
        if exclude is not None:
            excluded.update(exclude)
        return {k: v for k, v in event.items() if k not in excluded}
    return {k: event[k] for k in include if k != "model" and k in event}


@overload
def collect(
    events: Iterable[dict[str, Any]],
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    reducers: None = None,
) -> list[dict[str, Any]]: ...


@overload
def collect(
    events: Iterable[dict[str, Any]],
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    reducers: _Reducers,
) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...


def collect(
    events: Iterable[dict[str, Any]],
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    reducers: _Reducers | None = None,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect an event stream into a list of filtered event dicts.

    Args:
        events: Input event stream.
        include: Optional list of keys to keep. When provided, only these keys are
            copied into each collected event. ``"model"`` is always omitted even if
            listed here.
        exclude: Optional list of keys to drop. When ``include`` is not provided,
            events are copied with these keys removed. ``"model"`` is always excluded
            by default.
        reducers: Event keys mapped to ``(accumulator, value)`` callables. The first
            value initializes each accumulator. Reducers read unfiltered fields
            and must not mutate input values. Accumulator values are not copied.

    Returns:
        With ``reducers=None``, a list of shallow-copied, filtered event dicts.
        Supplying a mapping (including ``{}``) returns ``(history, reduction)``.
        An empty stream produces an empty reduction dictionary.

    Example:
        >>> events = [
        ...     {"step": 1, "train_loss": 0.5, "model": object()},
        ...     {"step": 2, "train_loss": 0.4, "model": object()},
        ... ]
        >>> collect(events)
        [{'step': 1, 'train_loss': 0.5}, {'step': 2, 'train_loss': 0.4}]
        >>> collect(events, include=["step"])
        [{'step': 1}, {'step': 2}]
        >>> collect(events, include=["step"], reducers={"train_loss": min})
        ([{'step': 1}, {'step': 2}], {'train_loss': 0.4})

    Notes:
        - Provide only one of ``include`` or ``exclude``; passing both raises
          ``ValueError``.
        - The input iterable is consumed.
        - Missing keys named in ``include`` are ignored.
        - Missing reducer keys raise ``KeyError``; reducer exceptions propagate.
        - Non-callable reducers raise ``TypeError`` before consuming the stream.
    """
    operations = dict(reducers) if reducers is not None else {}
    for key, reducer in operations.items():
        if not callable(reducer):
            raise TypeError(f"Reducer for {key!r} must be callable.")
    history: list[dict[str, Any]] = []
    reduction: dict[str, Any] = {}
    for event in events:
        for key, reducer in operations.items():
            value = event[key]
            reduction[key] = reducer(reduction[key], value) if key in reduction else value
        history.append(_filter_event(event, include=include, exclude=exclude))
    return history if reducers is None else (history, reduction)


def collect_jsonl(
    events: Iterable[dict[str, Any]],
    dest: str | Path | TextIO,
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> None:
    """Write an event stream to JSON Lines format.

    Args:
        events: Input event stream.
        dest: Output destination. May be a filesystem path or an open text-mode file
            object with a ``write(str)`` method.
        include: Optional list of keys to keep in each JSON object. ``"model"`` is
            always omitted even if listed here.
        exclude: Optional list of keys to drop. When ``include`` is not provided,
            ``"model"`` is excluded by default along with any keys listed here.

    Returns:
        ``None``. The function writes one JSON object per line to ``dest``.

    Example:
        >>> import io
        >>> buffer = io.StringIO()
        >>> events = [{"step": 1, "train_loss": 0.5, "model": object()}]
        >>> collect_jsonl(events, buffer)
        >>> buffer.getvalue()
        '{"step": 1, "train_loss": 0.5}\\n'

    Notes:
        - Provide only one of ``include`` or ``exclude``; passing both raises
          ``ValueError``.
        - The input iterable is consumed.
        - Remaining event values must be JSON-serializable by ``json.dumps``.
        - When ``dest`` is a path, the file is opened in text write mode and overwritten.
    """
    if isinstance(dest, (str, Path)):
        with Path(dest).open("w") as handle:
            collect_jsonl(events, handle, include=include, exclude=exclude)
        return
    for event in events:
        record = _filter_event(event, include=include, exclude=exclude)
        dest.write(json.dumps(record) + "\n")


def collect_pd(
    events: Iterable[dict[str, Any]],
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
):
    """Collect an event stream into a pandas DataFrame.

    Args:
        events: Input event stream.
        include: Optional list of keys to keep in each row. ``"model"`` is always
            omitted even if listed here.
        exclude: Optional list of keys to drop. When ``include`` is not provided,
            ``"model"`` is excluded by default along with any keys listed here.

    Returns:
        A pandas ``DataFrame`` with one row per event after filtering.

    Example:
        >>> df = collect_pd(
        ...     [{"step": 1, "train_loss": 0.5, "model": object()}],
        ...     include=["step", "train_loss"],
        ... )
        >>> list(df.columns)
        ['step', 'train_loss']

    Notes:
        - Provide only one of ``include`` or ``exclude``; passing both raises
          ``ValueError``.
        - The input iterable is consumed.
        - ``pandas`` is an optional dependency; if it is not installed, this function
          raises ``ImportError``.
    """
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("pandas is required for collect_pd.") from exc
    rows = [_filter_event(event, include=include, exclude=exclude) for event in events]
    return pd.DataFrame(rows)
