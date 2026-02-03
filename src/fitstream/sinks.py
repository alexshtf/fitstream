from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO
import json


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
    included = set(include)
    included.discard("model")
    return {k: event[k] for k in included if k in event}


def collect(
    events: Iterable[dict[str, Any]],
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect an event stream into a list of dicts."""
    return [_filter_event(event, include=include, exclude=exclude) for event in events]


def collect_jsonl(
    events: Iterable[dict[str, Any]],
    dest: str | Path | TextIO,
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> None:
    """Write events to JSONL (one JSON object per line)."""
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
    """Collect events into a pandas DataFrame."""
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("pandas is required for collect_pd.") from exc
    rows = [_filter_event(event, include=include, exclude=exclude) for event in events]
    return pd.DataFrame(rows)
