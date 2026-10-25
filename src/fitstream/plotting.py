from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class LivePlot:
    """Notebook plot callback and context manager returned by ``live_plot``."""

    def __init__(self, keys: tuple[str, ...], *, ax: Axes | None = None, min_interval: float = 1.0) -> None:
        self._keys = keys
        self._provided_ax = ax
        self._owned_figure: Figure | None = None
        self._min_interval = min_interval
        self._last_refresh: float | None = None
        self._steps: list[float] = []
        self._values: dict[str, list[float]] = {key: [] for key in keys}

    def __enter__(self) -> Self:
        import matplotlib.pyplot as plt
        from IPython.display import DisplayHandle

        if self._provided_ax is None:
            with plt.ioff():
                self._owned_figure, self.ax = plt.subplots()
            self.ax.set_xlabel("step")
        else:
            self.ax = self._provided_ax
        self._lines = {key: self.ax.plot([], [], label=key)[0] for key in self._keys}
        if self._provided_ax is None:
            self.ax.legend()
        self._display = DisplayHandle()
        self._display.display(self.ax.figure)
        return self

    def __call__(self, event: dict[str, Any]) -> None:
        """Record every event, redrawing when the minimum interval has elapsed."""
        self._steps.append(float(event["step"]))
        for key in self._keys:
            self._values[key].append(float(event[key]))
        if (
            self._last_refresh is None
            or len(self._steps) == 1
            or monotonic() - self._last_refresh >= self._min_interval
        ):
            self.refresh()

    def refresh(self) -> None:
        """Redraw all recorded points immediately, restarting the refresh interval."""
        for key, line in self._lines.items():
            line.set_data(self._steps, self._values[key])
        self.ax.relim()
        self.ax.autoscale_view()
        self._display.update(self.ax.figure)
        self._last_refresh = monotonic()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        import matplotlib.pyplot as plt

        self.refresh()
        if self._owned_figure is not None:
            plt.close(self._owned_figure)


def live_plot(*keys: str, ax: Axes | None = None, min_interval: float = 1.0) -> LivePlot:
    """Plot scalar event metrics against ``step`` in one updating notebook output.

    Use ``with live_plot(...) as plot`` and consume the stream through ``tap(plot)``.
    Install ``fitstream[notebook]`` for the optional plotting dependencies.
    The first event is drawn immediately. Subsequent automatic redraws are at least
    ``min_interval`` seconds apart (default: 1.0); zero redraws on every event.
    All points are retained. Updates happen at event boundaries, without a timer.
    Explicit ``refresh()`` and context exit redraw regardless of the interval.
    Pass ``ax`` to add curves to existing axes. Existing artists, labels, and legend
    are preserved; configure labels and the legend yourself when supplying axes.
    Context exit closes only figures created by this helper.
    The axes remain available as ``plot.ax`` for customization and saving afterward.
    """
    return LivePlot(keys, ax=ax, min_interval=min_interval)
