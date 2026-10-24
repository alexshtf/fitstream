from typing import Any, Self


class LivePlot:
    """Notebook plot callback and context manager returned by ``live_plot``."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self._keys = keys
        self._steps: list[float] = []
        self._values: dict[str, list[float]] = {key: [] for key in keys}

    def __enter__(self) -> Self:
        import matplotlib.pyplot as plt
        from IPython.display import DisplayHandle

        with plt.ioff():
            self._figure, self.ax = plt.subplots()
        self._lines = {key: self.ax.plot([], [], label=key)[0] for key in self._keys}
        self.ax.set_xlabel("step")
        self.ax.legend()
        self._display = DisplayHandle()
        self._display.display(self._figure)
        return self

    def __call__(self, event: dict[str, Any]) -> None:
        """Record an event's step and scalar metrics, then redraw the curves."""
        self._steps.append(float(event["step"]))
        for key in self._keys:
            self._values[key].append(float(event[key]))
        self.refresh()

    def refresh(self) -> None:
        """Update the notebook display with all recorded points."""
        for key, line in self._lines.items():
            line.set_data(self._steps, self._values[key])
        self.ax.relim()
        self.ax.autoscale_view()
        self._display.update(self._figure)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        import matplotlib.pyplot as plt

        self.refresh()
        plt.close(self._figure)


def live_plot(*keys: str) -> LivePlot:
    """Plot scalar event metrics against ``step`` in one updating notebook output.

    Use ``with live_plot(...) as plot`` and consume the stream through ``tap(plot)``.
    Install ``fitstream[notebook]`` for the optional plotting dependencies.
    Each event redraws the curves; context exit refreshes and closes the figure.
    The axes remain available as ``plot.ax`` for customization and saving afterward.
    """
    return LivePlot(keys)
