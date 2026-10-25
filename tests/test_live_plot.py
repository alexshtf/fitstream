import matplotlib
import pytest

matplotlib.use("Agg")

from IPython.display import DisplayHandle

import fitstream as fts


def test_live_plot_creates_curves_and_updates_through_tap(monkeypatch) -> None:
    displayed, updates = [], []

    def record_update(handle, figure):
        updates.append(
            {
                line.get_label(): (list(line.get_xdata()), list(line.get_ydata()))
                for ax in figure.get_axes()
                for line in ax.get_lines()
            }
        )

    monkeypatch.setattr(DisplayHandle, "display", lambda handle, figure: displayed.append(figure))
    monkeypatch.setattr(DisplayHandle, "update", record_update)
    events = [
        {"step": 2, "train_loss": 3.0, "val_loss": 4.0},
        {"step": 5, "train_loss": 2.0, "val_loss": 3.0},
    ]
    with fts.live_plot("train_loss", "val_loss") as plot:
        history = fts.collect(fts.pipe(events, fts.tap(plot)))
        assert plot.ax.figure in displayed
        assert plot.ax in plot.ax.figure.get_axes()
        assert updates  # Plotting happens during iteration, not only on exit.
    assert history == events
    assert len(plot.ax.get_lines()) == 2
    assert updates[-1] == {
        "train_loss": ([2.0, 5.0], [3.0, 2.0]),
        "val_loss": ([2.0, 5.0], [4.0, 3.0]),
    }


@pytest.fixture
def plot_updates(monkeypatch):
    updates = []
    monkeypatch.setattr(DisplayHandle, "display", lambda *args: None)
    monkeypatch.setattr(
        DisplayHandle,
        "update",
        lambda handle, figure: updates.append(list(figure.get_axes()[0].get_lines()[0].get_ydata())),
    )
    return updates


def test_live_plot_throttles_without_losing_points_and_allows_forced_refresh(monkeypatch, plot_updates) -> None:
    now = 0.0
    monkeypatch.setattr("fitstream.plotting.monotonic", lambda: now)
    with fts.live_plot("loss", min_interval=2.0) as plot:
        plot.refresh()
        plot({"step": 1, "loss": 5.0})
        assert plot_updates[-1] == [5.0]
        now = 1.0
        plot({"step": 2, "loss": 4.0})
        assert plot_updates[-1] == [5.0]
        now = 2.0
        plot({"step": 3, "loss": 3.0})
        assert plot_updates[-1] == [5.0, 4.0, 3.0]
        now = 2.5
        plot({"step": 4, "loss": 2.0})
        assert plot_updates[-1] == [5.0, 4.0, 3.0]
        plot.refresh()
        assert plot_updates[-1] == [5.0, 4.0, 3.0, 2.0]
        now = 4.0
        plot({"step": 5, "loss": 1.0})
        assert plot_updates[-1] == [5.0, 4.0, 3.0, 2.0]
    assert plot_updates[-1] == [5.0, 4.0, 3.0, 2.0, 1.0]


def test_live_plot_zero_interval_redraws_each_event(monkeypatch, plot_updates) -> None:
    monkeypatch.setattr("fitstream.plotting.monotonic", lambda: 0.0)
    with fts.live_plot("loss", min_interval=0.0) as plot:
        for step in range(1, 4):
            plot({"step": step, "loss": float(step)})
            assert plot_updates[-1] == list(range(1, step + 1))
