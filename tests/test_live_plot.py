import matplotlib
import pytest

matplotlib.use("Agg")

from IPython.display import DisplayHandle
import matplotlib.pyplot as plt

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


@pytest.mark.parametrize("supply_ax", [False, True])
def test_live_plot_preserves_supplied_axes_and_closes_only_owned_figures(monkeypatch, plot_updates, supply_ax) -> None:
    figure, (ax, other_ax) = plt.subplots(1, 2)
    other_curve = other_ax.plot([0, 1], [1, 0])[0]
    (baseline,) = ax.plot([0, 10], [4, 4], label="baseline")
    ax.set(xlabel="epoch", title="Experiment", xlim=(0, 10))
    legend = ax.legend()
    closed = []
    close = plt.close

    def record_close(fig):
        closed.append(fig)
        close(fig)

    monkeypatch.setattr(plt, "close", record_close)
    try:
        with fts.live_plot("loss", ax=ax if supply_ax else None) as plot:
            assert (plot.ax is ax) == supply_ax
            plot({"step": 2, "loss": 3.0})
        assert (plot.ax.figure in closed) == (not supply_ax)
        assert figure not in closed
        assert other_curve in other_ax.get_lines()
        assert list(other_curve.get_ydata()) == [1, 0]
        assert baseline in ax.get_lines()
        assert list(baseline.get_ydata()) == [4, 4]
        assert ax.get_xlabel() == "epoch"
        assert ax.get_title() == "Experiment"
        assert ax.get_xlim() == (0, 10)
        assert ax.get_legend() is legend
        curves = {line.get_label(): line for line in plot.ax.get_lines()}
        assert set(curves) == ({"baseline", "loss"} if supply_ax else {"loss"})
        assert list(curves["loss"].get_xdata()) == [2.0]
        assert list(curves["loss"].get_ydata()) == [3.0]
    finally:
        close(figure)
