import matplotlib

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
