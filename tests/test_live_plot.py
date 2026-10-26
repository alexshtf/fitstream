import builtins
from math import isnan

import matplotlib
import pytest

matplotlib.use("Agg")

from IPython.display import DisplayHandle
import matplotlib.pyplot as plt

import fitstream as fts


def consume(events) -> None:
    """Exhaust a lazy event stream so its plotting side effects run."""
    for _ in events:
        pass


def test_live_plot_creates_curves_and_updates_as_a_stage(monkeypatch) -> None:
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
        history = fts.collect(fts.pipe(events, plot))
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


def test_live_plot_stage_is_lazy_and_stops_consuming_after_exit(plot_updates) -> None:
    events = [{"step": 1, "loss": 3}, {"step": 2, "loss": 2}]
    consumed = []

    def source():
        for event in events:
            consumed.append(event)
            yield event

    with fts.live_plot("loss") as plot:
        stream = plot(source())
        assert not consumed
        assert next(stream) is events[0]
        assert plot_updates[-1] == [3]
    with pytest.raises(RuntimeError):
        next(stream)
    assert consumed == events[:1]


def test_live_plot_throttles_without_losing_points_and_allows_forced_refresh(monkeypatch, plot_updates) -> None:
    now = 0.0
    monkeypatch.setattr("fitstream.plotting.monotonic", lambda: now)
    with fts.live_plot("loss", min_interval=2.0) as plot:
        plot.refresh()
        consume(plot([{"step": 1, "loss": 5.0}]))
        assert plot_updates[-1] == [5.0]
        now = 1.0
        consume(plot([{"step": 2, "loss": 4.0}]))
        assert plot_updates[-1] == [5.0]
        now = 2.0
        consume(plot([{"step": 3, "loss": 3.0}]))
        assert plot_updates[-1] == [5.0, 4.0, 3.0]
        now = 2.5
        consume(plot([{"step": 4, "loss": 2.0}]))
        assert plot_updates[-1] == [5.0, 4.0, 3.0]
        plot.refresh()
        assert plot_updates[-1] == [5.0, 4.0, 3.0, 2.0]
        now = 4.0
        consume(plot([{"step": 5, "loss": 1.0}]))
        assert plot_updates[-1] == [5.0, 4.0, 3.0, 2.0]
    assert plot_updates[-1] == [5.0, 4.0, 3.0, 2.0, 1.0]


def test_live_plot_zero_interval_redraws_each_event(monkeypatch, plot_updates) -> None:
    monkeypatch.setattr("fitstream.plotting.monotonic", lambda: 0.0)
    with fts.live_plot("loss", min_interval=0.0) as plot:
        for step in range(1, 4):
            consume(plot([{"step": step, "loss": float(step)}]))
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
            consume(plot([{"step": 2, "loss": 3.0}]))
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


@pytest.mark.parametrize(
    "keys,interval",
    [
        ((), 1),
        (("",), 1),
        ((None,), 1),
        (("loss", "loss"), 1),
        (("loss",), -1),
        (("loss",), float("nan")),
        (("loss",), float("inf")),
    ],
)
def test_live_plot_rejects_invalid_configuration(keys, interval) -> None:
    with pytest.raises(ValueError):
        fts.live_plot(*keys, min_interval=interval)


@pytest.mark.parametrize("metrics", [{}, {"loss": float("nan")}, {"loss": float("inf")}, {"loss": -float("inf")}])
def test_live_plot_missing_and_nonfinite_metrics_leave_gaps(plot_updates, metrics) -> None:
    events = [
        {"step": 1, "loss": 5.0},
        {"step": 2, **metrics},
        {"step": 3, "loss": 3.0},
    ]
    with fts.live_plot("loss") as plot:
        consume(plot(events))
    first, gap, last = plot_updates[-1]
    assert (first, last) == (5.0, 3.0)
    assert isnan(gap)


@pytest.mark.parametrize(
    "event,error",
    [
        ({}, KeyError),
        ({"step": float("nan")}, ValueError),
        ({"step": float("inf")}, ValueError),
        ({"step": 2, "loss": "invalid"}, ValueError),
        ({"step": 2, "loss": 9, "other": [1, 2]}, TypeError),
    ],
)
def test_live_plot_rejected_event_does_not_change_curves(plot_updates, event, error) -> None:
    with fts.live_plot("loss", "other") as plot:
        consume(plot([{"step": 1, "loss": 5, "other": 6}]))
        with pytest.raises(error):
            consume(plot([event]))
        consume(plot([{"step": 3, "loss": 3, "other": 4}]))
    curves = {line.get_label(): line for line in plot.ax.get_lines()}
    for key, expected in {"loss": [5, 3], "other": [6, 4]}.items():
        assert list(curves[key].get_xdata()) == [1, 3]
        assert list(curves[key].get_ydata()) == expected


@pytest.mark.parametrize("dependency", ["matplotlib", "IPython"])
def test_live_plot_missing_dependency_explains_installation(monkeypatch, dependency) -> None:
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.split(".")[0] == dependency:
            raise ModuleNotFoundError(name, name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"pip install 'fitstream\[notebook\]'"):
        with fts.live_plot("loss"):
            pytest.fail("The body must not run without plotting dependencies.")


@pytest.mark.parametrize(
    "action",
    [lambda plot: consume(plot([{"step": 1, "loss": 2}])), lambda plot: plot.refresh()],
    ids=["record", "refresh"],
)
def test_live_plot_requires_active_context(plot_updates, action) -> None:
    plot = fts.live_plot("loss")
    with pytest.raises(RuntimeError):
        action(plot)
    with plot:
        pass
    with pytest.raises(RuntimeError):
        action(plot)


def test_live_plot_rejects_nested_entry_and_reuse(plot_updates) -> None:
    plot = fts.live_plot("loss")
    with plot:
        with pytest.raises(RuntimeError):
            with plot:
                pytest.fail("Nested entry must fail.")
        consume(plot([{"step": 1, "loss": 2}]))
    assert plot_updates[-1] == [2]
    with pytest.raises(RuntimeError):
        with plot:
            pytest.fail("Reusing a plotter must fail.")


@pytest.fixture
def closed_figures(monkeypatch):
    closed, original_close = [], plt.close

    def record_close(figure):
        closed.append(figure)
        original_close(figure)

    monkeypatch.setattr(plt, "close", record_close)
    return closed


@pytest.fixture(params=[False, True])
def supplied_ax(request):
    ax = plt.subplots()[1] if request.param else None
    try:
        yield ax
    finally:
        if ax is not None:
            plt.close(ax.figure)


def test_live_plot_closes_only_owned_figure_when_display_fails(monkeypatch, closed_figures, supplied_ax) -> None:
    failure = RuntimeError("Display failed")

    def fail_display(*args):
        raise failure

    monkeypatch.setattr(DisplayHandle, "display", fail_display)
    plot = fts.live_plot("loss", ax=supplied_ax)
    with pytest.raises(RuntimeError) as caught:
        with plot:
            pytest.fail("The context body must not run when display fails.")
    assert caught.value is failure
    assert (plot.ax.figure in closed_figures) == (supplied_ax is None)
    with pytest.raises(RuntimeError):
        plot.refresh()


@pytest.mark.parametrize("training_error", [None, ValueError("Training failed"), KeyboardInterrupt()])
def test_live_plot_final_refresh_failure_preserves_training_error(
    monkeypatch, plot_updates, closed_figures, training_error, supplied_ax
) -> None:
    failure = RuntimeError("Refresh failed")

    def fail_update(*args):
        raise failure

    expected = training_error if training_error is not None else failure
    with pytest.raises(type(expected)) as caught:
        with fts.live_plot("loss", ax=supplied_ax) as plot:
            consume(plot([{"step": 1, "loss": 2}]))
            monkeypatch.setattr(DisplayHandle, "update", fail_update)
            if training_error is not None:
                raise training_error
    assert caught.value is expected
    assert (plot.ax.figure in closed_figures) == (supplied_ax is None)
    with pytest.raises(RuntimeError):
        plot.refresh()
