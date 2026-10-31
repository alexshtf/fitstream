# FitStream (for PyTorch)

**Compose PyTorch training through ordinary Python iterators.**

For experiments with small PyTorch models and in-memory datasets: change metrics and stopping rules without rewriting
your training loop. Keep your existing model and train with a `for` loop.

<div align="center">

[![fitstream-build](https://github.com/alexshtf/fitstream/actions/workflows/ci.yml/badge.svg)](https://github.com/alexshtf/fitstream/actions/workflows/ci.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/fitstream)](https://pypi.org/project/fitstream/)
[![PyPI](https://img.shields.io/pypi/v/fitstream)](https://pypi.org/project/fitstream/)
![Python version](https://img.shields.io/badge/python-3.12+-important)

</div>

## Training as an iterator

Each iteration trains one epoch and yields its results. FitStream's utilities take an iterator and return another,
adding validation or stopping rules along the way. Chain them together with `pipe(...)`:

```python
from fitstream import augment, early_stop, epoch_stream, pipe, take, validation_loss

# Use your tensors, model, loss function, and optimizer.
events = pipe(
    epoch_stream((X, y), model, optimizer, loss, batch_size=32, shuffle=True),
    augment(validation_loss((X_val, y_val), loss)),  # add validation loss to each event
    take(500),                                    # limit the event stream
    early_stop(key="val_loss", patience=10),       # stop when validation loss stalls
)

# Training happens as you iterate.
for event in events:
    print(f"step={event['step']}, train={event['train_loss']}, val={event['val_loss']}")
```

`events` is an ordinary Python iterator. Each event is a dictionary with the epoch number (`step`), model, and losses.
The chain of utilities above forms a pipeline. Use `break` to stop, collect metrics in a list, or add your own generator
to the chain.

# Getting started

Using `uv`:
```shell
uv add fitstream
```

Using `pip`:
```shell
pip install fitstream
```

# Basics

The following building blocks enrich, select, observe, and collect events. Each stream transform accepts an iterable
and returns an iterable, so you can combine them with `pipe(...)` or call them directly.
`pipe(stream, first, second)` is simply `second(first(stream))`. The built-in stages are lazy: constructing a pipeline
does not train the model; consuming it does.

## Augmentation
The `augment` function turns an "augmenter" (a callable that looks at an event and returns extra keys) into a stream
transform stage. Augmenters can keep state between events. We typically compose stages with `pipe(...)`.

Here is an example - we add the norm of the model parameters to each event:
```python
from torch import nn, linalg
from fitstream import epoch_stream, augment, pipe

def model_param_norm(ev: dict) -> dict:
    model_params = nn.utils.parameters_to_vector(ev['model'].parameters())
    return {'model_param_norm': linalg.norm(model_params)}


events = pipe(
    epoch_stream(...),
    augment(model_param_norm),
)
for event in events:
    print(f"step={event['step']}", 
          f"model_param_norm={event['model_param_norm']}"
    )
```

We also have some built-in augmentation functions. Here is an example of adding validation loss to each event:
```python
from torch import nn
from fitstream import epoch_stream, augment, pipe, validation_loss

validation_set = get_validation_set()
events = pipe(
    epoch_stream(...),
    augment(validation_loss(validation_set, nn.CrossEntropyLoss())),
)
for event in events:
    print(f"step={event['step']}, val_loss={event['val_loss']}")
```

For several validation metrics, use `evaluate` to share one forward pass:

```python
import fitstream as fts

def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean()

events = fts.pipe(
    fts.epoch_stream(...),
    fts.augment(
        fts.evaluate(
            validation_set,
            metrics={"val_loss": nn.CrossEntropyLoss(), "val_accuracy": accuracy},
        )
    ),
)
```

Each metric receives the same predictions and labels and returns a single-element tensor or number.
The supplied names become event keys, with Python float values. `validation_loss` remains a shortcut for one metric.

We can, of course, augment the stream more than once:
```python
events = pipe(
    epoch_stream(...),
    augment(validation_loss(...)),
    augment(model_param_norm),
)
for event in events:
    print(f"step={event['step']}", 
          f"val_loss={event['val_loss']}",
          f"model_param_norm={event['model_param_norm']}"
    )
```

### Keeping the best model

Use `best_model` with `augment` to add the best observed model state to each event as `best_state`:

```python
import fitstream as fts

events = fts.pipe(
    fts.epoch_stream((X, y), model, optimizer, loss, batch_size=32),
    fts.take(500),
    fts.augment(fts.validation_loss((X_val, y_val), loss)),
    fts.early_stop(key="val_loss", patience=10),
    fts.augment(fts.best_model("val_loss")),
)
for event in events:
    print(event["step"], event["val_loss"])

model.load_state_dict(event["best_state"])
```

In `fts.best_model`, `direction="min"` (the default) selects lower values; use `direction="max"` for metrics such as accuracy.
Its `device=None` default keeps snapshot tensors on their original devices. If your model is on the GPU, you can pass
`device="cpu"` to `fts.best_model` to store the best state on the CPU.

## Selecting events
Since the training loop is a standard Python iterable, you can use any Python selection logic. FitStream includes a
small helper, `take(...)`, to limit the number of epochs:
```python
from fitstream import epoch_stream, take

for event in take(100)(epoch_stream(...)):
    print(event)
# {'step': 1, ....}
# {'step': 2, ...}
# ...
# { 'step': 100, ...}
```

`fitstream` has some of its own selection primitives, such as early stopping:
```python
from fitstream import augment, early_stop, epoch_stream, pipe, take, validation_loss

events = pipe(
    epoch_stream(...),
    augment(validation_loss(...)),
    take(500),  # safety cap
    early_stop(key="val_loss", patience=10, mode="min", min_delta=1e-4),
)
for event in events:
    print(event)
```

`mode="min"` is the default. Use `mode="max"` for metrics such as accuracy, and set
`min_delta` to ignore tiny noisy changes.

## Side effects
Sometimes you want to log metrics (or write to an external system) without changing the stream.
Use `tap(fn, every=...)` and the built-in `print_keys(...)` helper:
```python
from fitstream import epoch_stream, pipe, print_keys, tap, take

events = pipe(
    epoch_stream(...),
    tap(print_keys("train_loss"), every=5),
    take(20),
)
list(events)
```

### Live learning curves in notebooks

Install `pip install "fitstream[notebook]"` for Matplotlib and IPython support.
In a Jupyter-compatible notebook, consume a bounded training stream inside the plotting context:

```python
import fitstream as fts

events = fts.pipe(
    fts.epoch_stream((X, y), model, optimizer, loss, batch_size=32),
    fts.take(200),
    fts.augment(fts.validation_loss((X_val, y_val), loss)),
)
with fts.live_plot("train_loss", "val_loss", min_interval=1.0) as plot:
    for event in plot(events):
        print(event["step"], event["val_loss"])
```

Every event is recorded; automatic redraws are throttled to at most once per second after the first event.
Context exit displays the final points. Use `plot.refresh()` inside the context to force a redraw,
or `min_interval=0` to redraw every event. Create a new stream and plotter for each run.
See the [tutorial](docs/tutorial.md#live-learning-curves-in-notebooks) for supplied axes and saving figures.

## Sinks
Iterating over events and doing something yourself can be tedious, so we have some utilities to help you process the
event stream.

Use `collect` to store event records in a list. The `model` field is omitted by default:
```python
from fitstream import collect, epoch_stream, take

# collect 100 epochs to a list
history = collect(take(100)(epoch_stream(...)))
```

We can also store them to a `jsonl` file:
```python
from fitstream import collect_jsonl, epoch_stream, take

# collect 100 epochs to json
collect_jsonl(take(100)(epoch_stream(...)), 'runs/my_experiment.jsonl')
```

`collect` can aggregate data from events using its `reducers` parameter. For example, to collect the history and
calculate the total training time:

```python
from operator import add

events = take(100)(epoch_stream((X, y), model, optimizer, loss, batch_size=32))
history, reduction = collect(events, reducers={"train_time_sec": add})
print(reduction["train_time_sec"])
```

Each entry in `reducers` maps an event field to a function that combines the accumulated value with the next value.
The first event's value initializes the accumulator.

Reducers see fields independently of `include` and `exclude`. For a stream containing `best_state`, use
`exclude=["best_state"]` with `reducers={"best_state": fts.last_value}` to keep snapshots out of history and
retain the final best state.

With reducers, `collect_pd` returns `(dataframe, reduction)`; `collect_jsonl` writes the history and returns
only the reduction dictionary. Without reducers, their return values stay unchanged.

# Documentation
Full documentation is available at [https://fitstream.readthedocs.io/](https://fitstream.readthedocs.io/).

# Development
- After cloning this repo, run `make setup` to create a virtual environment and install all dependencies.
- Building is done via `uv build`.
- Running tests is done via `make test`
- Building documentation via `make doc`
- Serving documentation locally (and opening it) via `make doc-open`
- Linting via `make lint`
