# FitStream
A tiny library to make PyTorch experiment easy for small models and in-memory datasets.

# Getting started
Using `uv`
```shell
uv add fitstream
```

Using `pip`:
```shell
pip install fitstream
```

Training a model:
```python
from torch.optim import Adam

from fitstream import epoch_stream, take # epoch_stream is the main entry point

X, y = get_data()
model = get_model()
loss = get_loss()
optimizer = Adam(model.parameters())

# an infinite stream of training epochs (limit it with `take` or `early_stop`)
events = epoch_stream((X, y), model, optimizer, loss, batch_size=32, shuffle=True)
for event in take(events, 10):
    print(f"step={event['step']}, loss={event['train_loss']}")
# epoch=1, loss=...
# epoch=2, loss=...
# ...
```

# Basics
The core idea of the library is "training loop as a stream of events".  The `epoch_stream` is just an iterable over 
dictionaries comprising of the epoch, the model, and the training loss. Everything we do is transforming or enriching
these events. FitStream provides a small `pipe(...)` helper to compose transformations left-to-right.

## Augmentation
The `augment` function turns an "augmenter" (a function that looks at an event and returns extra keys) into a stream
transform stage. We typically compose stages with `pipe(...)`.

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

## Selecting events
Since the training loop is a standard Python iterable, you can use any Python selection logic. FitStream includes a
small helper, `take(...)`, to limit the number of epochs:
```python
from fitstream import epoch_stream, take

for event in take(epoch_stream(...), 100):
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
    early_stop(key="val_loss", patience=10),
)
for event in events:
    print(event)
```

## Side effects
Sometimes you want to log metrics (or write to an external system) without changing the stream. Use `tap(fn)`:
```python
from fitstream import epoch_stream, pipe, tap, take

events = pipe(
    epoch_stream(...),
    tap(lambda ev: print(ev["step"], ev["train_loss"])),
    take(10),
)
list(events)
```

## Sinks
Iterating over events and doing something yourself can be tedious, so we have some utilities to help you process the
event stream.

It is typically useful to collect all events into a list, but exclude the `model` and keep just the metrics. We have 
the `collect` sink for that:
```python
from fitstream import collect, epoch_stream, take

# collect 100 epochs to a list
history = collect(take(epoch_stream(...), 100))
```

We can also store them to a `jsonl` file:
```python
from fitstream import collect_jsonl, epoch_stream, take

# collect 100 epochs to json
collect_jsonl(take(epoch_stream(...), 100), 'runs/my_experiment.jsonl')
```

# Documentation
Full documentation is available at [https://fitstream.readthedocs.io/](https://fitstream.readthedocs.io/).

# Development
- After cloning this repo, run `make setup` to create a virtual environment and install all dependencies.
- Building is done via `uv build`.
- Running tests is done via `make test`
- Building documentation via `make doc`
- Linting via `make lint`
