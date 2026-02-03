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
from torch import nn
from torch.optim import Adam

from fitstream import epoch_stream # <-- the library's main entry point

X, y = get_data()
model = get_model()
loss = get_loss()
optimizer = Adam(model.parameters())

# an infinite stream of training epochs
for event in epoch_stream((X, y), batch_size=32, shuffle=True, model, optimizer, loss):
    print(f"step={event['step']}, loss={event['train_loss']}")
# epoch=1, loss=...
# epoch=2, loss=...
# ...
```

# Basics
The core idea of the library is "training loop as a stream of events".  The `epoch_stream` is just an iteable over 
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
Since the training loop is a standard Python iterable, we can use `itertools` directly. For example, we typically
want to limit the number of epochs:
```python
from itertools import islice

for event in islice(epoch_stream(...), n=100):
    print(event)
# {'step': 1, ....}
# {'step': 2, ...}
# ...
# { 'step': 100, ...}
```

`fitstream` has some of its own selection primitives, such as early stopping:
```python
from fitstream import epoch_stream, early_stop, augment, pipe, validation_loss

events = pipe(
    epoch_stream(...),
    augment(validation_loss(...)),
)
for event in early_stop(events, key="val_loss", patience=10):
    print(event)
```

## Sinks
Iterating over events and doing something yourself can be tedious, so we have some utilities to help you process the
event stream.

It is typically useful to collect all events into a list, but exclude the `model` and keep just the metrics. We have 
the `collect` sink for that:
```python
from fitstream import epoch_stream, collect
from itertools import islice

# collect 100 epochs to a list
events = islice(epoch_stream(...), n=100)
history = collect(events)
```

We can also store them to a `jsonl` file:
```python
from fitstream import epoch_stream, collect_jsonl

# collect 100 epochs to json
events = islice(epoch_stream(...), n=100)
collect_jsonl(events, 'runs/my_experiment.jsonl')
```

# Documentation
Full documentation is available at [https://fitstream.readthedocs.io/](https://fitstream.readthedocs.io/).

# Development
- After cloning this repo, run `make setup` to create a virtual environment and install all dependencies.
- Building is done via `uv build`.
- Running tests is done via `make test`
- Building documentation via `make doc`
- Linting via `make pre-commit`
