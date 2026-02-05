# Tutorial: Zero → Hero

This tutorial takes you from your first FitStream training loop to a composable “training pipeline” built from small,
testable pieces.

We’ll train a regression model on Google’s California Housing dataset:

```python
TRAIN_URL = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
TEST_URL = "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv"
```

The goal is to predict `median_house_value` from a few numeric features.

FitStream’s core idea is:

> Your training loop is an **event stream** (an iterable of dictionaries).
> Each epoch yields an event like `{"step": 1, "train_loss": ..., "model": ..., ...}`.

A stream is **lazy**: nothing happens until you iterate over it (with a `for` loop, `list(...)`, `collect(...)`, etc.).
Since `epoch_stream` is infinite, you’ll usually add a stop condition like `take(n)` or `early_stop(...)`.

## 0) Install (start from zero)

FitStream requires Python 3.12+ and PyTorch.

Using `pip`:

```bash
pip install fitstream
pip install torch pandas
```

Using `uv`:

```bash
uv add fitstream torch pandas
```

`pandas` is only used here for loading CSVs conveniently; FitStream itself does not depend on it at runtime.

## 1) Load the dataset

Let’s download the train/test CSVs directly from the URLs and inspect the columns.

```python
import pandas as pd

TRAIN_URL = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
TEST_URL = "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv"

train_df = pd.read_csv(TRAIN_URL)
test_df = pd.read_csv(TEST_URL)

print(train_df.columns.tolist())
# [
#   'longitude', 'latitude', 'housing_median_age', 'total_rooms',
#   'total_bedrooms', 'population', 'households', 'median_income',
#   'median_house_value'
# ]
```

We’ll use `median_house_value` as the regression target.

## 2) Your first FitStream training loop (no pipes yet)

Start with the simplest possible model and the simplest possible features: predict house value from *only*
`median_income`.

### 2.1 Convert DataFrame → tensors

FitStream expects **in-memory tensors** with the batch dimension first.

```python
import torch

feature_cols = ["median_income"]
label_col = "median_house_value"

x_train = torch.tensor(train_df[feature_cols].to_numpy(), dtype=torch.float32)
y_train = torch.tensor(train_df[[label_col]].to_numpy(), dtype=torch.float32)
```

The target is in “dollars” and can be large. For a friendlier numeric scale (and faster optimization),
we’ll predict **hundreds of thousands of dollars**:

```python
y_scale = 100_000.0
y_train = y_train / y_scale
```

### 2.2 Build a model

```python
from torch import nn

model = nn.Linear(in_features=x_train.shape[1], out_features=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()
```

### 2.3 Train with `epoch_stream`

`epoch_stream(...)` is FitStream’s main entry point. It yields one event **per epoch** forever, so we’ll stop it using
FitStream’s `take(...)` helper.

```python
from fitstream import epoch_stream, take

events = epoch_stream(
    (x_train, y_train),
    model,
    optimizer,
    loss_fn,
    batch_size=256,
    shuffle=True,
)

for event in take(10)(events):
    # each event is a dict-like object
    print(f"epoch={event['step']:03d} train_loss={event['train_loss']:.4f}")
```

That’s the most basic FitStream workflow: **iterate over events**.

### 2.4 Collect metrics to a list

Since a stream is just an iterable, you can “sink” it into a list. FitStream’s `collect(...)` drops `model` by default
so you store metrics instead of huge Python objects.

Note: `epoch_stream(...)` trains the live `model` you pass in. If you run multiple examples in order, you’re continuing
training unless you re-create `model` and `optimizer`.

```python
from fitstream import collect, take

history = collect(take(50)(epoch_stream((x_train, y_train), model, optimizer, loss_fn, batch_size=256)))
print(history[-1])
# {'step': 50, 'train_loss': 0.23, 'train_time_sec': 0.01}
```

## 3) Level up: use all features (and normalize)

Using only `median_income` is a nice “hello world”, but we can do much better with all available numeric features.

### 3.1 Build the full feature matrix

```python
feature_cols = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
label_col = "median_house_value"

x_train = torch.tensor(train_df[feature_cols].to_numpy(), dtype=torch.float32)
y_train = torch.tensor(train_df[[label_col]].to_numpy(), dtype=torch.float32) / 100_000.0

x_val = torch.tensor(test_df[feature_cols].to_numpy(), dtype=torch.float32)
y_val = torch.tensor(test_df[[label_col]].to_numpy(), dtype=torch.float32) / 100_000.0
```

### 3.2 Standardize features (recommended)

These columns have very different scales. Standardizing helps a lot for neural nets.

```python
mean = x_train.mean(dim=0, keepdim=True)
std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)

x_train = (x_train - mean) / std
x_val = (x_val - mean) / std
```

### 3.3 Upgrade the model (tiny MLP)

```python
from torch import nn

model = nn.Sequential(
    nn.Linear(len(feature_cols), 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()
```

Train for a fixed number of epochs:

```python
from fitstream import epoch_stream, take

for event in take(20)(epoch_stream((x_train, y_train), model, optimizer, loss_fn, batch_size=512, shuffle=True)):
    print(f"epoch={event['step']:03d} train_loss={event['train_loss']:.4f}")
```

## 4) Add validation loss with `augment(...)`

So far we only tracked training loss. FitStream’s `augment(...)` lets you attach extra keys to every event.

FitStream includes a built-in augmenter factory: `validation_loss(...)`.

```python
from fitstream import augment, epoch_stream, pipe, take, validation_loss

events = pipe(
    epoch_stream((x_train, y_train), model, optimizer, loss_fn, batch_size=512, shuffle=True),
    augment(validation_loss((x_val, y_val), loss_fn, key="val_loss")),
)

for event in take(10)(events):
    print(
        f"epoch={event['step']:03d}",
        f"train_loss={event['train_loss']:.4f}",
        f"val_loss={event['val_loss']:.4f}",
    )
```

What just happened?

- `epoch_stream(...)` yields base events: `step`, `train_loss`, `train_time_sec`, `model`
- `augment(...)` merges `{"val_loss": ...}` into each event
- `pipe(...)` composes stages left-to-right

## 5) Stop automatically with `early_stop`

`epoch_stream` is infinite by design, so you need a stop condition. The simplest is `take(...)`.
The more “ML-ish” approach is early stopping on validation loss. In practice, it’s common to use both: `take(max_epochs)`
as a safety cap plus `early_stop(...)` to stop early.

```python
from fitstream import early_stop, take

events = pipe(
    epoch_stream((x_train, y_train), model, optimizer, loss_fn, batch_size=512, shuffle=True),
    augment(validation_loss((x_val, y_val), loss_fn)),
    take(500),
    early_stop(key="val_loss", patience=10, mode="min", min_delta=1e-4),
)

history = list(events)  # finite now
print("stopped at epoch", history[-1]["step"])
```

Notes:

- `early_stop(..., mode="min")` (the default) treats lower values as better.
- Use `mode="max"` for metrics where higher is better (for example `val_acc`).
- `min_delta` is an absolute threshold for improvement, so tiny metric noise does not reset patience.
- It yields events up to (and including) the epoch that triggers stopping.

For an accuracy metric, switch to `mode="max"`:

```python
events = pipe(
    epoch_stream(...),
    augment(...),  # produce "val_acc" on each event
    take(500),
    early_stop(key="val_acc", patience=10, mode="max", min_delta=1e-3),
)
```

## 6) Become a hero: write your own augmenter

An **augmenter** is a function `event -> dict` (or `None`) that adds keys to the event. This is great for:

- extra metrics (MAE, R², accuracy, …)
- model stats (weight norms, gradient norms)
- derived / smoothed values (moving averages)

### 6.1 Example: model parameter norm

```python
from torch import nn

def model_param_norm(event: dict) -> dict[str, float]:
    params = nn.utils.parameters_to_vector(event["model"].parameters())
    return {"param_l2": params.norm().detach().cpu().item()}
```

Use it with `augment(...)`:

```python
from fitstream import augment, pipe

events = pipe(
    epoch_stream((x_train, y_train), model, optimizer, loss_fn, batch_size=512, shuffle=True),
    augment(validation_loss((x_val, y_val), loss_fn)),
    augment(model_param_norm),
)
```

### 6.2 Example: your own validation metric (MAE)

FitStream ships `validation_loss(...)`, but you can write any metric you want.

```python
import torch
from torch import nn

def validation_mae(x_val: torch.Tensor, y_val: torch.Tensor, *, key: str = "val_mae"):
    def compute(event: dict) -> dict[str, float]:
        model = event["model"]
        if not isinstance(model, nn.Module):
            raise TypeError("Expected event['model'] to be a torch.nn.Module")

        was_training = model.training
        model.eval()
        with torch.no_grad():
            preds = model(x_val)
            mae = (preds - y_val).abs().mean()
        if was_training:
            model.train()
        return {key: mae.detach().cpu().item()}

    return compute
```

Add it to the pipeline:

```python
events = pipe(
    epoch_stream((x_train, y_train), model, optimizer, loss_fn, batch_size=512, shuffle=True),
    augment(validation_loss((x_val, y_val), loss_fn)),
    augment(validation_mae(x_val, y_val)),
)
```

## 7) Become a bigger hero: write your own stream processor

An augmenter modifies a single event. A **stream processor** (aka a pipe stage) can do anything that needs state
across events:

- exponential moving averages
- periodic logging
- keeping track of the best metric
- checkpoints

### 7.1 Quick side effects with `tap(...)`

FitStream includes a small helper for side effects: `tap(fn)` calls `fn(event)` and yields the event unchanged.
It’s perfect for lightweight logging or writing metrics to an external system.

```python
from fitstream import augment, epoch_stream, pipe, take, tap, validation_loss

events = pipe(
    epoch_stream((x_train, y_train), model, optimizer, loss_fn, batch_size=512, shuffle=True),
    augment(validation_loss((x_val, y_val), loss_fn)),
    tap(lambda event: print(f"epoch={event['step']:03d} val_loss={event['val_loss']:.4f}")),
)

# Consume a few events to actually run it (streams are lazy).
list(take(5)(events))
```

### 7.2 Learning rate scheduling with `tick(...)`

If you have a `tick(fn)` stage (a no-argument cousin of `tap(...)` that calls `fn()` once per event), it’s a clean way
to integrate learning rate schedulers that step once per epoch.

For example, linear warmup from 10% → 100% over the first 10 epochs:

```python
from torch.optim.lr_scheduler import LinearLR
from fitstream import augment, epoch_stream, pipe, take, tick

scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)

events = pipe(
    epoch_stream((x_train, y_train), model, optimizer, loss_fn, batch_size=512, shuffle=True),
    # Record the LR used for this epoch...
    augment(lambda ev: {"lr": optimizer.param_groups[0]["lr"]}),
    # ...then step the scheduler to set the LR for the next epoch.
    tick(scheduler.step),
    take(50),
)

for ev in events:
    print(ev["step"], ev["train_loss"], ev["lr"])
```

If your scheduler needs a metric (e.g. `ReduceLROnPlateau`), use `tap(...)` instead so you can pass `event["val_loss"]`.

### 7.3 Example: exponential moving average (EMA)

This stage adds a new key like `val_loss_ema` to each event.

```python
from collections.abc import Iterable
from typing import Any

def ema(key: str, *, alpha: float = 0.2, out_key: str | None = None):
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    out_key = out_key or f"{key}_ema"

    def stage(events: Iterable[dict[str, Any]]):
        value_ema: float | None = None
        for event in events:
            value = float(event[key])
            value_ema = value if value_ema is None else (1.0 - alpha) * value_ema + alpha * value
            yield event | {out_key: value_ema}

    return stage
```

### 7.4 Example: print progress every N epochs

```python
from collections.abc import Iterable
from typing import Any

def print_every(n: int, *, keys: tuple[str, ...] = ("train_loss", "val_loss")):
    if n <= 0:
        raise ValueError("n must be >= 1.")

    def stage(events: Iterable[dict[str, Any]]):
        for event in events:
            if int(event["step"]) % n == 0:
                parts = [f"epoch={event['step']:04d}"]
                for k in keys:
                    if k in event:
                        parts.append(f"{k}={float(event[k]):.4f}")
                print(" ".join(parts))
            yield event

    return stage
```

## 8) The “zero → hero” pipeline (put it all together)

Here’s a complete training script with:

- `epoch_stream` training
- built-in `validation_loss` augmentation
- custom `param_l2` augmentation
- EMA smoothing of `val_loss`
- periodic printing
- early stopping
- max epoch cap (`take`)
- saving results to JSONL

```python
from pathlib import Path

import torch
from torch import nn

from fitstream import augment, collect_jsonl, early_stop, epoch_stream, pipe, take, validation_loss

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(len(feature_cols), 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

events = pipe(
    epoch_stream((x_train, y_train), model, optimizer, loss_fn, batch_size=512, shuffle=True),
    augment(validation_loss((x_val, y_val), loss_fn)),
    augment(model_param_norm),
    ema("val_loss", alpha=0.2),
    print_every(10, keys=("train_loss", "val_loss", "val_loss_ema", "param_l2")),
    take(500),
    early_stop(key="val_loss", patience=20, mode="min", min_delta=1e-4),
)

# Write the whole training history to disk (one JSON object per line).
collect_jsonl(events, RUNS_DIR / "california_housing.jsonl")
```

## 9) Next steps

- Use a **true train/val/test** split (here we used the provided “test” split as validation for simplicity).
- Add a `checkpoint_best(...)` stage that saves `event["model"].state_dict()` when `val_loss` improves.
- Replace full-dataset validation with a batched validation pass if your validation set is large.
- Use `collect_pd(...)` to turn your event history into a DataFrame for plotting/analysis.
