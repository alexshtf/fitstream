# Changelog

This changelog is organized by the repository's Git tags.

## v0.5.0 (2026-10-31)

Changes since `v0.4.1`.

### Breaking changes

- `best_model` is now an augmenter: use it with `augment` to add the best observed
  model state to each event as `best_state`, instead of using it with `tap`.
- Removed the `BestModel` class and `load_into` method. Load the collected state
  with PyTorch's `model.load_state_dict` instead.

To collect metrics and retain the best state from a stream containing `val_loss`:

```python
import fitstream as fts

events = fts.pipe(events, fts.augment(fts.best_model("val_loss")))
history, reduction = fts.collect(
    events,
    exclude=["best_state"],
    reducers={"best_state": fts.last_value},
)
model.load_state_dict(reduction["best_state"])
```

### Added

- `collect`, `collect_pd`, and `collect_jsonl` can aggregate event fields using
  an optional `reducers` mapping. Each reducer combines the accumulated value
  with the next field value, starting with the first event's value.
- Reducers read fields independently of `include` and `exclude`, so snapshots
  can be excluded from history while retaining the final best state.
- With reducers, `collect` and `collect_pd` return `(history, reduction)`;
  `collect_jsonl` writes records to disk and returns the reduction dictionary.
  Without reducers, existing return values are unchanged.
- `last_value` retains the latest field value when used as a reducer.

### Changed

- Updated the README and tutorial to demonstrate best-model augmentation and
  aggregation during collection.

## v0.4.1 (2026-10-26)

Changes since `v0.4.0`.

### Added

- Live learning curves in notebooks with `live_plot`. Visualize training and
  validation metrics as training runs to monitor convergence and spot signs of
  overfitting, directly within your existing training pipeline.

## v0.4.0 (2026-10-24)

Changes since `v0.3.1`.

### Breaking changes

- `take(n)` now consumes at most `n` upstream events. Previously, fully consuming
  the limited stream fetched one additional event; `take(0)` also fetched an event.
  With `epoch_stream`, this could train an extra epoch without yielding its metrics.
  For example, fully consuming `take(10)(epoch_stream(...))` now trains 10 epochs
  instead of 11, and `take(0)` performs no training.
- If your code depended on that extra advancement, request the intended number of
  events explicitly. Final model weights and upstream side effects may differ
  from earlier versions even when the yielded event count is unchanged.

### Added

- `evaluate(val_data, *, metrics, last_label=True)` computes multiple named metrics
  from one model forward pass. Use it with `augment` to add Python float values to
  each event. It supports multiple model inputs and unlabeled data.
- `best_model` and `BestModel` retain an independent snapshot of the best observed
  model state. Use the tracker with `tap`, then call `load_into(model)` after
  training. Both minimization and maximization are supported, with optional CPU
  snapshot storage.
- `fitstream.__version__` exposes the installed package version.

### Changed

- `validation_loss` delegates to `evaluate` while keeping its existing signature
  and metric keys. Evaluation restores each module's previous training/evaluation
  mode, including mixed modes and failures during inference or metric computation.
- Updated the README and tutorial with best-model tracking and shared validation
  metric examples.
- Added an automated release workflow to validate and build the package, publish
  to PyPI, update version metadata and the changelog, and create a GitHub release.

## v0.3.1 (2026-04-02)

Changes since `v0.3.0`.

### Added
- `epoch_stream(..., step_offset=...)` to offset emitted step numbers without changing the default `step=1, 2, ...` behavior.

### Changed
- Updated `epoch_stream(...)` docstrings and tutorial text to document step offsets.
- Added test coverage for offset step numbering and invalid `step_offset` values.

## v0.3.0 (2026-02-26)

Changes since `v0.2.0`.

### Added
- Open `Event` typing for stream events.
- Extra-field coverage for event payloads in tests.

### Changed
- Tutorial updates to reflect open `Event` typing and event payload usage.
- Clarified event-related docstrings and removed redundant tutorial text.
- General documentation fixes.

## v0.2.0 (2026-02-06)

Changes since `v0.1.1`.

### Added
- `ema(...)` stream stage for metric smoothing in `src/fitstream/fit.py`.
- `print_keys(...)` callback helper for compact metric logging in `src/fitstream/fit.py`.
- `doc-open` make target (`mkdocs serve --open`) in `Makefile`.
- `context7.json` metadata for Context7 docs integration.

### Changed
- `tap(...)` now supports optional scheduling controls:
  - `every` (run callback every N events)
  - `start` (1-based start index)
- `early_stop(...)` now supports:
  - `mode="min" | "max"`
  - `min_delta` (absolute improvement threshold)
- `src/fitstream/__init__.py` now exports `ema` and `print_keys`.
- `README.md` and tutorial examples were updated to use:
  - `tap(print_keys(...), every=...)`
  - `ema(..., decay=...)` and `ema(..., half_life=...)`
  - explicit early-stop settings (`mode`, `min_delta`)
- Added project badges to `README.md`.

### Compatibility Notes
- Existing `tap(fn)` usage remains valid (new arguments are optional).
- Existing `early_stop(key, patience)` usage remains valid with defaults.
- `ema(...)` requires exactly one of `decay` or `half_life`.

## v0.1.1 (2026-02-03)

Initial tagged release.

### Added
- Core stream primitives: `epoch_stream(...)`, `augment(...)`, `pipe(...)`, `take(...)`, `tap(...)`, `tick(...)`, `early_stop(...)`, and sink helpers.
- Project setup: packaging metadata, `Makefile`, CI workflow, MkDocs docs, and Read the Docs config.
- Apache 2.0 license and initial docs/tests coverage.

### Changed
- Made `augment(...)`, `take(...)`, and `early_stop(...)` curried/pipe-friendly.
- Improved typing/package metadata and fixed early README/docs issues.
