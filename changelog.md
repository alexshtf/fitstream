# Changelog

This release includes changes from `v0.1.1` to `72db3be` (`master`).

## Added
- `ema(...)` stream stage for metric smoothing in `src/fitstream/fit.py`.
- `print_keys(...)` callback helper for compact metric logging in `src/fitstream/fit.py`.
- `doc-open` make target (`mkdocs serve --open`) in `Makefile`.
- `context7.json` metadata for Context7 docs integration.

## Changed
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

## Tests
- Added EMA coverage: `tests/test_ema.py`.
- Added `print_keys` coverage: `tests/test_print_keys.py`.
- Expanded `tap` coverage for `every`/`start`: `tests/test_tap.py`.
- Expanded `early_stop` coverage for `mode`/`min_delta`: `tests/test_early_stop.py`.

## Compatibility Notes
- Existing `tap(fn)` usage remains valid (new arguments are optional).
- Existing `early_stop(key, patience)` usage remains valid with defaults.
- `ema(...)` requires exactly one of `decay` or `half_life`.
