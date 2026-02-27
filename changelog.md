# Changelog

This changelog is organized by the repository's Git tags.

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
