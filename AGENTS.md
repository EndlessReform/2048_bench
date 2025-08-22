# Engineering Guide

This repo aims for a small, fast 2048 engine with a clear, stable library API and a couple of focused binaries. Keep changes minimal, idiomatic, and performance‑aware.

## Rustdoc Standards

- Document all major public APIs with Rustdoc (`///`).
  - Types and constructors: `Board`, `Move`, AI types, trace types.
  - Core ops: `Board::shift`, `Board::make_move`, `insert_random_tile`, scoring, game‑over checks.
  - Any public helpers used by consumers.
- Keep docs concise. Prefer small, runnable examples.
  - Use seeded RNG (`StdRng::seed_from_u64(_)`) where randomness appears for deterministic doctests.
  - Prefer method‑based examples over free functions, but mention both when relevant.
- Crate overview lives in `src/lib.rs` (short description + quick start).
- Features: Call out feature flags (e.g., `wasm`) in item docs when applicable.
- Doctests must pass: run `cargo test` after doc changes. Update examples when APIs change.

Developer checklist for public API changes:
- Update/add `///` docs with a brief summary and example if helpful.
- Run `cargo test` (unit + doctests).
- If behavior/semantics changed, update crate‑level docs and README snippets.

## Code Style & Structure

- Favor idiomatic Rust and clarity over cleverness.
- Library vs binaries:
  - Keep core logic in the library (no CLI/logging in core paths).
  - Binaries live under `src/bin/` and may pull CLI‑only deps.
- Visibility:
  - Keep internals private or `pub(crate)`; expose a small, coherent public surface.
  - `Board` is the main state type; provide methods for common ops and an escape hatch to raw bits.
- Performance:
  - Hot paths should avoid unnecessary allocation and branching; prefer stack/packed forms.
  - Unsafe blocks must be justified by perf and covered by tests; do not expand unsafe casually.
- Randomness:
  - Core methods accept `&mut impl Rng` for determinism; provide a thread‑RNG convenience separately.
- Features:
  - Gate optional surfaces (e.g., WASM) behind features.

## Testing

- Run `cargo test` for unit + doctests.
- Add targeted tests for new functionality; keep them fast.
- If perf‑critical behavior changes, measure locally (e.g., Criterion). Ask for approval before adding dev deps.

## API Stability

- Avoid breaking changes; prefer additive changes.
- If a rename/refactor is needed, provide thin wrappers and deprecate gradually.

This guidance keeps the engine lean, the API pleasant to use, and the docs trustworthy without excess ceremony.
