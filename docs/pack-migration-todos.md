# 2048 Pack Migration TODOs

Purpose: replace the legacy `.a2pack` container with the RAM-friendly dataset format described in `docs/2048-pack.md`, keep changes minimal, and provide a clean migration path for Rust + PyO3 users.

## Completed
- Core `DataPack` (Rust): structs, save/load with validation, derived indices, rustdoc.
- Parallel default loading: step decode parallelized with Rayon.
- PackBuilder: recursive `.a2run2` loader (parallel), `build`, `write_to_file`.
- New CLI `datapack`: `build`, `validate`, `stats`, `inspect`.
- PyO3 Dataset: `Dataset` with `__len__`, `get_batch`, and `filter_by_score`.
- README and crate docs updated; tests added for Dataset.

## Validation & Tests
- Rust unit tests:
  - Round-trip `save`/`load` (small and larger synthetic sets).
  - Header/offset/length/CRC failure cases.
  - Builder correctness across mixed v1/v2 input.
- Python tests:
  - `Dataset` load, batch shapes/dtypes, filtering behavior, deterministic order with a fixed seed.
- Optional: quick `stats` test (count, total steps, mins/maxes) to mirror builder outputs.

## Migration & Interop
- Provide a CLI to build `pack.dat` directly from `.a2run2`. (Done)
- Optional one-shot converter from `.a2pack` using existing Rust `PackReader` to bootstrap large datasets. (Low priority)

## Future Work (Roadmap)
- Incremental append: `DataPack::append_from_directory` to add new runs and rebuild indices.
- Additional filters/views in Python: by engine, run length, highest tile, time ranges.
- Numpy outputs: return `numpy.ndarray` for boards/dirs to reduce Python overhead.
- Optional `.idx` sidecar: precomputed lookup/index file; rebuildable from `.dat`.
- Cached stats: embed a compact stats block (or sidecar) to avoid rescans.
- Magic/version: consider `b"A2DP"` and explicit endianness; document LE requirement.
- Compression: evaluate zstd at file level; avoid on-disk per-step compression initially.
- Sharding: split very large packs by time/size (e.g., `pack_0001.dat`, `pack_0002.dat`).
- Benchmarks: larger-scale random-read benches (10–100M steps) and memory footprint tracking.
- Migration tool: fast `.a2pack` → `.dat` converter using `PackReader` (if needed).
- Document the deprecation path for `.a2pack` readers once the new flow is stable.

## Decisions To Lock Down (Spec Clarifications)
- Magic/endianness: keep LE; consider a more specific magic (e.g., `b"A2DP"`). For now, enforce `b"2048"` + `version=1` and LE in code + docs.
- `engine` field: store UTF-8 string; use empty string to mean “unknown” (avoid on-disk Option). Document clearly.
- Batch contents (Py): initial minimum = boards only; defer moves/branches until needed.
- Indices: maintain only score/length for v1; additional filters can be computed from `runs` at query time.

## Nice-To-Haves (Defer unless needed)
- Separate `.idx` file for faster run lookups (rebuildable from `.dat`).
- JSONL export parity (per-run/per-step) for analysis; port only if users rely on it.
- Cached stats block in trailer (like legacy pack) to avoid rescans; optional if builder already reports stats.

## Out Of Scope (v1)
- On-disk compression or per-step deltas.
- Memory-mapped random access optimizations (design assumes full in-RAM use).
- Complex filter/query language beyond basic helpers.

## Notes From Review
- Fix checksum write/read sequencing vs. the sketch in `2048-pack.md`; compute CRC before appending it during save.
- Validate bounds thoroughly before any `from_raw_parts` usage; prefer copying into `Vec<Step>` on load for safety.
- Keep changes minimal and additive; do not break existing `PackReader` until new `Dataset` + tests are in place.
