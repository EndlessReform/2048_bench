# TODOs for Packfile + Tooling

These are non-blocking improvements that can be added incrementally as needs arise.

## Stats Cache (Footer)
- Store a small stats block (postcard) with: `count`, `total_steps`, `min_len`, `max_len`, `mean_len`, step-length quantiles (`p50`, `p90`, `p99`).
- Reader prefers the cached block; falls back to compute if absent.

## Filters + Ranges
- CLI and PyO3: accept index ranges (`10-20,50,70-75`) and filters:
  - `len>=N`, `len<=M`, `highest_tile>=T`, `engine_str~substr`.
- Apply to `extract`, `to-jsonl`, and iterators.

## Progress + Verbose UX
- Progress bars for long ops (`pack`, `to-jsonl`, recompute stats); `--quiet` to disable.

## Optional Compression
- Per-entry `zstd`/`lz4` with parallel encode/decode; default off. Feature-gated.

## JSONL Schema Toggles
- Control output granularity: per-run vs per-step, selected fields, include `engine_str` optionally.
- Optional compression to `.jsonl.zst` for large exports.

## Py Dataloader Niceties
- `iter_batches(prefetch=..., workers=..., drop_last=...)`.
- Distributed shuffles via `(rank, world_size)` downsampling.

## Fast Peek Meta
- Add minimal per-entry metadata (e.g., `steps`, `highest_tile`, hashed `engine_str`) to index or stats block to avoid decode for filtering.

## Validation Modes
- `validate --full` to checksum all entries; `--sample N` for spot-checks; potential `--repair-index`.

## I/O Polishing
- Use `BufWriter` consistently; align writes; avoid tiny tail flushes.

