# 2048 Pack Format — Implementation Notes & Errata

This document calls out clarifications and small deviations from `docs/2048-pack.md` as implemented in the codebase.

## Header & Trailer
- Checksum write: compute CRC32C over all bytes written so far, then append the 4‑byte checksum at the end. The original sketch wrote the file and then tried to read it back before writing the checksum; our implementation computes in-memory and appends directly.
- Endianness: all multi-byte values are little‑endian. We enforce `magic=b"2048"` and `version=1` on load; consider a more specific magic like `b"A2DP"` in a future version.

## Step Layout (32 bytes)
- Implemented field order (both in memory and on disk):
  - `board: u64`
  - `run_id: u32`
  - `index_in_run: u32`
  - `move_dir: u8` (0..=3)
  - `_padding: [u8; 15]` (reserved)
- Rationale: this ordering guarantees a stable 32‑byte layout with `#[repr(C)]` while keeping alignment simple. It is functionally equivalent to the spec; readers/writers both use this order.

## Parallel Loading
- Loading defaults to parallel decoding of the steps region: the fixed‑width records are split into equal chunks and decoded concurrently into a preallocated `Vec<Step>`. Run metadata remains sequential (small).

## Indices & Filtering
- We maintain in‑memory derived indices: `runs_by_score` and `runs_by_length`. We intentionally omit a separate `step_to_run` table because `Step` already contains `run_id`.
- Python `Dataset` currently exposes `filter_by_score`. Additional filters (engine, time windows, length) are planned.

## Optional Files
- The `pack.idx` sidecar is optional and currently not implemented. All lookups are in RAM. We may introduce a rebuildable index file later if needed.

## Append / Incremental Updates
- The implementation does not yet include `append_from_directory`. It is on the roadmap; the initial release targets full rebuilds.

