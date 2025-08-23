# Readonly Packfile Design for a2run2 Datasets

This document proposes a compact, readonly packfile format and APIs to efficiently read large collections of `a2run2` runs. It targets low syscall overhead, high-throughput sequential scans, and O(1) random access suitable for dataloaders. It integrates cleanly with the existing v1 (`trace.rs`) and v2 (`serialization/v2.rs`) codecs and exposes ergonomic Rust and PyO3 surfaces.

## Goals

- Minimal, stable on-disk format with fast open and validation.
- O(1) random addressing of runs; zero extra copies beyond what decoding needs.
- High-throughput sequential iteration and batched random access for training.
- Bulk export to JSONL from Rust and Python (via PyO3), avoiding per-file Python overhead.
- Parallel read/decode using `rayon` (already in the workspace).
- Simple CLI to build, validate, inspect, and convert packs.
- No write/update-in-place; packfiles are immutable once built.

## Workload Assumptions

- 5k–100k runs, each ~60 KiB (v1 or v2 encoded), 1.7k–2k steps typical.
- Readers on modern SSD + multi-core CPUs.
- Primary patterns: full scans for conversion; batched random access for dataloaders; occasional single-run lookup.

## File Format

Binary, little-endian. Designed for memory mapping and direct slicing. Stores raw `a2run2` payloads (v1 or v2) verbatim, with per-entry metadata in an index table.

### Layout

```
| Magic(8) | Version(u32) | Flags(u32) | Count(u64) | HeaderCrc(u32) |
| Index[Count] | OptionalExtra | DataRegion | Footer |
```

- Magic: `b"A2PACK\0\0"` (8 bytes) to distinguish from `a2run2`.
- Version: `1`.
- Flags: bitfield; reserved for compression policy, index options.
- Count: number of runs.
- HeaderCrc: CRC32C over the fixed header (magic..count).

#### Index Entry (fixed 32 bytes)

```
struct IndexEntry {
  u64 offset;     // absolute offset into DataRegion
  u32 length;     // bytes length of the run payload
  u16 kind;       // 1=v1, 2=v2
  u16 flags;      // reserved; e.g., per-entry compression type
  u32 crc32c;     // optional verification of payload (0 if omitted)
  u32 reserved;   // padding/reserved for future fields
}
```

- `offset` points into DataRegion start; DataRegion is 8–4096 byte aligned.
- Entries are sorted by `offset`; `length` may be 0 for tombstones (not used in v1).

#### Footer (fixed 32 bytes)

```
| DataEndOffset(u64) | IndexCrc(u32) | DataCrc(u32 | 0) | StatsOffset(u64) | FooterCrc(u32) |
```

- `IndexCrc`: CRC32C over the raw index bytes.
- `DataCrc`: optional whole-data-region CRC (0 if omitted or compressed per-entry).
- `StatsOffset`: 0 or absolute offset of an optional serialized stats block (postcard/JSON) with summary statistics.

### Alignment and Compression

- Alignment: pack entries to 4 KiB by default (`--align` tunable). Improves OS readahead and keeps slices page-aligned for mmap.
- Compression: optional per-entry compression (e.g., `zstd` or `lz4`) indicated in `flags`. Start uncompressed; add compression behind a feature flag if needed. Readers decide parallel decompression per batch.

### Integrity

- File-level: validate magic/version; CRCs for header, index, and optional data.
- Entry-level: if `crc32c != 0`, verify payload slice before decode for random reads; sequential conversions may skip per-entry CRC after global validation for throughput.

## Rust API

Add a small `pack` module in the library (public surface kept small):

```rust
pub struct PackReader { /* mmapped file and views */ }
pub struct RunSlice<'a> { pub bytes: &'a [u8], pub kind: RunKind }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunKind { V1, V2 }

pub struct PackStats { /* summarized at build time or lazily */ }

impl PackReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, PackError>;
    pub fn len(&self) -> usize;
    pub fn kind(&self, i: usize) -> RunKind;
    pub fn get_slice(&self, i: usize) -> Result<RunSlice<'_>, PackError>; // borrowing subslice

    // Decode helpers (single)
    pub fn decode_v2(&self, i: usize) -> Result<serialization::RunV2, PackError>;
    pub fn decode_v1(&self, i: usize) -> Result<trace::Run, PackError>;
    pub fn decode_auto_v2(&self, i: usize) -> Result<serialization::RunV2, PackError>; // v1->v2 via from_v1

    // Iteration
    pub fn iter_indices<'a>(&'a self, idxs: impl IntoIterator<Item=usize> + 'a) -> impl Iterator<Item=RunSlice<'a>> + 'a;
    pub fn iter(&self) -> impl Iterator<Item=RunSlice<'_>>;

    // Batched parallel decode (rayon inside)
    pub fn decode_batch_v2(&self, idxs: &[usize]) -> Result<Vec<serialization::RunV2>, PackError>;
    pub fn decode_batch_auto_v2(&self, idxs: &[usize]) -> Result<Vec<serialization::RunV2>, PackError>;

    // Bulk conversion (JSONL)
    pub fn to_jsonl<P: AsRef<Path>>(&self, out: P, parallel: bool) -> Result<(), PackError>;

    // Stats
    pub fn stats(&self) -> Result<PackStats, PackError>;
}
```

Implementation notes:

- Backed by `memmap2::Mmap` or `std::fs::File` + `read_at` fallback. Use slices (`&[u8]`) to feed existing decoders: `postcard::from_bytes` for v2, `trace::parse_run_bytes` for v1.
- Parallel decode uses `rayon::ParallelIterator` over indices; capture per-thread scratch to reduce allocs where possible.
- JSONL export: chunk indices into shards (e.g., 256–1024 runs per shard), decode in parallel, serialize to strings in worker threads, then write with a single buffered writer in submission order (or use per-shard temp buffers appended atomically to preserve order while keeping I/O sequential).
- Stats: if present in footer (`StatsOffset`), read and return. Otherwise compute lazily once with a parallel pass: counts, min/max/mean run length, total steps, distribution of `highest_tile`, and optional engine string counts. Cache result inside `PackReader`.

### Errors

`PackError` wraps I/O, mapping, and decode errors, and exposes variants for checksum mismatches and malformed headers.

## CLI: `a2pack`

Small, focused binary (lives under `src/bin/`) built on the library API.

- `a2pack pack --input DIR --output dataset.a2pack [--align 4096] [--compress none|zstd|lz4] [--workers N]`
  - Streams directory entries, reads each `*.a2run2`, determines kind (v1/v2) by magic/version, appends to DataRegion, and emits index entries. Computes CRCs and optional stats in one pass. Avoids loading the entire directory list in memory.
- `a2pack validate --packfile dataset.a2pack` — validates header/index/footer and spot-checks a sample of entries.
- `a2pack stats --packfile dataset.a2pack` — prints PackStats.
- `a2pack to-jsonl --packfile dataset.a2pack --output runs.jsonl [--parallel] [--workers N]`
- `a2pack extract --packfile dataset.a2pack --indices 0,5,42 --output out_dir/` — materialize selected runs.
- `a2pack inspect --packfile dataset.a2pack --index 123` — prints a decoded summary for a single run.

Implementation details:

- Use buffered I/O for pack building; align entries to the requested boundary.
- Maintain a running stats accumulator while packing to amortize cost.
- Keep CLI thin — no business logic beyond gluing to the library.

## PyO3 API

Expose a Pythonic, batch-friendly interface to minimize GIL crossings and filesystem syscalls.

```python
import ai_2048 as a2

r = a2.PackReader.open("dataset.a2pack")
len(r)                  # -> number of runs
r.kind(0)               # -> "v1" or "v2"
run = r.decode(0)       # -> a2.RunV2 (v1 auto-upgraded)

for run in r:           # sequential iteration
    ...

# Batched random access (good for dataloaders)
batch = r.decode_batch([1, 100, 42, 7])  # returns list[a2.RunV2]

# Shuffle+batch iterator (internal parallelism, stable yield order)
it = r.iter_batches(batch_size=256, shuffle=True, seed=123, workers=None)
for batch in it:
    ...  # list[a2.RunV2]

# Bulk JSONL export (fast, Rust-side)
r.to_jsonl("runs.jsonl", parallel=True, workers=None)

# Stats
s = r.stats
print(s.count, s.total_steps, s.max_len)
```

Proposed classes/functions (PyO3):

- `class PackReader`
  - `@staticmethod open(path: PathLike) -> PackReader`
  - `__len__(self) -> int`
  - `kind(self, i: int) -> str` — "v1" | "v2"
  - `decode(self, i: int) -> RunV2` — v1 auto-converted via `from_v1`
  - `decode_batch(self, indices: Sequence[int]) -> list[RunV2]`
  - `__iter__(self)` yielding `RunV2` (sequential scan; releases GIL; parallel decode + ordered yield)
  - `iter_indices(self, indices: Sequence[int]) -> Iterator[RunV2]`
  - `iter_batches(self, batch_size: int, shuffle: bool = False, seed: Optional[int] = None, workers: Optional[int] = None) -> Iterator[list[RunV2]]`
  - `to_jsonl(self, path: PathLike, parallel: bool = True, workers: Optional[int] = None) -> None`
  - `stats: PackStats` (property)

- `class PackStats`
  - `count: int`
  - `total_steps: int`
  - `min_len: int`
  - `max_len: int`
  - `mean_len: float`
  - `p50_len / p90_len / p99_len: int` (optional)
  - `highest_tile_hist: dict[int, int]` (optional)
  - `engine_counts: dict[str, int]` (optional)

Implementation notes:

- All heavy work happens in Rust; Py methods release the GIL (`Python::allow_threads`).
- Batch methods use `rayon` internally; `workers` can cap the Rayon thread pool for these calls.
- For iteration, decode in parallel into a small ring buffer and yield in-order to the Python side to balance throughput and latency.
- Data stays in Rust until fully decoded; Python receives existing `PyRunV2` objects already provided by the bindings.

## Bulk JSONL Conversion

Rust-side serializer to JSONL with a predictable schema for downstream tools (DuckDB, HF datasets):

- Suggested record fields (per step):
  - `run_idx` (u64), `step_idx` (u32)
  - `pre_board` (u64), `chosen` (str or small int),
  - `branches` (optional list[float|null] len=4)
  - `final_board` only for a terminal record or separate run-level JSON line
  - Run-level metadata mirrored once per run or written as a separate stream depending on consumer preference.

Serialization strategy:

- Shard runs into chunks; each worker decodes and serializes a shard to a local buffer; a coordinator writes buffers sequentially to the destination to ensure stable ordering and large, contiguous writes.
- Optionally support `--runs-only` and `--steps` modes to control output granularity.

## Parallelism

- Use `rayon` for parallel decode and serialization shards. Default to a sensible number of threads; allow per-call override without altering global thread pool behavior in unrelated parts of the library.
- Maintain bounded memory usage by limiting in-flight decoded runs per worker and using preallocated buffers for JSON lines.

## Performance Considerations

- Syscalls: the pack eliminates per-file open/read/close per run; a single mapped file enables fewer kernel crossings and better readahead.
- mmap vs read: prefer mmap; fall back to `pread` if needed. Postcard/v1 decoders work on `&[u8]` slices directly.
- Checksums: validate once for sequential passes; for random access, verify per-entry checksum on demand (configurable).
- Compression: start uncompressed; if added later, choose a decompressor that can decode directly from a slice per entry and parallelize across entries.

## Compatibility and Versioning

- Support both v1 and v2 payloads; `decode_auto_v2` upgrades v1 to v2 using existing `from_v1` logic.
- Strict versioned header; future versions can add an alternate index block referenced by a new footer while retaining legacy readers.

## Testing Plan

- Roundtrip: build a pack from a directory with mixed v1/v2 runs; reopen; verify `len`, random slices, and decoded equality with original files.
- Integrity: corrupt selected bytes and assert proper error variants.
- Parallel determinism: ensure batch decode and JSONL export are deterministic given seed and ordering flags.
- Py tests: open pack, iterate, batch, export JSONL; assert counts and spot-check decoded content using `uv run --locked pytest`.

## Future Extensions (Feature-Gated)

- Parquet/Arrow export for columnar analytics. This will require adding dependencies and should be gated behind a Cargo feature with explicit approval.
- Optional per-pack compression (zstd/lz4) and dictionary training.
- Async I/O variants for network filesystems, with the same high-level API.
- Secondary indexes (e.g., by `highest_tile`, length bins) to speed filtered scans.

---

This design stays small and fast, leveraging the existing serializers and `rayon` for parallelism, while giving Python consumers high-throughput, low-overhead access patterns and a simple CLI for dataset preparation.

