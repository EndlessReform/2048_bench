# 2048 AI

The goal of this project was to produce a highly optimised engine for the game 2048. The game engine implements all the behaviour needed to play the game 2048, such as shifting and merging tiles, inserting random new tiles, and calculating the score. My MSc thesis was on using artificial intelligence methods to find strategies for the game of 2048. The aim was to find strategies which human players could understand and replicate whilst maximising performance. This required testing millions of possible strategies, making billions of moves. Therefore, having a fast game engine was an essential foundation. To stress test the engine an expectimax algorithm was used. This algorithm can perform very well on 2048, often reaching the 32,768 tile. To achieve a score so large, more than 18,000 moves and over 90 billion game states need to be considered. If the engine was not extremely well optimised this would take a very long time.

This optimized 2048 engine is designed for high-performance AI testing and can handle millions of game simulations.

**Local Usage**
- **Build**: `cargo build`
- **Run (single-threaded)**: `cargo run`
- **Run (parallel)**: `cargo run --bin parallel [-- <flags>]`

## Documentation

Build and open the Rust API docs locally:

- Build docs: `cargo doc`
- Open docs: `cargo doc --open`

Notes:
- Doctests run with `cargo test` (examples in the docs are compiled and executed).

Quick example (from the docs):

```rust
use ai_2048::engine::{self as GameEngine, Board, Move};
use rand::{rngs::StdRng, SeedableRng};

GameEngine::new();
let mut rng = StdRng::seed_from_u64(42);
let b0 = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
let b1 = b0.shift(Move::Left);
assert!(b1.score() >= 0);
```

**Parallel Runner Flags**
- `--quiet`: suppress the live status line (no console output during run).
- `--steps <N>`: stop after N moves (global counter).
- `--stop-score <N>`: stop once score ≥ N (checked every 100 moves).
- `--stop-tile <N>`: stop once highest tile value ≥ N (e.g., 2048) (checked every 100 moves).
 - `--out <FILE>`: write a single run to this file (`.a2run`).

Subcommands:
- `forever`: continuously write runs until stopped or cap reached
  - `--out-dir <DIR>`: output directory (required)
  - `--max-gb <F>`: cap total size in gigabytes as a float (default 10.0)
  - `--quiet`: suppress spinner status
  - Per-run stopping conditions (same semantics as one-off mode):
    - `--steps <N>`: stop each run after N moves
    - `--stop-score <N>`: stop once score ≥ N (checked every 100 moves)
    - `--stop-tile <N>`: stop once highest tile ≥ N (checked every 100 moves)

Examples:
- Run parallel until 25k moves: `cargo run --bin parallel -- --steps 25000`
- Run parallel until score ≥ 50,000: `cargo run --bin parallel -- --stop-score 50000`
- Run parallel until reaching 2048: `cargo run --bin parallel -- --stop-tile 2048`
- Quiet compute (no status line): `cargo run --bin parallel -- --steps 25000 --quiet`
 - Write a binary trace: `cargo run --bin parallel -- --steps 25000 --out traces/run.a2run`

### Continuous Generation Mode

- Generate runs into a directory until stopped or size cap reached:
  - `cargo run --bin parallel -- forever --out-dir traces --max-gb 10 --stop-tile 4096`
  - Files are sharded by day (`dXXXXXXXX/`) with names `run-<unix_s>-<nanos>.a2run` for good sorting and low collision probability.
  - Minimal status shows run count and total size. Stop with Ctrl-C.
  - Default cap is ~10 GB if `--max-gb` is not provided.

**What The Status Line Shows**
- Elapsed time, total moves, moves/sec, and running score (score updates every 100 moves).
- Rendering runs on a background thread; the compute loop only does atomic counters to avoid overhead.

**Engines**
- `Expectimax` (single-threaded): used by `src/main.rs` (default `cargo run`).
- `ExpectimaxMultithread` (Rayon-based): used by `src/bin/parallel.rs`.
  - Parallelizes both max nodes (moves) and chance nodes (empty tile insertions) with thresholds to reduce overhead.

## Traces and Visualization

- Binary runs are saved per-file with extension `.a2run`.
- Format matches `ARCHITECTURE.md` (magic `A2T1`, version `1`, LE), with:
  - Fixed metadata: `steps`, `start_unix_s`, `elapsed_s`, `max_score`, `highest_tile`.
  - Optional engine string: free-form run metadata.
  - Payload: states `u64[steps+1]`, moves `u8[steps]`.
  - Trailer: CRC32C of preceding bytes.

Filenames and sharding:
- Generator shards by day since epoch (`dXXXXXXXX/`) and names files `run-<unix_s>-<nanos>.a2run`.
- Keeps folders smaller and filenames sortable; safe to merge across sessions.

### Visualize a run

- Create a run: `cargo run --bin parallel -- --steps 2000 --out traces/example.a2run`
- Pretty-print it: `cargo run --bin visualize -- traces/example.a2run`
  - Shows metadata, steps, moves/sec, and renders the final board.
  - Uses a 16-color 2048-themed palette and ASCII box for tiles.

Programmatic loading:
- Reader/writer live in `ai_2048::trace`.
  - `parse_run_file(path) -> Run` validates checksum and returns metadata, states, and moves.
  - `write_run_to_path(path, &meta, &states, &moves)` writes a single run.
  - The format is versioned; `engine_str` is optional (e.g., for git SHA) and does not affect compatibility.

### Packfile (a2pack)

When working with tens of thousands of small `.a2run2` files, filesystem overhead can dominate. Use the `a2pack` binary to pack many runs into a single indexed file for fast sequential scans and O(1) random access via the library API.

Pack a directory recursively:

```
cargo run -q -p ai-2048 --bin a2pack -- pack --input /path/to/runs --output dataset.a2pack
```

Options:
- `--input DIR`: directory to scan recursively for `*.a2run2` files.
- `--output FILE`: output `.a2pack` file path.
- `--align BYTES` (default `4096`): alignment for index/data and entries.
- `--entry-crc` (default `true`): store per-entry CRC32C for payload verification.

Reading a packfile from Rust:

```
use ai_2048::serialization::PackReader;

let pack = PackReader::open("dataset.a2pack")?;
println!("runs: {}", pack.len());
let run = pack.decode_auto_v2(0)?; // v1 auto-upgraded to v2
// fast JSONL export (parallel)
// Per-run (one JSON object per run)
pack.to_jsonl("runs.jsonl", true, false)?;
// Per-step (one JSON object per step; includes run_uuid, step_idx, and pre_board as 16 exponents)
pack.to_jsonl("steps.jsonl", true, true)?;
```

Reading a packfile from Python (PyO3):

```
import ai_2048 as a2

r = a2.PackReader.open("dataset.a2pack")
print(r.stats.count, r.stats.mean_len)
run0 = r.decode(0)                # -> RunV2 (v1 auto-upgraded)
batch = r.decode_batch([0, 5, 42])
# Per-run JSONL
r.to_jsonl("runs.jsonl", parallel=True)
# Per-step JSONL (flattened; pre_board is 16 exponents)
r.to_jsonl("steps.jsonl", parallel=True, by_step=True)
```

Additional CLI commands:

- Validate: `cargo run -q -p ai-2048 --bin a2pack -- validate --packfile dataset.a2pack`
- Stats: `cargo run -q -p ai-2048 --bin a2pack -- stats --packfile dataset.a2pack`
- To JSONL (per-run): `cargo run -q -p ai-2048 --bin a2pack -- to-jsonl --packfile dataset.a2pack --output runs.jsonl --parallel`
- To JSONL (per-step): `cargo run -q -p ai-2048 --bin a2pack -- to-jsonl --packfile dataset.a2pack --output steps.jsonl --parallel --by-step`
- Extract runs: `cargo run -q -p ai-2048 --bin a2pack -- extract --packfile dataset.a2pack --indices 0,5,42 --output out/`
- Inspect run: `cargo run -q -p ai-2048 --bin a2pack -- inspect --packfile dataset.a2pack --index 123`

### Dataset Pack (.dat) — RAM-friendly

For shuffled training workloads, load all steps into RAM using the dataset pack format. This flattens all runs into a single steps array with lightweight run metadata. Loading is parallelized.

Build a dataset pack from `.a2run2` files:

```
cargo run -q -p ai-2048 --bin datapack -- build --input /path/to/runs --output dataset.dat
```

Validate and inspect:

```
cargo run -q -p ai-2048 --bin datapack -- validate --pack dataset.dat
cargo run -q -p ai-2048 --bin datapack -- stats --pack dataset.dat
cargo run -q -p ai-2048 --bin datapack -- inspect --pack dataset.dat --index 0
```

Programmatic usage (Rust):

```rust
use ai_2048::serialization::{PackBuilder, DataPack};
use std::path::Path;

// Build and save
// let pb = PackBuilder::from_directory(Path::new("runs/"))?;
// pb.write_to_file("dataset.dat")?;

// Load into memory (parallel decode by default)
// let dp = DataPack::load(Path::new("dataset.dat"))?;
// assert!(dp.steps.len() > 0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Notes:
- Only `.a2run2` (postcard-v2) runs are supported in the builder.
- The binary layout and motivation are described in `docs/2048-pack.md`.

Incremental append new runs (atomic rewrite):

```
cargo run -q -p ai-2048 --bin datapack -- append --pack dataset.dat --input /path/to/new_runs --output dataset.dat
```

- Writes to `dataset.dat.tmp` and atomically renames to `dataset.dat` on success.
- Keeps memory bounded to the new runs; existing `.dat` is streamed, not fully loaded.

Merge two packs (A then B):

```
cargo run -q -p ai-2048 --bin datapack -- merge --a a.dat --b b.dat --output merged.dat
```

- Preserves order A then B. Adjusts B run ids and step indices automatically.

Time load and index creation costs:

```
cargo run -q -p ai-2048 --bin datapack -- time-load --pack dataset.dat
```

- Prints file size, runs, steps, total load time, and reindex time.

Benchmark random batch reads (e.g., batch size 768):

```
cargo run -q -p ai-2048 --bin datapack -- bench-batches --pack dataset.dat --batch 768 --iters 10000 --pregenerate true
```

- `--pregenerate true` excludes RNG overhead from timing. Increase `--iters` for steadier numbers on large packs.



**Development**
- Tests: `cargo test` (all tests should pass).
- Format/lints: follow standard Rust conventions; no custom toolchain required.

## Benchmarks (Criterion)

This repo includes Criterion microbenchmarks for engine ops and the heuristic. Cargo separates tests and benches:
- `cargo test` runs unit tests and doctests only.
- `cargo bench` builds and runs benches only (tests are not executed). This is why “tests look ignored” when running benches; they are simply not part of the bench profile.

Run benchmarks
- Engine ops (default benches):
  - Run: `cargo bench`
  - Compile only: `cargo bench --no-run`
- Heuristic bench (feature-gated to keep the public surface minimal):
  - Run: `cargo bench --features bench-internal`
  - Compile only: `cargo bench --features bench-internal --no-run`

Quiet/locked/offline (useful for CI or predictable runs)
- Quiet: add `-q` (e.g., `cargo bench -q`)
- Locked deps: add `--locked` to avoid updating `Cargo.lock`
- Offline: add `--offline` if all deps are already cached

Notes
- Benches seed RNGs (`StdRng::seed_from_u64(_)`) and warm the engine tables for determinism.
- Parallel policy benches (if added) can be stabilized by pinning Rayon threads via `RAYON_NUM_THREADS=N`.

**Performance Notes**
- On multi-core systems, the parallel runner saturates cores and typically achieves 2–3× throughput over the single-threaded engine (exact gains depend on depth/board state).
- The status line and periodic score/tile checks are designed to be negligible overhead (checked every 100 moves).
