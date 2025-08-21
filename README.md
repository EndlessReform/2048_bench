# 2048 AI

The goal of this project was to produce a highly optimised engine for the game 2048. The game engine implements all the behaviour needed to play the game 2048, such as shifting and merging tiles, inserting random new tiles, and calculating the score. My MSc thesis was on using artificial intelligence methods to find strategies for the game of 2048. The aim was to find strategies which human players could understand and replicate whilst maximising performance. This required testing millions of possible strategies, making billions of moves. Therefore, having a fast game engine was an essential foundation. To stress test the engine an expectimax algorithm was used. This algorithm can perform very well on 2048, often reaching the 32,768 tile. To achieve a score so large, more than 18,000 moves and over 90 billion game states need to be considered. If the engine was not extremely well optimised this would take a very long time.

The engine compiles to wasm and therefore can be ran in the browser. See it in action and read more about the project [here](https://2048-ai.mattkennedy.io/).

**Local Usage**
- **Build**: `cargo build`
- **Run (single-threaded)**: `cargo run`
- **Run (parallel)**: `cargo run --bin parallel [-- <flags>]`

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

**Development**
- Tests: `cargo test` (all tests should pass).
- Format/lints: follow standard Rust conventions; no custom toolchain required.
- WASM: A `cdylib` is provided; see `src/wasm.rs` for the Web binding entry points.

**Performance Notes**
- On multi-core systems, the parallel runner saturates cores and typically achieves 2–3× throughput over the single-threaded engine (exact gains depend on depth/board state).
- The status line and periodic score/tile checks are designed to be negligible overhead (checked every 100 moves).
