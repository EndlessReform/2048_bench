# PyO3 Public API Proposal for ai-2048

Author: GPT-4o (design draft)
Date: 2025-08-23
Target: Python bindings for the ai-2048 Rust crate

## Goals

- Small, stable Python surface that mirrors the crate’s focused Rust API.
- Ergonomic game-loop operations: move selection, applying moves, board value.
- Straightforward run capture (per-step) and v2 serialization interop.
- Determinism controls via seeded RNG without leaking unnecessary internals.
- Preserve performance: release GIL, avoid copies, keep hot paths lean.

## Summary Of Current Rust Abstractions (as of this draft)

- `engine`
  - `Board(u64)`: packed 4x4 state. Methods: `shift`, `with_random_tile(_|_thread)`, `make_move(Rng)`, `score`, `is_game_over`, `highest_tile`, `count_empty`, `tile_value`, `tiles()/to_vec()`, `from_raw`/`into_raw`.
  - `Move`: `Up|Down|Left|Right`.
  - Free fns mirroring methods (e.g., `shift`, `make_move`, `insert_random_tile`) for thread-RNG convenience.
  - One-shot `engine::new()` initializes lookup tables (idempotent).
- `expectimax`
  - `Expectimax` (seq) and `ExpectimaxParallel` (rayon). Config: `ExpectimaxConfig` + `ParThresholds`.
  - Core ops: `best_move`, `branch_evals -> [BranchEval;4]`, `state_value`, stats getters/reset.
  - Deterministic; no RNG inside the search. Uses engine shift/heuristics.
- `serialization`
  - v2 types: `RunV2 { meta, steps, final_board }`, `StepV2 { pre_board: u64, chosen: Move, branches: Option<[BranchV2;4]> }`, `BranchV2::{Legal(f32), Illegal}`.
  - Helpers: `normalize_branches`, `to/from_postcard_bytes`, `read/write_postcard_*`, `from_v1` (legacy trace -> v2).
- `trace` (v1 binary format) retained for migration.

These surfaces are compact and map cleanly to Python concepts without exposing internal tables or perf tricks.

## Proposed Python Module Layout

- Top-level module name: `ai_2048`
  - Classes: `Board`, `Move`, `Expectimax`, `ExpectimaxConfig`, `SearchStats`, `BranchEval`.
  - Parallel variant: either `Expectimax.parallel(...)` constructor or a distinct `ExpectimaxParallel` class mirroring methods (keeps Rust parity).
  - Serialization: `RunV2`, `StepV2`, `BranchV2`, `Meta` plus helpers for file/bytes.
  - Optional helper: `RunBuilder` for ergonomic recording during loops (pure-Python-facing, implemented in Rust for speed and zero-copy board storage).

## Public Python API (recommendation)

### Board

- Construction
  - `Board.empty() -> Board`
  - `Board.from_raw(raw: int) -> Board`
  - `Board.raw -> int` (property)

- Game mechanics
  - `Board.shift(direction: Move) -> Board`
  - `Board.make_move(direction: Move, rng: Optional[Rng] = None, *, seed: Optional[int] = None) -> Board`
    - Uses `rng` if provided; else creates a local `StdRng` from `seed`; else falls back to thread RNG.
  - `Board.with_random_tile(rng: Optional[Rng] = None, *, seed: Optional[int] = None) -> Board`

- Inspection
  - `Board.score() -> int`
  - `Board.is_game_over() -> bool`
  - `Board.highest_tile() -> int`
  - `Board.count_empty() -> int`
  - `Board.tile_value(index: int) -> int`  // row-major 0..15, 0 for empty
  - `Board.to_values() -> List[int]`       // 16 actual values [0,2,4,...]
  - `Board.to_exponents() -> List[int]`    // 16 nibbles [0,1,2,...]

- Python protocols
  - `__str__` -> grid rendering (delegates to Rust `Display`)
  - `__repr__` -> `Board(0x...)`
  - `__iter__` -> values (not exponents) for Python ergonomics
  - `__eq__`, `__hash__`

Note: keep underlying representation packed; no mutation-in-place to preserve Rust semantics and cheap copying.

### RNG (determinism helper)

- `class Rng:` wraps `StdRng` for repeatable sequences across steps.
  - `Rng(seed: int)`
  - `clone() -> Rng` (independent sequence copy)
  - Hidden internals; only usable where `rng:` is accepted.

This avoids “seed-per-call” pitfalls when users need deterministic full games.

### Expectimax (single-threaded and parallel)

- `class ExpectimaxConfig:`
  - `prob_cutoff: float = 1e-4`
  - `depth_cap: Optional[int] = None`
  - `cache_enabled: bool = True`
  - `parallel_thresholds: Optional[ParallelThresholds] = None`  // only used by parallel

- `class ParallelThresholds:`
  - `max_par_depth: int = 4`
  - `par_depth: int = 4`
  - `par_slots: int = 6`
  - `cache_min_depth: int = 3`

- `class BranchEval:`
  - `direction: Move`
  - `expected_value: float`
  - `is_legal: bool`

- `class SearchStats:`
  - `nodes_visited: int`
  - `peak_nodes: int`

- `class Expectimax:`
  - `__init__(config: Optional[ExpectimaxConfig] = None)`
  - `best_move(board: Board) -> Optional[Move]`        // releases GIL
  - `branch_evals(board: Board) -> List[BranchEval]`   // releases GIL; fixed order [Up, Down, Left, Right]
  - `state_value(board: Board) -> float`               // releases GIL
  - `last_stats() -> SearchStats`
  - `reset_stats() -> None`
  - `@staticmethod parallel(config: Optional[ExpectimaxConfig] = None) -> Expectimax`  // wraps parallel instance

Alternatively (closer to Rust), expose a distinct `ExpectimaxParallel` with identical methods and config. Both designs are fine; pick one based on Python package ergonomics. The separate class keeps parity with Rust and simplifies documentation.

### Serialization (v2)

- `class Meta:`
  - `steps: int`
  - `start_unix_s: int`
  - `elapsed_s: float`
  - `max_score: int`
  - `highest_tile: int`
  - `engine_str: Optional[str]`

- `class BranchV2(Enum):`
  - `LEGAL(value: float)`  // normalized [0,1), 1.0-eps as upper bound
  - `ILLEGAL`

- `class StepV2:`
  - Properties prefer high-level types:
    - `pre_board: Board` (exposed as a property backed by raw u64)
    - `chosen: Move`
    - `branches: Optional[List[BranchV2]]` (length 4; order [Up, Down, Left, Right])
  - Raw accessors when needed:
    - `pre_board_raw: int`

- `class RunV2:`
  - `__init__(meta: Meta, steps: List[StepV2], final_board: Board)`
  - Serialization
    - `to_bytes() -> bytes`
    - `@staticmethod from_bytes(data: bytes) -> RunV2`
  - Files
    - `save(path: str) -> None`
    - `@staticmethod load(path: str) -> RunV2`
  - v1 migration
    - `@staticmethod from_v1_file(path: str) -> RunV2`
  - Convenience
    - `final_board_raw: int`
    - `iter_pre_boards() -> Iterator[Board]`  // easy replay

- Module helpers
  - `normalize_branches(branches: List[BranchEval]) -> List[BranchV2]`  // mirrors Rust normalization

### RunBuilder (ergonomic recording; optional but recommended)

- `class RunBuilder:`
  - `__init__(engine_str: Optional[str] = None)`  // captures `start_unix_s` at construction
  - `record_step(pre_board: Board, chosen: Move, branches: Optional[List[BranchEval]] = None) -> None`
    - Internally normalizes branches to `BranchV2` if provided.
  - `finish(final_board: Board, max_score: Optional[int] = None, highest_tile: Optional[int] = None, elapsed_s: Optional[float] = None) -> RunV2`
    - If omitted, `elapsed_s` computed from start time; `max_score`/`highest_tile` pulled from `final_board`.

This yields a concise, safe pattern in loops without manual meta bookkeeping.

### Move

- `class Move(Enum): UP, DOWN, LEFT, RIGHT`
- Provide module constants for convenience if desired: `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT`.

## Usage Sketches

### Minimal game loop

```python
from ai_2048 import Board, Move, Expectimax

b = Board.empty().with_random_tile(seed=1).with_random_tile(seed=2)
ai = Expectimax()

while not b.is_game_over():
    m = ai.best_move(b)
    if m is None:
        break
    b = b.make_move(m)  # thread RNG is fine for quick sims; pass rng/seed for determinism

print(b.score(), b.highest_tile())
```

### Deterministic replay with RNG object

```python
from ai_2048 import Board, Move, Expectimax, Rng

rng = Rng(seed=42)
b = Board.empty().with_random_tile(rng=rng).with_random_tile(rng=rng)
ai = Expectimax()

steps = 0
while not b.is_game_over() and steps < 8:
    m = ai.best_move(b)
    if m is None:
        break
    b = b.make_move(m, rng=rng)
    steps += 1
```

### Recording a run

```python
from ai_2048 import Board, Move, Expectimax, RunBuilder, normalize_branches

b = Board.empty().with_random_tile(seed=11).with_random_tile(seed=12)
ai = Expectimax()
rec = RunBuilder(engine_str="expectimax-python")

while not b.is_game_over():
    branches = ai.branch_evals(b)
    move = ai.best_move(b)
    if move is None:
        break
    rec.record_step(b, move, branches)
    b = b.make_move(move)

run = rec.finish(b)
run.save("game.a2run2")

# Later
loaded = type(run).load("game.a2run2")
for i, pb in enumerate(loaded.iter_pre_boards()):
    pass  # analyze boards
```

## Mapping To Rust (PyO3 binding notes)

- Init
  - Call `engine::new()` in module init; constructors for Expectimax call warmers already.

- Board
  - Expose as `#[pyclass]` wrapping `Board` (Copy); methods call through 1:1.
  - `__iter__` returns values (convert exponents via existing `get_tile_val`).
  - Seeds: map to local `StdRng::seed_from_u64`; reuse on `Rng` paths via an owned `StdRng` kept inside `#[pyclass] Rng`.

- Expectimax
  - `best_move`, `branch_evals`, `state_value` run under `Python::allow_threads` to release the GIL.
  - Parallel constructor either as separate `ExpectimaxParallel` or `Expectimax.parallel()` that holds the parallel struct internally.
  - `SearchStats`/`BranchEval` `#[pyclass]` or `#[pyclass(get)]` with plain fields.

- Serialization
  - `RunV2`, `StepV2`, `Meta`, `BranchV2` as `#[pyclass]` with `#[pyo3(get, set)]` or read-only properties as appropriate.
  - Methods call `serialization::v2` functions directly. For `pre_board`, store raw `u64` and present as `Board` on access.

- RunBuilder
  - Maintain `Vec<StepV2>` and start time in Rust; normalize branches using existing `normalize_branches`.

## API Stability & Docs

- Keep Python surface minimal and aligned with existing Rust methods. Prefer additive changes.
- Document determinism: prefer passing `Rng` across steps; otherwise use `seed` parameters.
- Small, runnable examples in docstrings. Use `StdRng::seed_from_u64(_)` in doctests on Rust side and `seed=`/`Rng` in Python examples.

## Feature Flags & Packaging

- Gate PyO3 under a `python` (or `pyo3`) feature in the Rust crate; build `cdylib` when enabled.
- WASM remains gated behind `wasm` (unchanged).
- No core logic in Python-only paths; bindings forward to the library.

## Risks & Mitigations

- Long-running searches: always release GIL to avoid blocking Python threads.
- Determinism footguns: provide `Rng` object and `seed` keyword to avoid accidental nondeterminism.
- Data copying: keep boards packed (`u64`); expose `Board` by value; avoid copying large vectors.

## Next Steps

1) Confirm parallel exposure shape (separate class vs static constructor).
2) Prototype `Board`/`Move`/`Expectimax` bindings with doctested examples.
3) Add `RunV2`/`StepV2`/`RunBuilder` to support analysis workflows.
4) Measure overhead; verify <10% vs Rust for single-thread best_move on representative boards.

