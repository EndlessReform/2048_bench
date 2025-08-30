# PyO3 Python Bindings for ai-2048

High-performance Python bindings for the ai-2048 Rust library: a fast 2048 engine with an Expectimax AI and run serialization (v2, postcard).

## What’s Implemented

- Board: construction, shifting, full moves with random insert, scoring, inspection.
- Move: `UP`, `DOWN`, `LEFT`, `RIGHT` with nice repr/str and hashability.
- Rng: deterministic RNG wrapper around Rust `StdRng` with `clone()`.
- Expectimax (single-thread): `best_move`, `branch_evals`, `state_value`, `last_stats`, `reset_stats`.
- Serialization (v2): `RunV2`, `StepV2`, `Meta`, `BranchV2`, file and bytes I/O, `normalize_branches_py`.
- Packfile reader: `PackReader` for fast, low-overhead access to packed `.a2run2` datasets. Includes random access, batched decode, iterators, JSONL export, and summary stats.
- Python protocols: `__str__`, `__repr__`, `__eq__`, `__hash__`, `__iter__` for `Board`.

## Build & Test

- Build (dev install): `uv run maturin develop`
- Run tests (locked): `uv run --locked pytest`

These honor the Python environment and lockfile via `uv` and build the PyO3 extension with `maturin`.

## Quick Start

```python
from ai_2048 import Board, Move

b = Board.empty().with_random_tile(seed=1).with_random_tile(seed=2)
print(b)
print("score:", b.score(), "highest:", b.highest_tile())

# Pure shift (no random insert)
b2 = b.shift(Move.LEFT)

# Full move (shift + random tile)
b3 = b.make_move(Move.LEFT, seed=3)
```

## Full Game Loops

Deterministic loop using a persistent RNG:

```python
from ai_2048 import Board, Move, Expectimax, Rng

rng = Rng(42)
b = Board.empty().with_random_tile(rng=rng).with_random_tile(rng=rng)
ai = Expectimax()

while not b.is_game_over():
    m = ai.best_move(b)
    if m is None:
        break
    b = b.make_move(m, rng=rng)

print("final score:", b.score(), "highest:", b.highest_tile())
```

Non-deterministic loop (uses thread RNG):

```python
from ai_2048 import Board, Expectimax

b = Board.empty().with_random_tile(seed=11).with_random_tile(seed=12)
ai = Expectimax()

while not b.is_game_over():
    m = ai.best_move(b)
    if m is None:
        break
    b = b.make_move(m)  # thread RNG
```

## Branch Evals and Normalization

Get branch evaluations and normalize them into v2 `BranchV2` values (order: Up, Down, Left, Right):

```python
from ai_2048 import Board, Expectimax, normalize_branches_py

b = Board.empty().with_random_tile(seed=1).with_random_tile(seed=2)
ai = Expectimax()
branches = ai.branch_evals(b)
norm = normalize_branches_py(branches)

for br in norm:
    print(br.is_legal, br.value)
```

## Reading and Writing Runs (v2, postcard)

Load a run from disk, inspect meta, iterate steps, and access the final board:

```python
from ai_2048 import RunV2

run = RunV2.load("examples/example.a2run2")
print("steps:", run.meta.steps)
print("final highest tile:", run.final_board.highest_tile())

# Iterate steps
steps = run.steps
for i, step in enumerate(steps):
    pre = step.pre_board          # Board object
    chosen = step.chosen          # Move
    branches = step.branches      # Optional[List[BranchV2]]
    if i < 3:
        print(i, chosen, pre.score(), branches[0].is_legal if branches else None)
```

Roundtrip via bytes and files:

```python
# Bytes
data = run.to_bytes()
run2 = RunV2.from_bytes(data)

# Files
run.save("/tmp/roundtrip.a2run2")
run3 = RunV2.load("/tmp/roundtrip.a2run2")

## Reading Packed Datasets (PackReader)

The `PackReader` API provides high-throughput access to many `.a2run2` runs stored in a single indexed `.a2pack` file. Build packs with the Rust CLI (`a2pack`) from the repository root (see the top-level README).

Open a packfile and inspect:

```python
import ai_2048 as a2

r = a2.PackReader.open("/path/to/dataset.a2pack")
print("runs:", len(r))
print("stats:", r.stats.count, r.stats.mean_len)

# Random access
a_run = r.decode(0)              # -> RunV2 (v1 upgraded automatically)
print(a_run.meta.steps)

# Batched random access (parallel decode)
batch = r.decode_batch([0, 5, 42])
print([x.meta.steps for x in batch])

# Iteration
y = 0
for run in r.iter():             # sequential; decodes off-thread
    y += run.meta.steps

# Specific-order iteration
for run in r.iter_indices([2, 0, 1]):
    print(run.meta.steps)

# Batches for dataloaders
for batch in r.iter_batches(batch_size=256, shuffle=True, seed=123):
    # batch is a list[RunV2]; deterministic order with seed
    pass

# Bulk JSONL export (fast Rust path)
r.to_jsonl("/tmp/runs.jsonl", parallel=True)
```

Notes
- All heavy decode/export runs in Rust; methods release the GIL for throughput.
- Random access returns `RunV2`; legacy v1 runs are converted transparently.
- `iter_batches(shuffle=True, seed=...)` provides deterministic shuffles for training.
```

### Step-level batches for ML

For training that consumes individual decision steps, use step-level batching. Each batch yields a tuple `(pre_boards, chosen_dirs, branch_evs)`:

```
for (pre_boards, chosen_dirs, branch_evs) in r.iter_step_batches(batch_size=1024, shuffle=True, seed=123):
    # pre_boards: List[List[int]] with 16 exponents (0 empty, 1->2, 2->4, ...), row-major c1r1..c4r4
    # chosen_dirs: List[int] with 0:Up, 1:Down, 2:Left, 3:Right
    # branch_evs: List[List[BranchV2]] ordered [Up, Down, Left, Right]; chosen entry clamped to exactly 1.0 when maximal
    pass
```

Notes:
- The iterator flattens all (run, step) pairs across the pack; `shuffle=True` with a `seed` makes order deterministic.
- When branch EVs are absent on a step (legacy traces), `branch_evs` is synthesized as `[Illegal, Illegal, Illegal, Legal(1.0 at chosen)]`.

## Initialize a Board from a Step

`StepV2` exposes both the raw and high-level board. You can reconstruct the pre-move board and explore hypothetical outcomes:

```python
from ai_2048 import Board, Move, RunV2

run = RunV2.load("examples/example.a2run2")
step0 = run.steps[0]

# Option A: use the ready-made Board
b = step0.pre_board

# Option B: construct from raw
b_raw = Board.from_raw(step0.pre_board_raw)

# Deterministic hypothetical next state from this step
next_b = b.make_move(step0.chosen, seed=123)

# Or just inspect the pure shift without randomness
shift_only = b.shift(step0.chosen)
```

## API Reference (Brief)

- Board: `empty()`, `from_raw(int)`, `raw`, `shift(Move)`, `make_move(Move, rng=None, *, seed=None)`, `with_random_tile(rng=None, *, seed=None)`, `score()`, `is_game_over()`, `highest_tile()`, `count_empty()`, `tile_value(i)`, `to_values()`, `to_exponents()`.
- Move: `UP`, `DOWN`, `LEFT`, `RIGHT`.
- Rng: `Rng(seed)`, `clone()`.
- Expectimax: `best_move(Board)`, `branch_evals(Board)`, `state_value(Board)`, `last_stats()`, `reset_stats()`.
- SearchStats: `nodes`, `peak_nodes`.
- Serialization: `RunV2(meta, steps, final_board_raw)`, `StepV2(pre_board_raw, chosen, branches=None)`, `Meta(...)`, `BranchV2.legal(value)/illegal()`, `RunV2.load(path)`, `RunV2.save(path)`, `RunV2.to_bytes()`, `RunV2.from_bytes(data)`, `normalize_branches_py([...BranchEval...])`.

Notes
- The engine uses a packed `u64` internally; `Board` stays immutable and cheap to copy.
- On an empty board, `highest_tile()` returns `1` (no tiles present).
- For deterministic gameplay, reuse a single `Rng` across steps or pass `seed=`.

## Project Structure

- `ai-2048/`: Core Rust library (engine, expectimax, serialization v2).
- `py-ai-2048/`: PyO3 bindings exposing the Python API.

The Python surface mirrors the Rust API closely for familiarity and performance.
