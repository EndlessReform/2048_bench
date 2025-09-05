# PyO3 Python Bindings for ai-2048

High-performance Python bindings for the ai-2048 Rust library: a fast 2048 engine with an Expectimax AI, v2 run serialization (postcard), and a RAM-friendly dataset pack for shuffled ML training.

## What’s Implemented

- Board: construction, shifting, full moves with random insert, scoring, inspection.
- Move: `UP`, `DOWN`, `LEFT`, `RIGHT` with nice repr/str and hashability.
- Rng: deterministic RNG wrapper around Rust `StdRng` with `clone()`.
- Expectimax (single-thread): `best_move`, `branch_evals`, `state_value`, `last_stats`, `reset_stats`.
- Serialization (v2): `RunV2`, `StepV2`, `Meta`, `BranchV2`, file and bytes I/O, `normalize_branches_py`.
- Dataset pack: `Dataset` for fast, in-RAM, randomly shuffled step reads from a single `.dat` file built from `.a2run2` runs.
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
```

## Dataset Pack (.dat) — RAM-friendly (shuffled reads)

For shuffled training over large step datasets, build a `.dat` file (RAM-friendly) and read via `Dataset`.

Build a `.dat` from `.a2run2` runs (Rust CLI):

```bash
cargo run -q -p ai-2048 --bin datapack -- build --input /path/to/runs --output dataset.dat
```

Load in Python and feed into a PyTorch DataLoader:

```python
import ai_2048 as a2
import torch
from torch.utils.data import Dataset, DataLoader

class StepsDataset(Dataset):
    def __init__(self, dat_path: str):
        self.ds = a2.Dataset(dat_path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Fetch a single item via a 1-element batch
        pre, dirs, brs = self.ds.get_batch([idx])
        # pre is a list of one 16-exponent list; dirs is a list of one int; brs is a list of 4 BranchV2
        exps = pre[0]             # length 16, 0=empty, 1->2, ...
        dir_idx = dirs[0]         # 0:Up, 1:Down, 2:Left, 3:Right
        # Convert BranchV2 list to float EV targets in [0,1]
        evs = [b.value if b.is_legal else 0.0 for b in brs[0]]
        return exps, dir_idx, evs

def collate_steps(batch):
    # Batch is a list of (exps, dir, evs)
    exps = torch.tensor([b[0] for b in batch], dtype=torch.uint8)   # (N, 16)
    dirs = torch.tensor([b[1] for b in batch], dtype=torch.long)    # (N,)
    evs  = torch.tensor([b[2] for b in batch], dtype=torch.float32) # (N, 4)
    return exps, dirs, evs

ds = StepsDataset("dataset.dat")
loader = DataLoader(ds, batch_size=3072, shuffle=True, num_workers=0, collate_fn=collate_steps)

for exps, dirs, evs in loader:
    # exps: (B, 16) uint8 exponents (row-major c1r1..c4r4)
    # dirs: (B,) long indices 0..3
    # evs:  (B, 4) float32 targets in [0,1] (Up, Down, Left, Right)
    # model training step here
    pass

### Reading EVs clearly (binning, argmax, numpy)

If you want to work directly with the `Dataset` without a custom `torch.utils.data.Dataset`, you can grab a small batch and convert to numpy easily:

```python
import numpy as np
import ai_2048 as a2

ds = a2.Dataset("dataset.dat")

# Pull an arbitrary mini-batch by indices
pre, dirs, brs = ds.get_batch([0, 5, 10, 42])

# Boards: convert 16-exponent lists to numpy (B, 16) uint8
exps_np = np.asarray(pre, dtype=np.uint8)

# Branch EVs: brs is a list of 4 BranchV2 per item -> convert to floats in [0,1]
evs = np.array([[b.value if b.is_legal else 0.0 for b in branches] for branches in brs], dtype=np.float32)  # (B, 4)

# Example A: bin by the maximum EV per item into 4 buckets (0–.25, .25–.5, .5–.75, .75–1.0]
max_evs = evs.max(axis=1)
bins = np.digitize(max_evs, bins=[0.25, 0.5, 0.75])  # -> 0..3

# Example B: per-branch binning (e.g., one bucket per branch above 0.8)
high_mask = evs >= 0.8

# Example C: argmax label of best branch (0:Up, 1:Down, 2:Left, 3:Right)
labels = evs.argmax(axis=1)

print(exps_np.shape, evs.shape, labels)
```

Notes:
- `pre` is already the exponent encoding (0 empty, 1->2, ...) in row-major `(c1r1..c4r4)` order; if you prefer values instead of exponents, convert via `np.where(exps_np == 0, 0, 2 ** exps_np)`.
- `brs` elements are BranchV2 objects; use `b.value if b.is_legal else 0.0` to get float targets.

Filtering example (score range view):

```python
base = a2.Dataset("dataset.dat")
mid_scores = base.filter_by_score(50_000, 500_000)
print(len(mid_scores))  # number of steps in view
```

Notes:
- Branch EVs are stored in `.dat` as quantized values with exact 0.0 and 1.0 endpoints; `Dataset.get_batch` returns a third value with 4 BranchV2 labels (Up, Down, Left, Right). The chosen branch is exactly 1.0.
- The dataset is fully in memory; random access is just indexed reads of fixed-size records.
```python
for (pre, dirs, evs) in train.iter_step_batches(batch_size=4096, shuffle=True, seed=0):
    pass
for (pre, dirs, evs) in test.iter_step_batches(batch_size=4096, shuffle=True, seed=0):
    pass

# You can also get run-level batches restricted to a view
for runs in train.iter_batches(batch_size=64, shuffle=True, seed=99):
    pass
```

Parameters:
- unit: "run" (default) avoids correlated leakage; "step" hits step targets by adding whole runs.
- test_pct: float in (0,1); deterministic selection with `seed` when `order="random"`.
- test_size: integer in the chosen unit; mutually exclusive with `test_pct`.
- seed: RNG seed (default 0) for deterministic random ordering.
- order: "random" (default) or "sequential" (respect pack order before splitting).

Behavior and edge cases:
- For `test_pct` with `unit="run"`: uses `floor(pct * num_runs)`, clamped to at least 1 when `pct>0`.
- For `unit="step"`: greedily adds whole runs until the target step count is reached or exceeded.
- Train/test are disjoint and cover the selected universe (runs).

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
