# PyO3 Python Bindings for ai-2048

High-performance Python bindings for the ai-2048 Rust library: a fast 2048 engine with an Expectimax AI, v2 run serialization (postcard), and an idiomatic NPY + SQLite data path for training.

## What’s Implemented

- Board: construction, shifting, full moves with random insert, scoring, inspection.
- Move: `UP`, `DOWN`, `LEFT`, `RIGHT` with nice repr/str and hashability.
- Rng: deterministic RNG wrapper around Rust `StdRng` with `clone()`.
- Expectimax (single-thread): `best_move`, `branch_evals`, `state_value`, `last_stats`, `reset_stats`.
- Serialization (v2): `RunV2`, `StepV2`, `Meta`, `BranchV2`, file and bytes I/O, `normalize_branches_py`.
- Dataset helpers (NPY + SQLite): fast batch conversion of packed boards (u64) into 16-exponent arrays (GIL-free, parallel-capable).
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

## Dataset (steps.npy + metadata.db)

Build a dataset from `.a2run2` runs (Rust CLI):

```bash
cargo run -q -p ai-2048 --bin dataset -- build --input /path/to/runs --out dataset_dir
```

Python usage (NumPy + sqlite3):

```python
from pathlib import Path
import sqlite3
import numpy as np
import ai_2048 as a2

dataset = Path("dataset_dir")
steps = np.load(dataset / "steps.npy")  # structured array
conn = sqlite3.connect(dataset / "metadata.db")

# Example filter: runs with score between [50k, 500k]
rids = [row[0] for row in conn.execute(
    "SELECT id FROM runs WHERE max_score BETWEEN ? AND ?", (50_000, 500_000)
)]
mask = np.isin(steps['run_id'], np.array(rids, dtype=steps['run_id'].dtype))
idxs = np.flatnonzero(mask)

# Pull fields and convert boards to exponents efficiently (GIL-free + parallel)
exps_buf, dirs, evs = a2.batch_from_steps(steps, idxs, parallel=True)
exps = np.frombuffer(exps_buf, dtype=np.uint8).reshape(-1, 16)   # (N, 16)
dirs = dirs.astype(np.int64)                                     # (N,)
evs  = evs                                                       # (N, 4) float32
```

Notes
- `batch_from_steps` combines NumPy field slicing with fast GIL-free exponent conversion.
- The `bytearray` can be wrapped zero-copy via `np.frombuffer`.
- Keep `steps` loaded in RAM (`mmap_mode=None`) for fastest random sampling.

### PyTorch DataLoader Example

```python
import torch
from torch.utils.data import Dataset, DataLoader

class StepsDataset(Dataset):
    def __init__(self, dataset_dir: str, run_sql: str | None = None, sql_params: tuple = ()):  # optional SQL filter
        import sqlite3, numpy as np
        self.steps = np.load(f"{dataset_dir}/steps.npy")
        conn = sqlite3.connect(f"{dataset_dir}/metadata.db")
        if run_sql:
            rids = [row[0] for row in conn.execute(run_sql, sql_params)]
            mask = np.isin(self.steps['run_id'], np.array(rids, dtype=self.steps['run_id'].dtype))
            self.indices = np.flatnonzero(mask)
        else:
            self.indices = np.arange(self.steps.shape[0], dtype=np.int64)

    def __len__(self):
        return self.indices.size

    def __getitem__(self, idx):
        return int(self.indices[idx])   # return global index; we slice in the collate fn

def collate_steps(batch_indices, steps):
    import numpy as np, ai_2048 as a2
    idxs = np.array(batch_indices, dtype=np.int64)
    exps_buf, dirs, evs = a2.batch_from_steps(steps, idxs, parallel=True)
    exps = np.frombuffer(exps_buf, dtype=np.uint8).reshape(-1, 16)
    # Convert to tensors
    exps_t = torch.from_numpy(exps.copy())            # uint8 (copy ensures tensor owns memory)
    dirs_t = torch.from_numpy(dirs.astype(np.int64))  # long labels 0..3
    evs_t  = torch.from_numpy(evs.copy())             # float32 targets (N,4)
    return exps_t, dirs_t, evs_t

ds = StepsDataset("dataset_dir", run_sql="SELECT id FROM runs WHERE max_score BETWEEN ? AND ?", sql_params=(50_000, 500_000))
loader = DataLoader(ds, batch_size=768, shuffle=True, num_workers=0,
                    collate_fn=lambda batch: collate_steps(batch, ds.steps))

for exps, dirs, evs in loader:
    # exps: (B, 16) uint8, dirs: (B,) long, evs: (B,4) float32
    pass
```
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
