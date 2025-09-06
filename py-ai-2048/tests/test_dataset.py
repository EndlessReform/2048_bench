from pathlib import Path
import subprocess
import sqlite3
import numpy as np

import ai_2048 as a2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_v2_run(steps: int, base: int) -> a2.RunV2:
    meta = a2.Meta(steps, 1_700_000_000, 1.0, 1000, 128, None)
    steplist = []
    for i in range(steps):
        pre = base + i
        mv = a2.Move.UP if (i % 4) == 0 else a2.Move.DOWN if (i % 4) == 1 else a2.Move.LEFT if (i % 4) == 2 else a2.Move.RIGHT
        steplist.append(a2.StepV2(pre, mv))
    final_board = base + steps
    return a2.RunV2(meta, steplist, final_board)


def test_dataset_npy_sqlite_flow(tmp_path: Path):
    # Prepare small runs and build dataset (steps.npy + metadata.db)
    r1 = build_v2_run(steps=3, base=0x1000)
    r2 = build_v2_run(steps=2, base=0x2000)
    rundir = tmp_path / "runs"
    rundir.mkdir(parents=True, exist_ok=True)
    (rundir / "r1.a2run2").write_bytes(r1.to_bytes())
    (rundir / "r2.a2run2").write_bytes(r2.to_bytes())

    out_dir = tmp_path / "ds"
    root = repo_root()
    cmd = [
        "cargo", "run", "-q", "-p", "ai-2048", "--bin", "dataset", "--",
        "build", "--input", str(rundir), "--out", str(out_dir)
    ]
    subprocess.run(cmd, cwd=root, check=True)

    steps = np.load(out_dir / "steps.npy")
    conn = sqlite3.connect(out_dir / "metadata.db")

    # Stats sanity
    assert steps.shape == (5,)
    (run_count,) = next(conn.execute("SELECT COUNT(*) FROM runs"))
    assert run_count == 2

    # Filter runs by id from SQL and gather indices
    rids = [row[0] for row in conn.execute("SELECT id FROM runs ORDER BY id")]
    idxs = np.flatnonzero(np.isin(steps['run_id'], np.array(rids, dtype=steps['run_id'].dtype)))
    assert idxs.size == 5

    # Convert boards to exponents efficiently
    exps_buf, dirs_np, evs_np = a2.batch_from_steps(steps, idxs, parallel=True)
    exps = np.frombuffer(exps_buf, dtype=np.uint8).reshape(-1, 16)
    assert exps.shape == (5, 16)
    assert exps.dtype == np.uint8

    # Direction and EV fields
    assert set(dirs_np.tolist()) <= {0, 1, 2, 3}
    assert evs_np.shape == (5, 4)
