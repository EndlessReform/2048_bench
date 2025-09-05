from pathlib import Path
import subprocess

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


def test_dataset_load_and_batch(tmp_path: Path):
    # Prepare a few runs and pack into .dat using the datapack CLI
    r1 = build_v2_run(steps=3, base=0x1000)
    r2 = build_v2_run(steps=2, base=0x2000)

    rundir = tmp_path / "runs"
    rundir.mkdir(parents=True, exist_ok=True)
    (rundir / "r1.a2run2").write_bytes(r1.to_bytes())
    (rundir / "r2.a2run2").write_bytes(r2.to_bytes())

    pack = tmp_path / "dataset.dat"
    root = repo_root()
    cmd = [
        "cargo", "run", "-q", "-p", "ai-2048", "--bin", "datapack", "--",
        "build", "--input", str(rundir), "--output", str(pack)
    ]
    subprocess.run(cmd, cwd=root, check=True)

    ds = a2.Dataset(pack)
    assert len(ds) == 5

    pre, dirs = ds.get_batch([0, 1, 4])
    assert len(pre) == 3 and len(dirs) == 3
    assert all(len(p) == 16 for p in pre)
    assert set(dirs) <= {0, 1, 2, 3}


def test_dataset_filter_by_score(tmp_path: Path):
    r1 = build_v2_run(steps=3, base=0x1000)
    r2 = build_v2_run(steps=2, base=0x2000)
    # Override scores to make a split
    m1 = r1.meta
    r1 = a2.RunV2(a2.Meta(m1.steps, m1.start_unix_s, m1.elapsed_s, 500, m1.highest_tile, m1.engine_str), r1.steps, r1.final_board_raw)
    m2 = r2.meta
    r2 = a2.RunV2(a2.Meta(m2.steps, m2.start_unix_s, m2.elapsed_s, 50_000, m2.highest_tile, m2.engine_str), r2.steps, r2.final_board_raw)

    rundir = tmp_path / "runs"
    rundir.mkdir(parents=True, exist_ok=True)
    (rundir / "r1.a2run2").write_bytes(r1.to_bytes())
    (rundir / "r2.a2run2").write_bytes(r2.to_bytes())

    pack = tmp_path / "dataset.dat"
    root = repo_root()
    cmd = [
        "cargo", "run", "-q", "-p", "ai-2048", "--bin", "datapack", "--",
        "build", "--input", str(rundir), "--output", str(pack)
    ]
    subprocess.run(cmd, cwd=root, check=True)

    ds = a2.Dataset(pack)
    hi = ds.filter_by_score(1000, 1_000_000)
    lo = ds.filter_by_score(0, 999)
    assert len(hi) == r2.meta.steps
    assert len(lo) == r1.meta.steps

