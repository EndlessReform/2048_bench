from pathlib import Path
import subprocess

import ai_2048 as a2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_v2_run_with_branches(steps: int, base: int) -> a2.RunV2:
    meta = a2.Meta(steps, 1_700_000_000, 1.0, 1000, 128, None)
    steplist = []
    # chosen is always UP (0) to make EV checks easy
    for i in range(steps):
        pre = base + i
        mv = a2.Move.UP
        # Branch EVs: chosen (UP) is the maximum, others lower; ensure normalization yields near-1
        branches = [
            a2.BranchV2.legal(1.0),  # UP
            a2.BranchV2.legal(0.5),  # DOWN
            a2.BranchV2.legal(0.25), # LEFT
            a2.BranchV2.illegal(),   # RIGHT
        ]
        steplist.append(a2.StepV2(pre, mv, branches))
    final_board = base + steps
    return a2.RunV2(meta, steplist, final_board)


def test_iter_step_batches_shapes_and_clamp(tmp_path: Path):
    # Build two small runs with branches so chosen EV normalizes to max
    r1 = build_v2_run_with_branches(steps=3, base=0x1000_0000_0000_0000)
    r2 = build_v2_run_with_branches(steps=2, base=0x2000_0000_0000_0000)

    rundir = tmp_path / "runs"
    rundir.mkdir(parents=True, exist_ok=True)
    (rundir / "r1.a2run2").write_bytes(r1.to_bytes())
    (rundir / "r2.a2run2").write_bytes(r2.to_bytes())

    pack = tmp_path / "dataset.a2pack"
    # Build pack using the binary for determinism
    root = repo_root()
    cmd = [
        "cargo", "run", "-q", "-p", "ai-2048", "--bin", "a2pack", "--",
        "pack", "--input", str(rundir), "--output", str(pack)
    ]
    subprocess.run(cmd, cwd=root, check=True)

    reader = a2.PackReader.open(pack)
    total_steps = sum(reader.decode(i).meta.steps for i in range(len(reader)))
    assert total_steps == 5

    # Step-level batches without shuffle
    batches = list(reader.iter_step_batches(batch_size=3))
    # Should yield 2 batches: 3 then 2
    assert len(batches) == 2
    pre0, dirs0, br0 = batches[0]
    pre1, dirs1, br1 = batches[1]
    assert len(pre0) == 3 and len(dirs0) == 3 and len(br0) == 3
    assert len(pre1) == 2 and len(dirs1) == 2 and len(br1) == 2

    # Check pre_board shape/content (16 exponents) and direction encoding
    assert all(len(p) == 16 for p in pre0 + pre1)
    assert set(dirs0 + dirs1) == {0}  # always UP in our synthetic data

    # Branch EVs structure and clamping: UP is chosen, should be Legal(1.0)
    for branches in br0 + br1:
        assert len(branches) == 4
        assert branches[0].is_legal and abs(branches[0].value - 1.0) < 1e-12
        assert branches[1].is_legal and branches[1].value < 1.0
        assert branches[2].is_legal and branches[2].value < 1.0
        assert not branches[3].is_legal

    # Deterministic shuffle by seed
    # Deterministic order check: compare the chosen-dir sequence across runs
    flat_dirs_a = [d for (_, dirs, _) in reader.iter_step_batches(batch_size=10, shuffle=True, seed=42) for d in dirs]
    flat_dirs_b = [d for (_, dirs, _) in reader.iter_step_batches(batch_size=10, shuffle=True, seed=42) for d in dirs]
    assert flat_dirs_a == flat_dirs_b
