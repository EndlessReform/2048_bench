from pathlib import Path
import subprocess

import ai_2048 as a2


def build_v2_run(steps: int, base: int) -> a2.RunV2:
    meta = a2.Meta(steps, 1_700_000_000, 1.0, 1000, 128, None)
    steplist = []
    for i in range(steps):
        pre = base + i
        mv = a2.Move.UP if (i % 4) == 0 else a2.Move.DOWN if (i % 4) == 1 else a2.Move.LEFT if (i % 4) == 2 else a2.Move.RIGHT
        steplist.append(a2.StepV2(pre, mv))
    final_board = base + steps
    return a2.RunV2(meta, steplist, final_board)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_packreader_basic_and_iterators(tmp_path: Path):
    # Build three small v2 runs and pack them
    r1 = build_v2_run(steps=3, base=0x1000)
    r2 = build_v2_run(steps=2, base=0x2000)
    r3 = build_v2_run(steps=4, base=0x3000)

    rundir = tmp_path / "runs"
    rundir.mkdir(parents=True, exist_ok=True)
    (rundir / "r1.a2run2").write_bytes(r1.to_bytes())
    (rundir / "r2.a2run2").write_bytes(r2.to_bytes())
    (rundir / "r3.a2run2").write_bytes(r3.to_bytes())

    pack = tmp_path / "dataset.a2pack"
    # Use the compiled a2pack binary via cargo to build the packfile deterministically
    root = repo_root()
    cmd = [
        "cargo", "run", "-q", "-p", "ai-2048", "--bin", "a2pack", "--",
        "pack", "--input", str(rundir), "--output", str(pack)
    ]
    subprocess.run(cmd, cwd=root, check=True)

    reader = a2.PackReader.open(pack)
    assert len(reader) == 3

    stats = reader.stats
    assert stats.count == 3
    assert stats.min_len == 2
    assert stats.max_len == 4
    assert 2.0 <= stats.mean_len <= 4.0

    run0 = reader.decode(0)
    assert run0.meta.steps == 3

    # Sequential iterator
    seen = 0
    for _ in reader.iter():
        seen += 1
    assert seen == 3

    # Indices iterator
    order = []
    for r in reader.iter_indices([2, 0]):
        order.append(r.meta.steps)
    assert order == [4, 3]

    # Batches iterator (deterministic without shuffle)
    batches = list(reader.iter_batches(batch_size=2))
    assert len(batches) == 2
    assert sum(len(b) for b in batches) == 3
    assert [len(b) for b in batches] == [2, 1]
    assert [batches[0][0].meta.steps, batches[0][1].meta.steps] == [3, 2]

    # Batches shuffled
    shuf1 = [r.meta.steps for batch in reader.iter_batches(batch_size=3, shuffle=True, seed=123) for r in batch]
    shuf2 = [r.meta.steps for batch in reader.iter_batches(batch_size=3, shuffle=True, seed=123) for r in batch]
    assert shuf1 == shuf2  # deterministic by seed
