from pathlib import Path

import ai_2048 as a2


def repo_root() -> Path:
    # tests/ -> py-ai-2048/ -> repo root
    return Path(__file__).resolve().parents[2]


def test_load_example_run_and_roundtrip(tmp_path: Path):
    example = repo_root() / "examples" / "example.a2run2"
    assert example.exists(), f"missing example run at {example}"

    run = a2.RunV2.load(example)

    # Basic sanity checks
    assert run.meta.steps > 0
    fb = run.final_board
    assert isinstance(fb, a2.Board)
    assert fb.highest_tile() >= 2048  # example file reaches 2048 tile

    # Roundtrip save/load
    out = tmp_path / "roundtrip.a2run2"
    run.save(out)
    run2 = a2.RunV2.load(out)

    assert run2.meta.steps == run.meta.steps
    assert run2.final_board_raw == run.final_board_raw


def test_normalize_branches_integration():
    # Build a simple board and evaluate branches, then normalize
    b = a2.Board.empty().with_random_tile(seed=1).with_random_tile(seed=2)
    ex = a2.Expectimax()
    branches = ex.branch_evals(b)
    norm = a2.normalize_branches_py(branches)

    assert len(norm) == 4
    # At least one should be legal on any non-terminal board
    assert any(br.is_legal for br in norm)

