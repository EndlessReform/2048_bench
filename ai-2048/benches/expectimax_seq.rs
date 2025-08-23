use ai_2048::engine::{self as GameEngine, Board, Move};
use ai_2048::expectimax::{Expectimax, ExpectimaxConfig};
use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use std::hint::black_box;

fn warm() { GameEngine::new(); }

fn corpus() -> Vec<Board> {
    let mut rng = StdRng::seed_from_u64(4242);
    let mut boards = Vec::new();
    let mut b = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
    boards.push(b);
    let seq = [Move::Left, Move::Up, Move::Right, Move::Down];
    for i in 0..64 {
        let dir = seq[i % seq.len()];
        let nb = b.shift(dir);
        if nb != b { b = nb.with_random_tile(&mut rng); }
        boards.push(b);
    }
    boards
}

fn bench_seq_branch_and_value(c: &mut Criterion) {
    warm();
    let boards = corpus();
    let cfg = ExpectimaxConfig { depth_cap: Some(5), ..Default::default() };
    let mut ex = Expectimax::with_config(cfg);

    c.bench_function("expectimax_seq/branch_evals", |bch| {
        bch.iter(|| {
            let mut acc = 0.0;
            for &bd in &boards {
                let branches = ex.branch_evals(bd);
                for be in branches { if be.legal { acc += be.ev; } }
            }
            black_box(acc)
        })
    });

    c.bench_function("expectimax_seq/state_value", |bch| {
        bch.iter(|| {
            let mut acc = 0.0;
            for &bd in &boards { acc += ex.state_value(bd); }
            black_box(acc)
        })
    });

    c.bench_function("expectimax_seq/best_move", |bch| {
        bch.iter(|| {
            let mut acc = 0u64;
            for &bd in &boards {
                let m = ex.best_move(bd);
                acc ^= m.map(|mv| mv as u64).unwrap_or(0);
            }
            black_box(acc)
        })
    });
}

fn bench_seq_e2e(c: &mut Criterion) {
    warm();
    let cfg = ExpectimaxConfig { depth_cap: Some(5), ..Default::default() };
    let mut ex = Expectimax::with_config(cfg);
    c.bench_function("e2e_seq/64_moves", |bch| {
        bch.iter(|| {
            let mut rng = StdRng::seed_from_u64(7);
            let mut b = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
            let mut steps = 0;
            while steps < 64 && !b.is_game_over() {
                if let Some(dir) = ex.get_next_move(b) { b = b.make_move(dir, &mut rng); } else { break; }
                steps += 1;
            }
            black_box((b.raw(), steps))
        })
    });
}

criterion_group!(expectimax_seq, bench_seq_branch_and_value, bench_seq_e2e);
criterion_main!(expectimax_seq);

