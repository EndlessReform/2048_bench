#![cfg(feature = "bench-internal")]
use ai_2048::engine::{self as GameEngine, Board, Move};
use ai_2048::expectimax;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use std::hint::black_box;

fn warm() { GameEngine::new(); }

fn corpus() -> Vec<Board> {
    let mut rng = StdRng::seed_from_u64(1337);
    let mut boards = Vec::new();
    boards.push(Board::EMPTY);
    let mut b = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
    boards.push(b);
    let seq = [Move::Left, Move::Up, Move::Right, Move::Down];
    for i in 0..24 {
        let dir = seq[i % seq.len()];
        let nb = b.shift(dir);
        if nb != b { b = nb.with_random_tile(&mut rng); }
        boards.push(b);
    }
    boards
}

fn bench_heuristic(c: &mut Criterion) {
    warm();
    let boards = corpus();
    c.bench_function("heuristic/value", |bch| {
        bch.iter(|| {
            let mut acc = 0f64;
            for &bd in &boards {
                let v = expectimax::heuristic_value(bd);
                acc = acc.mul_add(1.000_000_1, v);
            }
            black_box(acc)
        })
    });
}

criterion_group!(heuristic, bench_heuristic);
criterion_main!(heuristic);

