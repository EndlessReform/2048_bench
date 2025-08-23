use ai_2048::engine::{self as GameEngine, Board, Move, get_score, count_empty, get_highest_tile_val};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use std::hint::black_box;

fn warm() { GameEngine::new(); }

fn corpus() -> Vec<Board> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut boards = Vec::new();
    // Empty and two-tile starts
    boards.push(Board::EMPTY);
    let mut b = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
    boards.push(b);
    // Derive a variety of densities deterministically
    let seq = [Move::Left, Move::Up, Move::Right, Move::Down];
    for i in 0..20 {
        let dir = seq[i % seq.len()];
        let nb = b.shift(dir);
        if nb != b { b = nb.with_random_tile(&mut rng); }
        boards.push(b);
    }
    boards
}

fn bench_shift(c: &mut Criterion) {
    warm();
    c.bench_function("shift/left", |bch| {
        let boards = corpus();
        bch.iter(|| {
            let mut acc = 0u64;
            for &bd in &boards { acc ^= bd.shift(Move::Left).raw(); }
            black_box(acc)
        })
    });
    c.bench_function("shift/right", |bch| {
        let boards = corpus();
        bch.iter(|| {
            let mut acc = 0u64;
            for &bd in &boards { acc ^= bd.shift(Move::Right).raw(); }
            black_box(acc)
        })
    });
    c.bench_function("shift/up", |bch| {
        let boards = corpus();
        bch.iter(|| {
            let mut acc = 0u64;
            for &bd in &boards { acc ^= bd.shift(Move::Up).raw(); }
            black_box(acc)
        })
    });
    c.bench_function("shift/down", |bch| {
        let boards = corpus();
        bch.iter(|| {
            let mut acc = 0u64;
            for &bd in &boards { acc ^= bd.shift(Move::Down).raw(); }
            black_box(acc)
        })
    });
}

fn bench_make_move_and_insert(c: &mut Criterion) {
    warm();
    c.bench_function("board/with_random_tile", |bch| {
        bch.iter_batched(
            || (Board::EMPTY, StdRng::seed_from_u64(7)),
            |(mut bd, mut rng)| {
                for _ in 0..16 { bd = bd.with_random_tile(&mut rng); }
                black_box(bd)
            },
            BatchSize::SmallInput,
        )
    });
    c.bench_function("board/make_move_left", |bch| {
        bch.iter_batched(
            || {
                let mut rng = StdRng::seed_from_u64(9);
                let bd = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
                (bd, rng)
            },
            |(mut bd, mut rng)| {
                for _ in 0..64 { bd = bd.make_move(Move::Left, &mut rng); }
                black_box(bd)
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_score_and_queries(c: &mut Criterion) {
    warm();
    c.bench_function("score/get_score", |bch| {
        let boards = corpus();
        bch.iter(|| {
            let mut acc = 0u64;
            for &bd in &boards { acc = acc.wrapping_add(get_score(bd)); }
            black_box(acc)
        })
    });
    c.bench_function("query/count_empty", |bch| {
        let boards = corpus();
        bch.iter(|| {
            let mut acc = 0u64;
            for &bd in &boards { acc ^= count_empty(bd); }
            black_box(acc)
        })
    });
    c.bench_function("query/highest_tile_val", |bch| {
        let boards = corpus();
        bch.iter(|| {
            let mut acc = 0u64;
            for &bd in &boards { acc ^= get_highest_tile_val(bd); }
            black_box(acc)
        })
    });
}

criterion_group!(engine_ops, bench_shift, bench_make_move_and_insert, bench_score_and_queries);
criterion_main!(engine_ops);
