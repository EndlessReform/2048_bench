//! Criterion benches for batch board-to-exponent conversion.
//!
//! These benches generate a synthetic pool of u64 boards and measure the cost
//! of converting random/sequential batches into 16-element exponent arrays.
//! This mirrors the hot path described in docs/serialization-batch.md without
//! relying on any legacy pack/.dat structures.

use ai_2048::engine::state::Board as RsBoard;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_boards(total: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..total).map(|_| rng.gen::<u64>()).collect()
}

fn bench_board_exps(c: &mut Criterion) {
    // Modest pool by default; adjust locally as needed.
    let total = 200_000; // ~1.6 MB of boards
    let boards = make_boards(total, 123);

    let mut group = c.benchmark_group("board_exps");
    for &batch in &[64usize, 256, 1024] {
        // Pre-generate random index pool deterministically
        let mut rng = StdRng::seed_from_u64(42);
        let pool_len = 10_000usize;
        let pool: Vec<usize> = (0..pool_len).map(|_| rng.gen_range(0..total)).collect();

        group.bench_function(format!("batch_{batch}_random_exps"), |b| {
            b.iter_batched(
                || {
                    let mut idxs = Vec::with_capacity(batch);
                    let mut i = 0usize;
                    while idxs.len() < batch { idxs.push(pool[i % pool_len]); i += 1; }
                    idxs
                },
                |idxs| {
                    let mut out: Vec<[u8; 16]> = Vec::with_capacity(idxs.len());
                    for &gi in &idxs {
                        let exps_vec = RsBoard::from_raw(boards[gi]).to_vec();
                        let exps: [u8; 16] = exps_vec.try_into().unwrap();
                        out.push(exps);
                    }
                    std::hint::black_box(out)
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(format!("batch_{batch}_sequential_exps"), |b| {
            b.iter_batched(
                || {
                    let start = 0usize;
                    (start..start + batch).map(|i| i % total).collect::<Vec<_>>()
                },
                |idxs| {
                    let mut out: Vec<[u8; 16]> = Vec::with_capacity(idxs.len());
                    for &gi in &idxs {
                        let exps_vec = RsBoard::from_raw(boards[gi]).to_vec();
                        let exps: [u8; 16] = exps_vec.try_into().unwrap();
                        out.push(exps);
                    }
                    std::hint::black_box(out)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_board_exps);
criterion_main!(benches);

