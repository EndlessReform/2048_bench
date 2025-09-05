//! Criterion benches for DataPack random and batched reads.
//!
//! Focus: validate that in-RAM batched random reads are fast and do not incur
//! mmap-style thrashing. We synthesize a DataPack in-memory and benchmark
//! converting pre_board u64s into 16-exponent arrays for batches of varying sizes.

use ai_2048::serialization::{DataPack, RunMeta, Step};
use ai_2048::engine::state::Board as RsBoard;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_synth_datapack(total_runs: usize, steps_per_run: usize, seed: u64) -> DataPack {
    let mut steps: Vec<Step> = Vec::with_capacity(total_runs * steps_per_run);
    let mut runs: Vec<RunMeta> = Vec::with_capacity(total_runs);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut first = 0u32;
    for rid in 0..total_runs as u32 {
        for i in 0..steps_per_run as u32 {
            // Random-ish board values within u64 range; realistic distribution not required for cache behavior
            let board = rng.gen::<u64>();
            let dir = (i % 4) as u8;
            steps.push(Step { board, run_id: rid, index_in_run: i, move_dir: dir, _padding: [0; 15] });
        }
        runs.push(RunMeta {
            id: rid,
            first_step_idx: first,
            num_steps: steps_per_run as u32,
            max_score: rng.gen_range(1000..100_000) as u64,
            highest_tile: 2048,
            engine: String::new(),
            start_time: 1_700_000_000,
            elapsed_s: 1.0,
        });
        first += steps_per_run as u32;
    }
    let mut dp = DataPack { steps, runs, runs_by_score: vec![], runs_by_length: vec![] };
    dp.rebuild_indices();
    dp
}

fn bench_random_read_batches(c: &mut Criterion) {
    // Keep dataset modest for CI; adjust sizes locally if needed.
    let runs = 200;
    let steps_per_run = 1000; // total steps = 200k (~6.4 MB)
    let dp = make_synth_datapack(runs, steps_per_run, 123);
    let total_steps = dp.steps.len();

    let mut group = c.benchmark_group("datapack_random_read");
    for &batch in &[64usize, 256, 1024] {
        // Pre-generate a large pool of random indices to sample from deterministically
        let mut rng = StdRng::seed_from_u64(42);
        let pool_len = 10_000usize;
        let pool: Vec<usize> = (0..pool_len).map(|_| rng.gen_range(0..total_steps)).collect();

        group.bench_function(format!("batch_{batch}_random_exps"), |b| {
            b.iter_batched(
                || {
                    // Build a batch of indices from the pool (random access)
                    let mut idxs = Vec::with_capacity(batch);
                    let mut i = 0usize;
                    while idxs.len() < batch { idxs.push(pool[i % pool_len]); i += 1; }
                    idxs
                },
                |idxs| {
                    // Convert boards to exponent arrays (16 bytes each)
                    let mut out: Vec<[u8; 16]> = Vec::with_capacity(idxs.len());
                    for &gi in &idxs {
                        let exps_vec = RsBoard::from_raw(dp.steps[gi].board).to_vec();
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
                    // Sequential slice of indices to compare against random
                    let start = 0usize;
                    (start..start + batch).map(|i| i % total_steps).collect::<Vec<_>>()
                },
                |idxs| {
                    let mut out: Vec<[u8; 16]> = Vec::with_capacity(idxs.len());
                    for &gi in &idxs {
                        let exps_vec = RsBoard::from_raw(dp.steps[gi].board).to_vec();
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

criterion_group!(benches, bench_random_read_batches);
criterion_main!(benches);
