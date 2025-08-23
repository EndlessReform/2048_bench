use std::sync::OnceLock;

use crate::engine as GameEngine;
use crate::engine::Board;

static HEURISTIC_SCORES: OnceLock<Box<[f64]>> = OnceLock::new();

pub(crate) fn warm() {
    let _ = heuristic_scores();
}

fn heuristic_scores() -> &'static [f64] {
    HEURISTIC_SCORES
        .get_or_init(|| {
            let mut v = vec![0.0f64; 0x1_0000];
            for (i, slot) in v.iter_mut().enumerate() {
                *slot = calc_heuristic_score(i as u64);
            }
            v.into_boxed_slice()
        })
        .as_ref()
}

/// Expected value approximation for a board using precomputed line scores.
#[inline]
pub(crate) fn get_heuristic_score(board: Board) -> f64 {
    let transpose_board = GameEngine::transpose(board.raw());
    (0..4).fold(0., |score, line_idx| {
        let row_val = GameEngine::extract_line(board.raw(), line_idx);
        let col_val = GameEngine::extract_line(transpose_board, line_idx);
        let scores = heuristic_scores();
        let row_score = unsafe { scores.get_unchecked(row_val as usize) };
        let col_score = unsafe { scores.get_unchecked(col_val as usize) };
        score + row_score + col_score
    })
}

// Credit to Nneonneo for heuristic structure
fn calc_heuristic_score(line: u64) -> f64 {
    const LOST_PENALTY: f64 = 200_000.0;
    let tiles = GameEngine::line_to_vec(line);
    LOST_PENALTY + calc_empty(&tiles) + calc_merges(&tiles) - calc_monotonicity(&tiles) - calc_sum(&tiles)
}

fn calc_sum(line: &[u64]) -> f64 {
    const SUM_POWER: f64 = 3.5;
    const SUM_WEIGHT: f64 = 11.0;
    line.iter()
        .fold(0., |acc, &tile_val| acc + (tile_val as f64).powf(SUM_POWER))
        * SUM_WEIGHT
}

fn calc_empty(line: &[u64]) -> f64 {
    const EMPTY_WEIGHT: f64 = 270.0;
    line.iter()
        .fold(0., |num_empty_tiles, &tile_val| if tile_val == 0 { num_empty_tiles + 1. } else { num_empty_tiles })
        * EMPTY_WEIGHT
}

fn calc_merges(line: &Vec<u64>) -> f64 {
    const MERGES_WEIGHT: f64 = 700.0;
    let mut prev = 0;
    let mut counter = 0.;
    let mut merges = 0.;
    for &tile_val in line {
        if prev == tile_val && tile_val != 0 {
            counter += 1.;
        } else if counter > 0. {
            merges += 1. + counter;
            counter = 0.;
        }
        prev = tile_val;
    }
    if counter > 0. {
        merges += 1. + counter;
    }
    merges * MERGES_WEIGHT
}

fn calc_monotonicity(line: &[u64]) -> f64 {
    const MONOTONICITY_POWER: f64 = 4.0;
    const MONOTONICITY_WEIGHT: f64 = 47.0;
    let mut monotonicity_left = 0.;
    let mut monotonicity_right = 0.;
    for i in 1..4 {
        let tile1 = line[i - 1] as f64;
        let tile2 = line[i] as f64;
        if tile1 > tile2 {
            monotonicity_left += tile1.powf(MONOTONICITY_POWER) - tile2.powf(MONOTONICITY_POWER);
        } else {
            monotonicity_right += tile2.powf(MONOTONICITY_POWER) - tile1.powf(MONOTONICITY_POWER);
        }
    }
    monotonicity_left.min(monotonicity_right) * MONOTONICITY_WEIGHT
}

// Credit to Nneonneo
pub(crate) fn count_unique(board: Board) -> i32 {
    let mut bitset = 0u64;
    let mut board_copy = board.raw();
    while board_copy != 0 {
        bitset |= 1 << (board_copy & 0xf);
        board_copy >>= 4;
    }
    bitset >>= 1; // don't count empty tiles
    let mut count = 0;
    while bitset != 0 {
        bitset &= bitset - 1;
        count += 1;
    }
    count
}

