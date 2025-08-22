use dashmap::DashMap;
use rayon::prelude::*;
use ahash::RandomState as AHasher;

use crate::engine::{Board, Move};

use super::{warm_engine_and_heuristics, BranchEval, ExpectimaxConfig, ParThresholds, SearchStats};
use super::heuristic::{count_unique, get_heuristic_score};

#[derive(Clone, Copy)]
struct TranspositionEntry { score: f64, move_depth: u64 }

/// Parallel Expectimax using rayon and a shared `DashMap` transposition table.
///
/// Preserves thresholds and caching behavior from the original implementation.
pub struct ExpectimaxParallel {
    cfg: ExpectimaxConfig,
    stats: SearchStats,
}

impl ExpectimaxParallel {
    pub fn new() -> Self { Self::with_config(ExpectimaxConfig::default()) }

    pub fn with_config(cfg: ExpectimaxConfig) -> Self {
        warm_engine_and_heuristics();
        Self { cfg, stats: SearchStats::default() }
    }

    /// Compute the best move using parallel expectimax.
    ///
    /// This is a convenience wrapper around `branch_evals` that just picks the best move.
    #[inline]
    pub fn best_move(&mut self, board: Board) -> Option<Move> {
        let branches = self.branch_evals(board);
        branches
            .iter()
            .filter(|branch| branch.legal)
            .max_by(|a, b| a.ev.partial_cmp(&b.ev).unwrap_or(std::cmp::Ordering::Equal))
            .map(|branch| branch.dir)
    }

    /// Back-compat shim.
    ///
    /// Equivalent to [`Self::best_move`].
    #[inline]
    pub fn get_next_move(&mut self, board: Board) -> Option<Move> { self.best_move(board) }

    /// Convenience function for parallel runner: get both best move and all branch evaluations.
    ///
    /// This calls `branch_evals` once and extracts the best move from those results.
    #[inline]
    pub fn best_move_with_branches(&mut self, board: Board) -> (Option<Move>, [BranchEval; 4]) {
        let branches = self.branch_evals(board);
        let best_move = branches
            .iter()
            .filter(|branch| branch.legal)
            .max_by(|a, b| a.ev.partial_cmp(&b.ev).unwrap_or(std::cmp::Ordering::Equal))
            .map(|branch| branch.dir);
        (best_move, branches)
    }

    /// Core function: compute EV for each direction (no normalization) in parallel.
    ///
    /// Returns a fixed array in order: `[Up, Down, Left, Right]` and marks
    /// illegal moves as `legal=false`. This is the single source of truth for all evaluations.
    pub fn branch_evals(&mut self, board: Board) -> [BranchEval; 4] {
        let depth = self.compute_depth(board);
        let dirs = [Move::Up, Move::Down, Move::Left, Move::Right];
        let map: DashMap<Board, TranspositionEntry, AHasher> = DashMap::with_hasher(AHasher::new());
        let out_vec: Vec<(usize, BranchEval)> = dirs
            .par_iter()
            .enumerate()
            .map(|(i, &dir)| {
                let new_board = board.shift(dir);
                if new_board == board {
                    (i, BranchEval { dir, ev: 0.0, legal: false })
                } else {
                    let ev = self.expectimax_parallel(new_board, Node::Chance, depth, 1.0, &map);
                    (i, BranchEval { dir, ev, legal: true })
                }
            })
            .collect();
        // Convert to fixed array preserving order
        let mut out: [BranchEval; 4] = [
            BranchEval { dir: Move::Up, ev: 0.0, legal: false },
            BranchEval { dir: Move::Down, ev: 0.0, legal: false },
            BranchEval { dir: Move::Left, ev: 0.0, legal: false },
            BranchEval { dir: Move::Right, ev: 0.0, legal: false },
        ];
        for (i, be) in out_vec { out[i] = be; }
        self.stats.nodes = 0;
        out
    }

    /// EV at root (max node), equivalent to the best branch EV.
    pub fn state_value(&mut self, board: Board) -> f64 {
        let branches = self.branch_evals(board);
        branches
            .iter()
            .filter(|branch| branch.legal)
            .map(|branch| branch.ev)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Statistics collected from the last call to [`best_move`],
    /// [`branch_evals`] or [`state_value`].
    #[inline]
    pub fn last_stats(&self) -> SearchStats { self.stats }

    /// Reset accumulated stats to zero.
    #[inline]
    pub fn reset_stats(&mut self) { self.stats = SearchStats::default(); }

    #[inline]
    fn compute_depth(&self, board: Board) -> u64 {
        let dyn_depth = 3.max(count_unique(board) - 2) as u64;
        match self.cfg.depth_cap { Some(cap) => dyn_depth.min(cap), None => dyn_depth }
    }


    fn expectimax_parallel(
        &self,
        board: Board,
        node: Node,
        move_depth: u64,
        cum_prob: f32,
        map: &DashMap<Board, TranspositionEntry, AHasher>,
    ) -> f64 {
        match node {
            Node::Max => self.evaluate_max_parallel(board, move_depth, cum_prob, map),
            Node::Chance => self.evaluate_chance_parallel(board, move_depth, cum_prob, map),
        }
    }

    fn evaluate_max_parallel(
        &self,
        board: Board,
        move_depth: u64,
        cum_prob: f32,
        map: &DashMap<Board, TranspositionEntry, AHasher>,
    ) -> f64 {
        let directions = [Move::Up, Move::Down, Move::Left, Move::Right];
        let ParThresholds { max_par_depth, .. } = self.cfg.par_thresholds;
        if move_depth >= max_par_depth {
            directions
                .par_iter()
                .map(|&dir| {
                    let new_board = board.shift(dir);
                    if new_board == board { 0.0 } else { self.expectimax_parallel(new_board, Node::Chance, move_depth, cum_prob, map) }
                })
                .reduce(|| 0.0, |a, b| a.max(b))
        } else {
            directions.iter().fold(0.0, |acc, &dir| {
                let new_board = board.shift(dir);
                if new_board == board { acc.max(0.0) } else { acc.max(self.expectimax_parallel(new_board, Node::Chance, move_depth, cum_prob, map)) }
            })
        }
    }

    fn evaluate_chance_parallel(
        &self,
        board: Board,
        move_depth: u64,
        cum_prob: f32,
        map: &DashMap<Board, TranspositionEntry, AHasher>,
    ) -> f64 {
        if move_depth == 0 || cum_prob < self.cfg.prob_cutoff {
            return get_heuristic_score(board);
        }
        if self.cfg.cache_enabled {
            if let Some(entry) = map.get(&board) {
                if entry.move_depth >= move_depth { return entry.score; }
            }
        }
        let num_empty_tiles = board.count_empty() as usize;
        if num_empty_tiles == 0 { return get_heuristic_score(board); }
        let mut slots = Vec::with_capacity(num_empty_tiles);
        let mut tiles_searched = 0;
        let mut tmp = board.raw();
        let mut insert_tile = 1u64;
        while tiles_searched < num_empty_tiles {
            if (tmp & 0xf) == 0 { slots.push(insert_tile); tiles_searched += 1; }
            tmp >>= 4;
            insert_tile <<= 4;
        }
        let base_prob = cum_prob / (num_empty_tiles as f32);
        let ParThresholds { par_depth, par_slots, cache_min_depth, .. } = self.cfg.par_thresholds;
        let sum: f64 = if move_depth >= par_depth && slots.len() >= par_slots {
            slots
                .par_iter()
                .map(|&ins| {
                    let new_board_2 = Board::from_raw(board.raw() | ins);
                    let s2 = self.expectimax_parallel(new_board_2, Node::Max, move_depth - 1, base_prob * 0.9, map) * 0.9;
                    let new_board_4 = Board::from_raw(board.raw() | (ins << 1));
                    let s4 = self.expectimax_parallel(new_board_4, Node::Max, move_depth - 1, base_prob * 0.1, map) * 0.1;
                    s2 + s4
                })
                .sum()
        } else {
            slots.iter().fold(0.0, |acc, &ins| {
                let new_board_2 = Board::from_raw(board.raw() | ins);
                let s2 = self.expectimax_parallel(new_board_2, Node::Max, move_depth - 1, base_prob * 0.9, map) * 0.9;
                let new_board_4 = Board::from_raw(board.raw() | (ins << 1));
                let s4 = self.expectimax_parallel(new_board_4, Node::Max, move_depth - 1, base_prob * 0.1, map) * 0.1;
                acc + s2 + s4
            })
        };
        let score = sum / (num_empty_tiles as f64);
        if self.cfg.cache_enabled && move_depth >= cache_min_depth { map.insert(board, TranspositionEntry { score, move_depth }); }
        score
    }
}

#[derive(Clone, Copy)]
enum Node { Max, Chance }

impl Default for ExpectimaxParallel { fn default() -> Self { Self::new() } }
