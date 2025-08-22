use std::collections::HashMap;

use crate::engine::Board;
use crate::engine::Move;

use super::{warm_engine_and_heuristics, BranchEval, ExpectimaxConfig, SearchStats};
use super::heuristic::{count_unique, get_heuristic_score};

enum Node { Max, Chance }

#[derive(Clone, Copy)]
struct TranspositionEntry { score: f64, move_depth: u64 }

/// Single-threaded Expectimax search.
///
/// Constructors warm engine/heuristic tables. Methods preserve the
/// existing decision logic and performance characteristics.
pub struct Expectimax {
    cfg: ExpectimaxConfig,
    stats: SearchStats,
}

impl Expectimax {
    pub fn new() -> Self { Self::with_config(ExpectimaxConfig::default()) }

    pub fn with_config(cfg: ExpectimaxConfig) -> Self {
        warm_engine_and_heuristics();
        Self { cfg, stats: SearchStats::default() }
    }

    /// Back-compat shim.
    ///
    /// Equivalent to [`Self::best_move`].
    #[inline]
    pub fn get_next_move(&mut self, board: Board) -> Option<Move> { self.best_move(board) }

    /// Compute the best move using expectimax.
    ///
    /// Example
    /// ```
    /// use ai_2048::engine::{self as GameEngine, Board};
    /// use ai_2048::expectimax::Expectimax;
    /// use rand::{SeedableRng, rngs::StdRng};
    /// GameEngine::new();
    /// let mut rng = StdRng::seed_from_u64(7);
    /// let b = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
    /// let mut ex = Expectimax::new();
    /// assert!(ex.best_move(b).is_some());
    /// ```
    #[inline]
    pub fn best_move(&mut self, board: Board) -> Option<Move> {
        let mut map: HashMap<Board, TranspositionEntry> = HashMap::new();
        let mut state_count = 0u64;
        let depth = self.compute_depth(board);
        let result = self.expectimax(board, Node::Max, depth, 1.0, &mut map, &mut state_count);
        self.stats.nodes = state_count;
        self.stats.peak_nodes = self.stats.peak_nodes.max(state_count);
        result.move_dir
    }

    /// Compute EV for each direction (no normalization).
    ///
    /// Returns a fixed array in order: `[Up, Down, Left, Right]` and marks
    /// illegal moves as `legal=false`.
    ///
    /// Example
    /// ```
    /// use ai_2048::engine::{self as GameEngine, Board};
    /// use ai_2048::expectimax::{Expectimax, BranchEval};
    /// use rand::{SeedableRng, rngs::StdRng};
    /// GameEngine::new();
    /// let mut rng = StdRng::seed_from_u64(9);
    /// let b = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
    /// let mut ex = Expectimax::new();
    /// let branches = ex.branch_evals(b);
    /// assert_eq!(branches.len(), 4);
    /// ```
    pub fn branch_evals(&mut self, board: Board) -> [BranchEval; 4] {
        let depth = self.compute_depth(board);
        let mut map: HashMap<Board, TranspositionEntry> = HashMap::new();
        let mut state_count = 0u64;
        let dirs = [Move::Up, Move::Down, Move::Left, Move::Right];
        let mut out: [BranchEval; 4] = [
            BranchEval { dir: Move::Up, ev: 0.0, legal: false },
            BranchEval { dir: Move::Down, ev: 0.0, legal: false },
            BranchEval { dir: Move::Left, ev: 0.0, legal: false },
            BranchEval { dir: Move::Right, ev: 0.0, legal: false },
        ];
        for (i, &dir) in dirs.iter().enumerate() {
            let new_board = board.shift(dir);
            if new_board != board {
                let ev = self.expectimax(new_board, Node::Chance, depth, 1.0, &mut map, &mut state_count).score;
                out[i] = BranchEval { dir, ev, legal: true };
            } else {
                out[i] = BranchEval { dir, ev: 0.0, legal: false };
            }
        }
        self.stats.nodes = state_count;
        self.stats.peak_nodes = self.stats.peak_nodes.max(state_count);
        out
    }

    /// EV at root (max node), equivalent to the best branch EV.
    ///
    /// This does not apply normalization or stochasticity; it reflects the
    /// raw expected value used internally by the policy.
    pub fn state_value(&mut self, board: Board) -> f64 {
        let mut map: HashMap<Board, TranspositionEntry> = HashMap::new();
        let mut state_count = 0u64;
        let depth = self.compute_depth(board);
        let res = self.expectimax(board, Node::Max, depth, 1.0, &mut map, &mut state_count);
        self.stats.nodes = state_count;
        self.stats.peak_nodes = self.stats.peak_nodes.max(state_count);
        res.score
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

    fn expectimax(
        &self,
        board: Board,
        node: Node,
        move_depth: u64,
        cum_prob: f32,
        map: &mut HashMap<Board, TranspositionEntry>,
        state_count: &mut u64,
    ) -> ExpectimaxResult {
        *state_count += 1;
        match node {
            Node::Max => self.evaluate_max(board, move_depth, cum_prob, map, state_count),
            Node::Chance => self.evaluate_chance(board, move_depth, cum_prob, map, state_count),
        }
    }

    fn evaluate_max(
        &self,
        board: Board,
        move_depth: u64,
        cum_prob: f32,
        map: &mut HashMap<Board, TranspositionEntry>,
        state_count: &mut u64,
    ) -> ExpectimaxResult {
        let mut best_score = 0.0;
        let mut best_move = None;
        for &direction in &[Move::Up, Move::Down, Move::Left, Move::Right] {
            let new_board = board.shift(direction);
            if new_board != board {
                let score = self.expectimax(new_board, Node::Chance, move_depth, cum_prob, map, state_count).score;
                if score > best_score {
                    best_score = score;
                    best_move = Some(direction);
                }
            }
        }
        ExpectimaxResult { score: best_score, move_dir: best_move }
    }

    fn evaluate_chance(
        &self,
        board: Board,
        move_depth: u64,
        cum_prob: f32,
        map: &mut HashMap<Board, TranspositionEntry>,
        state_count: &mut u64,
    ) -> ExpectimaxResult {
        if move_depth == 0 || cum_prob < self.cfg.prob_cutoff {
            return ExpectimaxResult { score: get_heuristic_score(board), move_dir: None };
        }
        // Cache lookup (respect depth)
        if self.cfg.cache_enabled {
            if let Some(entry) = map.get(&board) {
                if entry.move_depth >= move_depth {
                    return ExpectimaxResult { score: entry.score, move_dir: None };
                }
            }
        }
        let num_empty_tiles = board.count_empty();
        let mut tiles_searched = 0;
        let mut tmp: u64 = board.raw();
        let mut insert_tile: u64 = 1;
        let mut score = 0.0;
        let base_prob = cum_prob / num_empty_tiles as f32;
        while tiles_searched < num_empty_tiles {
            if (tmp & 0xf) == 0 {
                let new_board2 = Board::from_raw(board.raw() | insert_tile);
                score += self
                    .expectimax(new_board2, Node::Max, move_depth - 1, base_prob * 0.9, map, state_count)
                    .score
                    * 0.9;
                let new_board4 = Board::from_raw(board.raw() | (insert_tile << 1));
                score += self
                    .expectimax(new_board4, Node::Max, move_depth - 1, base_prob * 0.1, map, state_count)
                    .score
                    * 0.1;
                tiles_searched += 1;
            }
            tmp >>= 4;
            insert_tile <<= 4;
        }
        score /= num_empty_tiles as f64;
        if self.cfg.cache_enabled {
            map.insert(board, TranspositionEntry { score, move_depth });
        }
        ExpectimaxResult { score, move_dir: None }
    }
}

#[derive(Debug, Clone, Copy)]
struct ExpectimaxResult { score: f64, move_dir: Option<Move> }

impl Default for Expectimax { fn default() -> Self { Self::new() } }
