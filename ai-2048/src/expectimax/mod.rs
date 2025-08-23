//! Expectimax search policy (single-threaded and parallel) for 2048.
//!
//! This module provides two policy implementations:
//! - [`Expectimax`]: single-threaded expectimax.
//! - [`ExpectimaxParallel`]: rayon-based parallel expectimax.
//!
//! Both variants share the same public surface and defaults, matching prior
//! behavior while adding branch evaluations and simple stats.
//!
//! Notes
//! - The engine’s lookup and heuristic tables are initialized lazily; the
//!   constructors call them for you. It’s still fine to call `engine::new()`.
//! - Expectimax is deterministic; randomness only occurs when applying moves
//!   with `Board::make_move` or `engine::make_move`.
//!
//! Quick start
//! ```
//! use ai_2048::engine::{self as GameEngine, Board, Move};
//! use ai_2048::expectimax::{Expectimax, ExpectimaxParallel};
//! use rand::{rngs::StdRng, SeedableRng};
//!
//! // Initialize tables explicitly (optional)
//! GameEngine::new();
//!
//! // Deterministic board setup
//! let mut rng = StdRng::seed_from_u64(123);
//! let b0 = Board::EMPTY
//!     .with_random_tile(&mut rng)
//!     .with_random_tile(&mut rng);
//!
//! // Single-threaded expectimax
//! let mut ex = Expectimax::new();
//! let m = ex.best_move(b0);
//! assert!(m.is_some());
//!
//! // Parallel expectimax
//! let mut ex_par = ExpectimaxParallel::new();
//! let mv = ex_par.get_next_move(b0);
//! assert!(m.is_some() && mv.is_some());
//! ```

use crate::engine;

mod heuristic;
mod search_seq;
mod search_par;

pub use search_seq::Expectimax;
pub use search_par::ExpectimaxParallel;

/// Back-compat alias used by binaries.
pub type ExpectimaxMultithread = ExpectimaxParallel;

/// Configurable knobs for Expectimax. Defaults preserve existing behavior.
///
/// - `prob_cutoff`: prune chance branches when cumulative probability falls below this value.
/// - `depth_cap`: optional hard cap for depth (None keeps dynamic depth uncapped).
/// - `cache_enabled`: enable/disable transposition table usage.
/// - `par_thresholds`: thresholds used only by the parallel implementation.
#[derive(Debug, Clone)]
pub struct ExpectimaxConfig {
    /// Probability cutoff for chance-node pruning.
    pub prob_cutoff: f32,
    /// Optional hard cap on depth (None preserves prior behavior).
    pub depth_cap: Option<u64>,
    /// Enable/disable transposition caching.
    pub cache_enabled: bool,
    /// Thresholds used by the parallel implementation.
    pub par_thresholds: ParThresholds,
}

impl Default for ExpectimaxConfig {
    fn default() -> Self {
        Self {
            prob_cutoff: 1e-4,
            depth_cap: None,
            cache_enabled: true,
            par_thresholds: ParThresholds::default(),
        }
    }
}

/// Thresholds used to balance parallel overheads.
///
/// These mirror the previous constants and preserve behavior by default.
#[derive(Debug, Clone, Copy)]
pub struct ParThresholds {
    pub max_par_depth: u64,
    pub par_depth: u64,
    pub par_slots: usize,
    pub cache_min_depth: u64,
}

impl Default for ParThresholds {
    fn default() -> Self {
        Self { max_par_depth: 4, par_depth: 4, par_slots: 6, cache_min_depth: 3 }
    }
}

/// Per-branch expected value at the root (no normalization).
///
/// - `ev` is the expected value for taking `dir` from the current board.
/// - `legal` is false when the move is a no-op for the current board.
#[derive(Debug, Clone, Copy)]
pub struct BranchEval {
    pub dir: crate::engine::Move,
    pub ev: f64,
    pub legal: bool,
}

/// Basic search stats for a single evaluation.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchStats {
    pub nodes: u64,
    pub peak_nodes: u64,
}

/// Common helper for constructors to ensure tables are initialized.
fn warm_engine_and_heuristics() {
    // Safe to call multiple times.
    engine::new();
    heuristic::warm();
}

/// Bench-only: expose the raw heuristic value for a board.
///
/// Enabled only with the `bench-internal` feature to keep the public API small.
#[cfg(feature = "bench-internal")]
#[inline]
pub fn heuristic_value(board: crate::engine::Board) -> f64 { heuristic::get_heuristic_score(board) }
