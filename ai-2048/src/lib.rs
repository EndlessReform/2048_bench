//! ai-2048: a 2048 game engine + Expectimax policy
//!
//! This crate provides:
//! - A compact `Board` type with ergonomic methods (`shift`, `make_move`, `score`, ...)
//! - An Expectimax AI (`expectimax` module) with single-threaded and parallel variants
//! - A binary trace format for runs (`trace` module)
//! - Dataset pack utilities for RAM-friendly training data (`serialization::DataPack`, `serialization::PackBuilder`)
//!
//! Quick start:
//! ```
//! use ai_2048::engine::{self as GameEngine, Board, Move};
//! use rand::{rngs::StdRng, SeedableRng};
//!
//! // One-time table init
//! GameEngine::new();
//!
//! // Deterministic board initialization with a seeded RNG
//! let mut rng = StdRng::seed_from_u64(42);
//! let b0 = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
//! let b1 = b0.shift(Move::Left);
//! assert!(b1.score() >= 0);
//! ```
//!
//! Note: For convenience, there are also free functions mirroring the `Board` methods
//! (e.g., `engine::shift`, `engine::make_move`) that use thread-local RNG where relevant.
//! Prefer the methods when you need determinism.
//!
//! Expectimax quick start
//! ```
//! use ai_2048::engine::{self as GameEngine, Board};
//! use ai_2048::expectimax::{Expectimax, ExpectimaxParallel};
//! use rand::{rngs::StdRng, SeedableRng};
//! GameEngine::new();
//! let mut rng = StdRng::seed_from_u64(42);
//! let b = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
//! let mut ex = Expectimax::new();
//! let m = ex.best_move(b);
//! let mut ex_par = ExpectimaxParallel::new();
//! assert!(m.is_some() && ex_par.get_next_move(b).is_some());
//! ```
//!
//! Full loop (simplest possible)
//! ```
//! use ai_2048::engine::{self as GameEngine, Board};
//! use ai_2048::expectimax::Expectimax;
//! use rand::{rngs::StdRng, SeedableRng};
//!
//! // 1) Initialize tables and policy
//! GameEngine::new();
//! let mut policy = Expectimax::new();
//! let mut rng = StdRng::seed_from_u64(123);
//!
//! // 2) Start board with two random tiles (deterministic via seeded RNG)
//! let mut b = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
//! let mut moves = 0u32;
//!
//! // 3) Loop a couple of moves to demonstrate flow (keep doctests fast)
//! while !b.is_game_over() && moves < 4 {
//!     if let Some(dir) = policy.get_next_move(b) {
//!         b = b.make_move(dir, &mut rng);
//!         moves += 1;
//!     } else {
//!         break;
//!     }
//! }
//!
//! // 4) Inspect final state (score, highest tile, etc.)
//! let _final_score = b.score();
//! assert!(moves > 0);
//! ```
//!
pub mod engine;
pub mod expectimax;
pub mod trace;
pub mod serialization;
