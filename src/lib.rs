//! ai-2048: a 2048 game engine + Expectimax policy
//!
//! This crate provides:
//! - A compact `Board` type with ergonomic methods (`shift`, `make_move`, `score`, ...)
//! - An Expectimax AI (`expectimax` module) with single-threaded and parallel variants
//! - A binary trace format for runs (`trace` module)
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
pub mod engine;
pub mod expectimax;
#[cfg(feature = "wasm")]
pub mod wasm;
pub mod trace;
