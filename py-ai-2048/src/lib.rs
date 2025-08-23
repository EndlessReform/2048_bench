//! PyO3 Python bindings for ai-2048
//!
//! This crate provides Python bindings for the ai-2048 Rust library,
//! exposing a clean, ergonomic API for 2048 game simulation and AI.

use pyo3::prelude::*;

mod board;
mod move_enum;

pub use board::{PyBoard, PyRng};
pub use move_enum::PyMove;

/// Initialize the ai-2048 Python module
#[pymodule]
fn ai_2048(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize the engine's lookup tables
    ai_2048_lib::engine::new();
    
    // Register classes
    m.add_class::<PyBoard>()?;
    m.add_class::<PyMove>()?;
    m.add_class::<PyRng>()?;
    
    Ok(())
}