//! PyO3 Python bindings for ai-2048
//!
//! This crate provides Python bindings for the ai-2048 Rust library,
//! exposing a clean, ergonomic API for 2048 game simulation and AI.

use pyo3::prelude::*;

mod board;
mod move_enum;
mod expectimax;
mod serialization;
mod dataset;

pub use board::{PyBoard, PyRng};
pub use move_enum::PyMove;
pub use expectimax::{PyExpectimax, PyExpectimaxConfig, PySearchStats};
pub use serialization::{PyRunV2, PyStepV2, PyBranchV2, PyMeta, normalize_branches_py};
pub use dataset::{exps_from_boards, batch_from_steps};

/// Initialize the ai-2048 Python module
#[pymodule]
fn ai_2048(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize the engine's lookup tables
    ai_2048_lib::engine::new();
    
    // Register classes
    m.add_class::<PyBoard>()?;
    m.add_class::<PyMove>()?;
    m.add_class::<PyRng>()?;
    m.add_class::<PyExpectimax>()?;
    m.add_class::<PyExpectimaxConfig>()?;
    m.add_class::<PySearchStats>()?;
    m.add_class::<expectimax::PyBranchEval>()?;
    // Serialization (v2)
    m.add_class::<PyRunV2>()?;
    m.add_class::<PyStepV2>()?;
    m.add_class::<PyBranchV2>()?;
    m.add_class::<PyMeta>()?;
    m.add_function(wrap_pyfunction!(normalize_branches_py, m)?)?;

    // Dataset helpers (NPY + SQLite path)
    m.add_function(wrap_pyfunction!(exps_from_boards, m)?)?;
    m.add_function(wrap_pyfunction!(batch_from_steps, m)?)?;

    Ok(())
}
