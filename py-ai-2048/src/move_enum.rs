//! PyO3 bindings for the Move enum

use pyo3::prelude::*;
use ai_2048_lib::engine::Move;

/// A direction to move/merge tiles in the 2048 game.
#[pyclass(name = "Move")]
#[derive(Clone, Copy)]
pub struct PyMove {
    pub(crate) inner: Move,
}

#[pymethods]
impl PyMove {
    /// Move tiles up
    #[classattr]
    const UP: PyMove = PyMove { inner: Move::Up };
    
    /// Move tiles down
    #[classattr]
    const DOWN: PyMove = PyMove { inner: Move::Down };
    
    /// Move tiles left
    #[classattr]
    const LEFT: PyMove = PyMove { inner: Move::Left };
    
    /// Move tiles right
    #[classattr]
    const RIGHT: PyMove = PyMove { inner: Move::Right };
    
    fn __repr__(&self) -> String {
        match self.inner {
            Move::Up => "Move.UP".to_string(),
            Move::Down => "Move.DOWN".to_string(),
            Move::Left => "Move.LEFT".to_string(),
            Move::Right => "Move.RIGHT".to_string(),
        }
    }
    
    fn __str__(&self) -> String {
        match self.inner {
            Move::Up => "UP".to_string(),
            Move::Down => "DOWN".to_string(),
            Move::Left => "LEFT".to_string(),
            Move::Right => "RIGHT".to_string(),
        }
    }
    
    fn __eq__(&self, other: &PyMove) -> bool {
        self.inner == other.inner
    }
    
    fn __hash__(&self) -> u64 {
        match self.inner {
            Move::Up => 0,
            Move::Down => 1,
            Move::Left => 2,
            Move::Right => 3,
        }
    }
}

impl From<Move> for PyMove {
    fn from(inner: Move) -> Self {
        PyMove { inner }
    }
}

impl From<PyMove> for Move {
    fn from(py_move: PyMove) -> Self {
        py_move.inner
    }
}