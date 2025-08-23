//! PyO3 bindings for the Board struct

use pyo3::prelude::*;
use pyo3::types::PyList;
use ai_2048_lib::engine::state::{Board, TilesIter};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::move_enum::PyMove;

/// Optional RNG wrapper for deterministic operations
#[pyclass(name = "Rng")]
pub struct PyRng {
    inner: StdRng,
}


#[pymethods]
impl PyRng {
    #[new]
    fn new(seed: u64) -> Self {
        PyRng {
            inner: StdRng::seed_from_u64(seed),
        }
    }

    /// Create an independent copy of this RNG
    fn clone(&self) -> Self {
        PyRng {
            inner: self.inner.clone(),
        }
    }
}

/// A packed 4x4 2048 board
#[pyclass(name = "Board")]
#[derive(Clone, Copy)]
pub struct PyBoard {
    inner: Board,
}


impl PyBoard {
    pub(crate) fn inner(&self) -> Board {
        self.inner
    }
}

#[pymethods]
impl PyBoard {
    

    /// Create an empty board
    #[staticmethod]
    fn empty() -> Self {
        PyBoard {
            inner: Board::EMPTY,
        }
    }

    /// Create a board from its raw packed representation
    #[staticmethod]
    fn from_raw(raw: u64) -> Self {
        PyBoard {
            inner: Board::from_raw(raw),
        }
    }

    /// Get the raw packed representation
    #[getter]
    fn raw(&self) -> u64 {
        self.inner.raw()
    }

    /// Shift tiles in the given direction (no random tile insertion)
    fn shift(&self, direction: PyMove) -> Self {
        PyBoard {
            inner: self.inner.shift(direction.inner),
        }
    }

    /// Make a move and insert a random tile if the board changed
    #[pyo3(signature = (direction, rng = None, *, seed = None))]
    fn make_move(&self, direction: PyMove, mut rng: Option<&mut PyRng>, seed: Option<u64>) -> PyResult<Self> {
        let new_board = if let Some(ref mut rng) = rng {
            self.inner.make_move(direction.inner, &mut rng.inner)
        } else if let Some(seed) = seed {
            let mut local_rng = StdRng::seed_from_u64(seed);
            self.inner.make_move(direction.inner, &mut local_rng)
        } else {
            let mut thread_rng = rand::thread_rng();
            self.inner.make_move(direction.inner, &mut thread_rng)
        };

        Ok(PyBoard { inner: new_board })
    }

    /// Insert a random tile into an empty cell
    #[pyo3(signature = (rng = None, *, seed = None))]
    fn with_random_tile(&self, mut rng: Option<&mut PyRng>, seed: Option<u64>) -> PyResult<Self> {
        let new_board = if let Some(ref mut rng) = rng {
            self.inner.with_random_tile(&mut rng.inner)
        } else if let Some(seed) = seed {
            let mut local_rng = StdRng::seed_from_u64(seed);
            self.inner.with_random_tile(&mut local_rng)
        } else {
            self.inner.with_random_tile_thread()
        };

        Ok(PyBoard { inner: new_board })
    }

    /// Get the current score
    fn score(&self) -> u64 {
        self.inner.score()
    }

    /// Check if the game is over (no legal moves)
    fn is_game_over(&self) -> bool {
        self.inner.is_game_over()
    }

    /// Get the highest tile value on the board
    fn highest_tile(&self) -> u64 {
        self.inner.highest_tile()
    }

    /// Count empty cells
    fn count_empty(&self) -> u64 {
        self.inner.count_empty()
    }

    /// Get the tile value at a specific index (0-15, row-major)
    fn tile_value(&self, index: usize) -> u16 {
        self.inner.tile_value(index)
    }

    /// Get all tile values as a 4x4 2D list
    fn to_values(&self, py: Python) -> PyResult<Py<PyList>> {
        let mut rows = Vec::with_capacity(4);
        for row in 0..4 {
            let mut row_values = Vec::with_capacity(4);
            for col in 0..4 {
                let index = row * 4 + col;
                row_values.push(self.inner.tile_value(index));
            }
            rows.push(row_values);
        }
        let py_list = PyList::new_bound(py, &rows);
        Ok(py_list.into())
    }

    /// Get all tile exponents as a list (0 for empty, 1 for 2, 2 for 4, etc.)
    fn to_exponents(&self, py: Python) -> PyResult<Py<PyList>> {
        let exponents: Vec<u8> = self.inner.to_vec();
        let py_list = PyList::new_bound(py, &exponents);
        Ok(py_list.into())
    }

    /// String representation showing the board layout
    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    /// Debug representation
    fn __repr__(&self) -> String {
        format!("Board({:#018x})", self.inner.raw())
    }

    /// Iterator over tile values (not exponents)
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyBoardValueIter>> {
        let iter = PyBoardValueIter {
            iter: slf.inner.tiles(),
        };
        Py::new(slf.py(), iter)
    }

    /// Equality comparison
    fn __eq__(&self, other: &PyBoard) -> bool {
        self.inner == other.inner
    }

    /// Hash for use in dictionaries/sets
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

#[pyclass]
struct PyBoardValueIter {
    iter: TilesIter,
}

#[pymethods]
impl PyBoardValueIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<u16> {
        slf.iter.next().map(|val| {
            if val == 0 {
                0
            } else {
                1 << val
            }
        })
    }
}


impl From<Board> for PyBoard {
    fn from(inner: Board) -> Self {
        PyBoard { inner }
    }
}

impl From<PyBoard> for Board {
    fn from(py_board: PyBoard) -> Self {
        py_board.inner
    }
}
