//! PyO3 bindings for the Expectimax AI

use pyo3::prelude::*;
use ai_2048_lib::expectimax::{Expectimax, ExpectimaxConfig, SearchStats, BranchEval as RsBranchEval};
use crate::board::PyBoard;
use crate::move_enum::PyMove;

#[pyclass(name = "ExpectimaxConfig")]
#[derive(Clone)]
pub struct PyExpectimaxConfig {
    inner: ExpectimaxConfig,
}

#[pymethods]
impl PyExpectimaxConfig {
    #[new]
    fn new() -> Self {
        PyExpectimaxConfig {
            inner: ExpectimaxConfig::default(),
        }
    }

    #[getter]
    fn prob_cutoff(&self) -> f32 { self.inner.prob_cutoff }

    #[setter]
    fn set_prob_cutoff(&mut self, v: f32) { self.inner.prob_cutoff = v; }

    #[getter]
    fn depth_cap(&self) -> Option<u64> { self.inner.depth_cap }

    #[setter]
    fn set_depth_cap(&mut self, v: Option<u64>) { self.inner.depth_cap = v; }

    #[getter]
    fn cache_enabled(&self) -> bool { self.inner.cache_enabled }

    #[setter]
    fn set_cache_enabled(&mut self, v: bool) { self.inner.cache_enabled = v; }
}

#[pyclass(name = "SearchStats")]
#[derive(Clone, Copy)]
pub struct PySearchStats { inner: SearchStats }

#[pymethods]
impl PySearchStats {
    #[getter]
    fn nodes(&self) -> u64 {
        self.inner.nodes
    }

    #[getter]
    fn peak_nodes(&self) -> u64 {
        self.inner.peak_nodes
    }
}

#[pyclass(name = "BranchEval")]
#[derive(Clone, Copy)]
pub struct PyBranchEval {
    #[pyo3(get)]
    direction: PyMove,
    #[pyo3(get)]
    expected_value: f64,
    #[pyo3(get)]
    is_legal: bool,
}

impl From<RsBranchEval> for PyBranchEval {
    fn from(be: RsBranchEval) -> Self {
        PyBranchEval { direction: be.dir.into(), expected_value: be.ev, is_legal: be.legal }
    }
}

impl PyBranchEval {
    pub(crate) fn to_raw(&self) -> RsBranchEval {
        RsBranchEval { dir: self.direction.into(), ev: self.expected_value, legal: self.is_legal }
    }
}

#[pyclass(name = "Expectimax")]
pub struct PyExpectimax {
    inner: Expectimax,
}

#[pymethods]
impl PyExpectimax {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyRef<'_, PyExpectimaxConfig>>) -> Self {
        let inner_config = config.map_or_else(ExpectimaxConfig::default, |c| c.inner.clone());
        PyExpectimax {
            inner: Expectimax::with_config(inner_config),
        }
    }

    fn best_move(&mut self, py: Python<'_>, board: PyRef<'_, PyBoard>) -> Option<PyMove> {
        // Extract the raw Board value before releasing the GIL to avoid capturing PyRef in the closure.
        let b = board.inner();
        py.allow_threads(|| self.inner.best_move(b).map(|m| m.into()))
    }

    fn branch_evals(&mut self, py: Python<'_>, board: PyRef<'_, PyBoard>) -> Vec<PyBranchEval> {
        let b = board.inner();
        py.allow_threads(|| self.inner.branch_evals(b).into_iter().map(Into::into).collect())
    }

    fn state_value(&mut self, py: Python<'_>, board: PyRef<'_, PyBoard>) -> f64 {
        let b = board.inner();
        py.allow_threads(|| self.inner.state_value(b))
    }

    fn last_stats(&self) -> PySearchStats {
        PySearchStats {
            inner: self.inner.last_stats(),
        }
    }

    fn reset_stats(&mut self) {
        self.inner.reset_stats();
    }
}
