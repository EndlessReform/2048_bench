//! PyO3 bindings for run serialization (v2, postcard)

use pyo3::prelude::*;
use pyo3::types::{PyList, PyBytes};
use std::path::PathBuf;

use ai_2048_lib::engine::state::Board as RsBoard;
use ai_2048_lib::engine::Move as RsMove;
use ai_2048_lib::expectimax::BranchEval as RsBranchEval;
use ai_2048_lib::serialization as ser;
use ai_2048_lib::trace::Meta as RsMeta;

use crate::board::PyBoard;
use crate::expectimax::PyBranchEval;
use crate::move_enum::PyMove;

#[pyclass(module = "ai_2048", name = "Meta")]
#[derive(Clone, Debug)]
pub struct PyMeta {
    inner: RsMeta,
}

#[pymethods]
impl PyMeta {
    #[new]
    #[pyo3(signature = (steps, start_unix_s, elapsed_s, max_score, highest_tile, engine_str=None))]
    fn new(
        steps: u32,
        start_unix_s: u64,
        elapsed_s: f32,
        max_score: u64,
        highest_tile: u32,
        engine_str: Option<String>,
    ) -> Self {
        Self { inner: RsMeta { steps, start_unix_s, elapsed_s, max_score, highest_tile, engine_str } }
    }

    #[getter]
    fn steps(&self) -> u32 { self.inner.steps }
    #[getter]
    fn start_unix_s(&self) -> u64 { self.inner.start_unix_s }
    #[getter]
    fn elapsed_s(&self) -> f32 { self.inner.elapsed_s }
    #[getter]
    fn max_score(&self) -> u64 { self.inner.max_score }
    #[getter]
    fn highest_tile(&self) -> u32 { self.inner.highest_tile }
    #[getter]
    fn engine_str(&self) -> Option<String> { self.inner.engine_str.clone() }
}

impl From<RsMeta> for PyMeta { fn from(inner: RsMeta) -> Self { Self { inner } } }
impl From<PyMeta> for RsMeta { fn from(p: PyMeta) -> Self { p.inner } }

#[pyclass(module = "ai_2048", name = "BranchV2")]
#[derive(Clone, Debug)]
pub struct PyBranchV2 { inner: ser::BranchV2 }

#[pymethods]
impl PyBranchV2 {
    #[staticmethod]
    fn legal(value: f32) -> Self { PyBranchV2 { inner: ser::BranchV2::Legal(value) } }
    #[staticmethod]
    fn illegal() -> Self { PyBranchV2 { inner: ser::BranchV2::Illegal } }

    #[getter]
    fn is_legal(&self) -> bool { matches!(self.inner, ser::BranchV2::Legal(_)) }
    #[getter]
    fn value(&self) -> Option<f32> { match self.inner { ser::BranchV2::Legal(v) => Some(v), ser::BranchV2::Illegal => None } }

    fn __repr__(&self) -> String {
        match self.inner {
            ser::BranchV2::Legal(v) => format!("BranchV2.legal({:.6})", v),
            ser::BranchV2::Illegal => "BranchV2.illegal()".to_string(),
        }
    }
}

impl From<ser::BranchV2> for PyBranchV2 { fn from(inner: ser::BranchV2) -> Self { Self { inner } } }
impl From<PyBranchV2> for ser::BranchV2 { fn from(p: PyBranchV2) -> Self { p.inner } }

#[pyclass(module = "ai_2048", name = "StepV2")]
#[derive(Clone, Debug)]
pub struct PyStepV2 { inner: ser::StepV2 }

#[pymethods]
impl PyStepV2 {
    #[new]
    #[pyo3(signature = (pre_board_raw, chosen, branches=None))]
    fn new(pre_board_raw: u64, chosen: PyMove, branches: Option<Vec<PyBranchV2>>) -> Self {
        let branches_inner = branches.map(|v| {
            let mut arr: [ser::BranchV2; 4] = [ser::BranchV2::Illegal; 4];
            for (i, b) in v.into_iter().take(4).enumerate() { arr[i] = b.into(); }
            arr
        });
        Self { inner: ser::StepV2 { pre_board: pre_board_raw, chosen: chosen.into(), branches: branches_inner } }
    }

    #[getter]
    fn pre_board_raw(&self) -> u64 { self.inner.pre_board }
    #[getter]
    fn pre_board(&self) -> PyBoard { RsBoard::from_raw(self.inner.pre_board).into() }
    #[getter]
    fn chosen(&self) -> PyMove { self.inner.chosen.into() }
    #[getter]
    fn branches(&self, py: Python<'_>) -> Option<Py<PyList>> {
        self.inner.branches.map(|arr| {
            let list = PyList::empty_bound(py);
            for b in arr.into_iter().map(PyBranchV2::from) {
                let obj = Py::new(py, b).expect("alloc PyBranchV2");
                list.append(obj).expect("append branch");
            }
            list.into()
        })
    }
}

impl From<ser::StepV2> for PyStepV2 { fn from(inner: ser::StepV2) -> Self { Self { inner } } }
impl From<PyStepV2> for ser::StepV2 { fn from(p: PyStepV2) -> Self { p.inner } }

#[pyclass(module = "ai_2048", name = "RunV2")]
#[derive(Clone, Debug)]
pub struct PyRunV2 { inner: ser::RunV2 }

#[pymethods]
impl PyRunV2 {
    #[new]
    fn new(meta: PyMeta, steps: Vec<PyStepV2>, final_board_raw: u64) -> Self {
        let meta_inner: RsMeta = meta.into();
        let steps_inner: Vec<ser::StepV2> = steps.into_iter().map(Into::into).collect();
        Self { inner: ser::RunV2 { meta: meta_inner, steps: steps_inner, final_board: final_board_raw } }
    }

    #[getter]
    fn meta(&self) -> PyMeta { self.inner.meta.clone().into() }
    #[getter]
    fn steps(&self, py: Python<'_>) -> Py<PyList> {
        let list = PyList::empty_bound(py);
        for s in self.inner.steps.clone().into_iter().map(PyStepV2::from) {
            let obj = Py::new(py, s).expect("alloc PyStepV2");
            list.append(obj).expect("append step");
        }
        list.into()
    }
    #[getter]
    fn final_board_raw(&self) -> u64 { self.inner.final_board }
    #[getter]
    fn final_board(&self) -> PyBoard { RsBoard::from_raw(self.inner.final_board).into() }

    fn to_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let v = ser::to_postcard_bytes(&self.inner).map_err(map_ser_err)?;
        Ok(PyBytes::new_bound(py, &v).unbind())
    }
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> { Ok(PyRunV2 { inner: ser::from_postcard_bytes(data).map_err(map_ser_err)? }) }

    fn save(&self, path: PathBuf) -> PyResult<()> { ser::write_postcard_to_path(path, &self.inner).map_err(map_ser_err) }
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> { Ok(PyRunV2 { inner: ser::read_postcard_from_path(path).map_err(map_ser_err)? }) }
}

// Conversions for ergonomic bridging from Rust library to PyO3 wrapper
impl From<ser::RunV2> for PyRunV2 { fn from(inner: ser::RunV2) -> Self { Self { inner } } }
impl From<PyRunV2> for ser::RunV2 { fn from(p: PyRunV2) -> Self { p.inner } }

#[pyfunction]
pub fn normalize_branches_py(evals: Vec<PyBranchEval>) -> PyResult<Vec<PyBranchV2>> {
    if evals.len() != 4 { return Err(pyo3::exceptions::PyValueError::new_err("expected 4 branch evals (Up, Down, Left, Right)")); }
    let mut arr: [RsBranchEval; 4] = [
        RsBranchEval { dir: RsMove::Up, ev: 0.0, legal: false },
        RsBranchEval { dir: RsMove::Down, ev: 0.0, legal: false },
        RsBranchEval { dir: RsMove::Left, ev: 0.0, legal: false },
        RsBranchEval { dir: RsMove::Right, ev: 0.0, legal: false },
    ];
    for (i, e) in evals.into_iter().enumerate() {
        arr[i] = e.to_raw();
    }
    let out = ser::normalize_branches(arr);
    Ok(out.into_iter().map(Into::into).collect())
}

fn map_ser_err(err: ser::SerializationError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("serialization error: {}", err))
}
