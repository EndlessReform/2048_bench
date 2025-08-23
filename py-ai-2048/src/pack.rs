//! PyO3 bindings for PackReader (readonly a2pack files)

use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

use ai_2048_lib::serialization as ser;

use crate::serialization::PyRunV2;
use rand::{rngs::StdRng, SeedableRng};
use rand::seq::SliceRandom;

#[pyclass(module = "ai_2048", name = "PackStats")]
#[derive(Clone, Debug)]
pub struct PyPackStats {
    #[pyo3(get)]
    pub count: u64,
    #[pyo3(get)]
    pub total_steps: u64,
    #[pyo3(get)]
    pub min_len: u32,
    #[pyo3(get)]
    pub max_len: u32,
    #[pyo3(get)]
    pub mean_len: f64,
}

impl From<ser::PackStats> for PyPackStats {
    fn from(s: ser::PackStats) -> Self {
        PyPackStats { count: s.count, total_steps: s.total_steps, min_len: s.min_len, max_len: s.max_len, mean_len: s.mean_len }
    }
}

#[pyclass(module = "ai_2048", name = "PackReader")]
pub struct PyPackReader { inner: Arc<ser::PackReader> }

#[pymethods]
impl PyPackReader {
    #[staticmethod]
    pub fn open(path: PathBuf) -> PyResult<Self> {
        let inner = ser::PackReader::open(path).map_err(map_pack_err)?;
        Ok(Self { inner: Arc::new(inner) })
    }

    fn __len__(&self) -> usize { self.inner.len() }

    pub fn kind(&self, i: usize) -> PyResult<String> {
        let k = self.inner.kind(i).map_err(map_pack_err)?;
        Ok(match k { ser::RunKind::V1 => "v1".to_string(), ser::RunKind::V2 => "v2".to_string() })
    }

    pub fn decode(&self, py: Python<'_>, i: usize) -> PyResult<Py<PyRunV2>> {
        let inner = self.inner.clone();
        let run = py.allow_threads(move || inner.decode_auto_v2(i).map_err(map_pack_err))?;
        Py::new(py, PyRunV2::from(run))
    }

    pub fn decode_batch(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Vec<Py<PyRunV2>>> {
        let inner = self.inner.clone();
        let runs = py.allow_threads(move || inner.decode_batch_auto_v2(&indices).map_err(map_pack_err))?;
        let mut out = Vec::with_capacity(runs.len());
        for r in runs { out.push(Py::new(py, PyRunV2::from(r))?); }
        Ok(out)
    }

    #[pyo3(signature = (path, parallel=None))]
    pub fn to_jsonl(&self, py: Python<'_>, path: PathBuf, parallel: Option<bool>) -> PyResult<()> {
        let inner = self.inner.clone();
        py.allow_threads(move || inner.to_jsonl(path, parallel.unwrap_or(true)).map_err(map_pack_err))
    }

    #[getter]
    pub fn stats(&self, py: Python<'_>) -> PyResult<PyPackStats> {
        let inner = self.inner.clone();
        let s = py.allow_threads(move || inner.stats().map_err(map_pack_err))?;
        Ok(PyPackStats::from(s))
    }

    // Iterators
    pub fn iter(&self, py: Python<'_>) -> PyResult<Py<PyPackIter>> {
        let idxs: Vec<usize> = (0..self.inner.len()).collect();
        Py::new(py, PyPackIter { reader: self.inner.clone(), idxs, pos: 0 })
    }

    pub fn iter_indices(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Py<PyPackIter>> {
        Py::new(py, PyPackIter { reader: self.inner.clone(), idxs: indices, pos: 0 })
    }

    #[pyo3(signature = (batch_size, shuffle=None, seed=None))]
    pub fn iter_batches(&self, py: Python<'_>, batch_size: usize, shuffle: Option<bool>, seed: Option<u64>) -> PyResult<Py<PyPackBatchesIter>> {
        let mut idxs: Vec<usize> = (0..self.inner.len()).collect();
        if shuffle.unwrap_or(false) {
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0));
            idxs.as_mut_slice().shuffle(&mut rng);
        }
        Py::new(py, PyPackBatchesIter { reader: self.inner.clone(), idxs, pos: 0, batch_size })
    }
}

fn map_pack_err(err: ser::PackError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("pack error: {}", err))
}

#[pyclass(module = "ai_2048", name = "_PackIter")]
pub struct PyPackIter {
    reader: Arc<ser::PackReader>,
    idxs: Vec<usize>,
    pos: usize,
}

#[pymethods]
impl PyPackIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __next__(&mut self, py: Python<'_>) -> Option<Py<PyRunV2>> {
        if self.pos >= self.idxs.len() { return None; }
        let idx = self.idxs[self.pos];
        self.pos += 1;
        let reader = self.reader.clone();
        let res = py.allow_threads(move || reader.decode_auto_v2(idx).map_err(map_pack_err));
        match res { Ok(run) => Py::new(py, PyRunV2::from(run)).ok(), Err(_) => None }
    }
}

#[pyclass(module = "ai_2048", name = "_PackBatchesIter")]
pub struct PyPackBatchesIter {
    reader: Arc<ser::PackReader>,
    idxs: Vec<usize>,
    pos: usize,
    batch_size: usize,
}

#[pymethods]
impl PyPackBatchesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __next__(&mut self, py: Python<'_>) -> Option<Vec<Py<PyRunV2>>> {
        if self.pos >= self.idxs.len() { return None; }
        let start = self.pos;
        let end = (self.pos + self.batch_size).min(self.idxs.len());
        self.pos = end;
        let slice: Vec<usize> = self.idxs[start..end].to_vec();
        let reader = self.reader.clone();
        let res = py.allow_threads(move || reader.decode_batch_auto_v2(&slice).map_err(map_pack_err));
        match res {
            Ok(runs) => {
                let mut out = Vec::with_capacity(runs.len());
                for r in runs { if let Ok(pyobj) = Py::new(py, PyRunV2::from(r)) { out.push(pyobj); } }
                Some(out)
            }
            Err(_) => None,
        }
    }
}
