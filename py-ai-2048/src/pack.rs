//! PyO3 bindings for PackReader (readonly a2pack files)

use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

use ai_2048_lib::serialization as ser;

use crate::serialization::{PyRunV2, PyBranchV2};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use ai_2048_lib::engine::state::Board as RsBoard;
use ai_2048_lib::engine::Move as RsMove;

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
    #[pyo3(get)]
    pub p50: u32,
    #[pyo3(get)]
    pub p90: u32,
    #[pyo3(get)]
    pub p99: u32,
}

impl From<ser::PackStats> for PyPackStats {
    fn from(s: ser::PackStats) -> Self {
        PyPackStats {
            count: s.count,
            total_steps: s.total_steps,
            min_len: s.min_len,
            max_len: s.max_len,
            mean_len: s.mean_len,
            p50: s.p50,
            p90: s.p90,
            p99: s.p99,
        }
    }
}

#[pyclass(module = "ai_2048", name = "PackReader")]
pub struct PyPackReader {
    inner: Arc<ser::PackReader>,
}

#[pymethods]
impl PyPackReader {
    #[staticmethod]
    pub fn open(path: PathBuf) -> PyResult<Self> {
        let inner = ser::PackReader::open(path).map_err(map_pack_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn kind(&self, i: usize) -> PyResult<String> {
        let k = self.inner.kind(i).map_err(map_pack_err)?;
        Ok(match k {
            ser::RunKind::V1 => "v1".to_string(),
            ser::RunKind::V2 => "v2".to_string(),
        })
    }

    pub fn decode(&self, py: Python<'_>, i: usize) -> PyResult<Py<PyRunV2>> {
        let inner = self.inner.clone();
        let run = py.allow_threads(move || inner.decode_auto_v2(i).map_err(map_pack_err))?;
        Py::new(py, PyRunV2::from(run))
    }

    pub fn decode_batch(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Vec<Py<PyRunV2>>> {
        let inner = self.inner.clone();
        let runs =
            py.allow_threads(move || inner.decode_batch_auto_v2(&indices).map_err(map_pack_err))?;
        let mut out = Vec::with_capacity(runs.len());
        for r in runs {
            out.push(Py::new(py, PyRunV2::from(r))?);
        }
        Ok(out)
    }

    #[pyo3(signature = (path, parallel=None))]
    pub fn to_jsonl(&self, py: Python<'_>, path: PathBuf, parallel: Option<bool>) -> PyResult<()> {
        let inner = self.inner.clone();
        py.allow_threads(move || {
            inner
                .to_jsonl(path, parallel.unwrap_or(true))
                .map_err(map_pack_err)
        })
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
        Py::new(
            py,
            PyPackIter {
                reader: self.inner.clone(),
                idxs,
                pos: 0,
            },
        )
    }

    pub fn iter_indices(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Py<PyPackIter>> {
        Py::new(
            py,
            PyPackIter {
                reader: self.inner.clone(),
                idxs: indices,
                pos: 0,
            },
        )
    }

    #[pyo3(signature = (batch_size, shuffle=None, seed=None))]
    pub fn iter_batches(
        &self,
        py: Python<'_>,
        batch_size: usize,
        shuffle: Option<bool>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyPackBatchesIter>> {
        let mut idxs: Vec<usize> = (0..self.inner.len()).collect();
        if shuffle.unwrap_or(false) {
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0));
            idxs.as_mut_slice().shuffle(&mut rng);
        }
        Py::new(
            py,
            PyPackBatchesIter {
                reader: self.inner.clone(),
                idxs,
                pos: 0,
                batch_size,
            },
        )
    }

    /// Iterate step-level batches across the packfile, optionally shuffled.
    /// Each yielded batch is a tuple (pre_boards, chosen_dirs, branch_evs):
    /// - pre_boards: List[List[int]] of size (N, 16) with exponents (0 empty, 1->2, 2->4, ...), c1r1..c4r4 row-major.
    /// - chosen_dirs: List[int] with 0:Up, 1:Down, 2:Left, 3:Right.
    /// - branch_evs: List[List[BranchV2]] with 4 entries [Up, Down, Left, Right]; chosen entry clamped to exactly 1.0 when maximal.
    #[pyo3(signature = (batch_size, shuffle=None, seed=None))]
    pub fn iter_step_batches(
        &self,
        py: Python<'_>,
        batch_size: usize,
        shuffle: Option<bool>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyStepBatchesIter>> {
        // Materialize all (run_idx, step_idx) pairs
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.inner.len() {
            let steps = {
                let inner = self.inner.clone();
                let run = py.allow_threads(move || inner.decode_auto_v2(i).map_err(map_pack_err))?;
                run.meta.steps as usize
            };
            for si in 0..steps { pairs.push((i, si)); }
        }
        if shuffle.unwrap_or(false) {
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0));
            pairs.as_mut_slice().shuffle(&mut rng);
        }
        Py::new(
            py,
            PyStepBatchesIter {
                reader: self.inner.clone(),
                pairs,
                pos: 0,
                batch_size,
            },
        )
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
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> Option<Py<PyRunV2>> {
        if self.pos >= self.idxs.len() {
            return None;
        }
        let idx = self.idxs[self.pos];
        self.pos += 1;
        let reader = self.reader.clone();
        let res = py.allow_threads(move || reader.decode_auto_v2(idx).map_err(map_pack_err));
        match res {
            Ok(run) => Py::new(py, PyRunV2::from(run)).ok(),
            Err(_) => None,
        }
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
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> Option<Vec<Py<PyRunV2>>> {
        if self.pos >= self.idxs.len() {
            return None;
        }
        let start = self.pos;
        let end = (self.pos + self.batch_size).min(self.idxs.len());
        self.pos = end;
        let slice: Vec<usize> = self.idxs[start..end].to_vec();
        let reader = self.reader.clone();
        let res =
            py.allow_threads(move || reader.decode_batch_auto_v2(&slice).map_err(map_pack_err));
        match res {
            Ok(runs) => {
                let mut out = Vec::with_capacity(runs.len());
                for r in runs {
                    if let Ok(pyobj) = Py::new(py, PyRunV2::from(r)) {
                        out.push(pyobj);
                    }
                }
                Some(out)
            }
            Err(_) => None,
        }
    }
}

// ------------------------
// Step-level batching API
// ------------------------

#[pyclass(module = "ai_2048", name = "_StepBatchesIter")]
pub struct PyStepBatchesIter {
    reader: Arc<ser::PackReader>,
    pairs: Vec<(usize, usize)>, // (run_idx, step_idx)
    pos: usize,
    batch_size: usize,
}

#[pymethods]
impl PyStepBatchesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    /// Yield next batch as (pre_boards, chosen_dirs, branch_evs)
    /// - pre_boards: List[List[int]] with 16 exponents in row-major order
    /// - chosen_dirs: List[int] with 0..3 for Up,Down,Left,Right
    /// - branch_evs: List[List[BranchV2]] with 4 entries [Up, Down, Left, Right]
    fn __next__(&mut self, py: Python<'_>) -> Option<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        if self.pos >= self.pairs.len() { return None; }
        let start = self.pos;
        let end = (self.pos + self.batch_size).min(self.pairs.len());
        self.pos = end;
        let slice: &[(usize, usize)] = &self.pairs[start..end];

        // Group by run index to avoid decoding the same run multiple times
        use std::collections::BTreeMap;
        let mut by_run: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &(ri, si) in slice { by_run.entry(ri).or_default().push(si); }

        let mut pre_boards: Vec<Vec<u8>> = Vec::with_capacity(slice.len());
        let mut chosen_dirs: Vec<u8> = Vec::with_capacity(slice.len());
        let mut branches_vec: Vec<[ser::BranchV2; 4]> = Vec::with_capacity(slice.len());

        // We must preserve the order of pairs in `slice`, so collect outputs into a temp map
        let mut out_map: BTreeMap<(usize, usize), (Vec<u8>, u8, [ser::BranchV2; 4])> = BTreeMap::new();

        for (run_idx, step_idxs) in by_run.into_iter() {
            let reader = self.reader.clone();
            // Decode run without holding the GIL
            let run = match py.allow_threads(move || reader.decode_auto_v2(run_idx).map_err(map_pack_err)) {
                Ok(r) => r,
                Err(_) => return None,
            };
            for &si in &step_idxs {
                if si >= run.steps.len() { return None; }
                let st = &run.steps[si];
                let exps: Vec<u8> = RsBoard::from_raw(st.pre_board).to_vec();
                let dir_idx: u8 = match st.chosen { RsMove::Up => 0, RsMove::Down => 1, RsMove::Left => 2, RsMove::Right => 3 };

                // Build branch EVs; if absent, mark chosen as Legal(1.0) and others Illegal.
                let mut branches = if let Some(arr) = st.branches { arr } else {
                    let mut tmp = [ser::BranchV2::Illegal; 4];
                    tmp[dir_idx as usize] = ser::BranchV2::Legal(1.0);
                    tmp
                };
                // Clamp chosen branch to exactly 1.0 if it is near 1.
                if let ser::BranchV2::Legal(v) = branches[dir_idx as usize] {
                    if v > 1.0 - 1e-6 { branches[dir_idx as usize] = ser::BranchV2::Legal(1.0); }
                }

                out_map.insert((run_idx, si), (exps, dir_idx, branches));
            }
        }

        for &(ri, si) in slice {
            if let Some((exps, d, branches)) = out_map.remove(&(ri, si)) {
                pre_boards.push(exps);
                chosen_dirs.push(d);
                branches_vec.push(branches);
            } else {
                return None;
            }
        }

        let py_pre = pyo3::types::PyList::new_bound(py, &pre_boards).unbind().into_any();
        let py_dirs = pyo3::types::PyList::new_bound(py, &chosen_dirs).unbind().into_any();
        // Convert to nested list of BranchV2 objects
        let outer = pyo3::types::PyList::empty_bound(py);
        for arr in branches_vec.into_iter() {
            let inner = pyo3::types::PyList::empty_bound(py);
            for b in arr {
                match Py::new(py, PyBranchV2::from(b)) {
                    Ok(obj) => { let _ = inner.append(obj); },
                    Err(_) => return None,
                }
            }
            let _ = outer.append(inner);
        }
        Some((py_pre, py_dirs, outer.unbind().into_any()))
    }
}

// Add to existing PyPackReader methods block above
