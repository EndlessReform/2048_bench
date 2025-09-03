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

    /// Create deterministic train/test views over this packfile.
    ///
    /// Parameters:
    /// - unit: "run" (default; split by runs) or "step" (split targeting total steps; may overshoot to whole runs).
    /// - test_pct: fraction in (0,1) for test partition size (mutually exclusive with test_size).
    /// - test_size: integer count in the chosen unit (mutually exclusive with test_pct).
    /// - seed: RNG seed for deterministic random ordering when order = "random" (default 0).
    /// - order: "random" (default) or "sequential" (respect pack order before carving off test set).
    ///
    /// Returns (train_view, test_view). Train and test are disjoint and cover the selected universe.
    #[pyo3(signature = (unit=None, test_pct=None, test_size=None, seed=None, order=None))]
    pub fn split(
        &self,
        py: Python<'_>,
        unit: Option<&str>,
        test_pct: Option<f64>,
        test_size: Option<usize>,
        seed: Option<u64>,
        order: Option<&str>,
    ) -> PyResult<(Py<PyPackView>, Py<PyPackView>)> {
        let unit = unit.unwrap_or("run");
        if !(unit == "run" || unit == "step") {
            return Err(pyo3::exceptions::PyValueError::new_err("unit must be 'run' or 'step'"));
        }
        if test_pct.is_some() && test_size.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err("provide only one of test_pct or test_size"));
        }
        if test_pct.is_none() && test_size.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err("must provide test_pct or test_size"));
        }
        if let Some(p) = test_pct { if !(p > 0.0 && p < 1.0) {
            return Err(pyo3::exceptions::PyValueError::new_err("test_pct must be in (0,1)"));
        }}
        let order = order.unwrap_or("random");
        if !(order == "random" || order == "sequential") {
            return Err(pyo3::exceptions::PyValueError::new_err("order must be 'random' or 'sequential'"));
        }

        let n_runs = self.inner.len();
        let mut idxs: Vec<usize> = (0..n_runs).collect();
        if order == "random" {
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0));
            idxs.as_mut_slice().shuffle(&mut rng);
        }

        // Helper to decode step counts for all runs
        let steps_for = |i: usize| -> PyResult<usize> {
            let inner = self.inner.clone();
            let run = py.allow_threads(move || inner.decode_auto_v2(i).map_err(map_pack_err))?;
            Ok(run.meta.steps as usize)
        };

        enum ViewUnit { Run, Step }
        let vunit = if unit == "run" { ViewUnit::Run } else { ViewUnit::Step };

        // Determine test selection
        let test_runs: Vec<usize> = match vunit {
            ViewUnit::Run => {
                let target = if let Some(p) = test_pct {
                    let mut c = (p * (n_runs as f64)).floor() as usize;
                    if c == 0 && p > 0.0 { c = 1; }
                    c.min(n_runs)
                } else {
                    test_size.unwrap().min(n_runs)
                };
                idxs[..target].to_vec()
            }
            ViewUnit::Step => {
                // Compute total steps if pct
                let total_steps = if test_pct.is_some() {
                    let mut total = 0usize;
                    for &i in &idxs { total += steps_for(i)?; }
                    total
                } else { 0 };
                let target_steps = if let Some(p) = test_pct {
                    ((p * (total_steps as f64)).floor() as usize).max(1)
                } else {
                    test_size.unwrap().max(1)
                };
                let mut acc = 0usize;
                let mut chosen: Vec<usize> = Vec::new();
                for &i in &idxs {
                    let s = steps_for(i)?;
                    chosen.push(i);
                    acc += s;
                    if acc >= target_steps { break; }
                }
                chosen
            }
        };

        // Train = all minus test
        use std::collections::HashSet;
        let test_set: HashSet<usize> = test_runs.iter().copied().collect();
        let train_runs: Vec<usize> = (0..n_runs).filter(|i| !test_set.contains(i)).collect();

        let train = Py::new(py, PyPackView::new(self.inner.clone(), unit == "run", train_runs))?;
        let test = Py::new(py, PyPackView::new(self.inner.clone(), unit == "run", test_runs))?;
        Ok((train, test))
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

    #[pyo3(signature = (path, parallel=None, by_step=None, progress=None))]
    pub fn to_jsonl(&self, py: Python<'_>, path: PathBuf, parallel: Option<bool>, by_step: Option<bool>, progress: Option<bool>) -> PyResult<()> {
        let inner = self.inner.clone();
        py.allow_threads(move || {
            inner
                .to_jsonl(path, parallel.unwrap_or(true), by_step.unwrap_or(false), progress.unwrap_or(true))
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
        if shuffle.unwrap_or(false) {
            // Shuffled: materialize all (run_idx, step_idx) pairs (decodes each run once for step count)
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            for i in 0..self.inner.len() {
                let steps = {
                    let inner = self.inner.clone();
                    let run = py.allow_threads(move || inner.decode_auto_v2(i).map_err(map_pack_err))?;
                    run.meta.steps as usize
                };
                for si in 0..steps { pairs.push((i, si)); }
            }
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0));
            pairs.as_mut_slice().shuffle(&mut rng);
            Py::new(
                py,
                PyStepBatchesIter {
                    reader: self.inner.clone(),
                    pairs,
                    pos: 0,
                    batch_size,
                    run_cache: std::collections::HashMap::with_capacity(16),
                    lru: std::collections::VecDeque::with_capacity(16),
                    cache_cap: 16,
                    streaming: false,
                    runs_stream: Vec::new(),
                    run_pos_stream: 0,
                    step_pos_stream: 0,
                    current_run_stream: None,
                },
            )
        } else {
            // Non-shuffled: stream runs sequentially; avoid upfront decodes
            let runs: Vec<usize> = (0..self.inner.len()).collect();
            Py::new(
                py,
                PyStepBatchesIter {
                    reader: self.inner.clone(),
                    pairs: Vec::new(),
                    pos: 0,
                    batch_size,
                    run_cache: std::collections::HashMap::with_capacity(16),
                    lru: std::collections::VecDeque::with_capacity(16),
                    cache_cap: 16,
                    streaming: true,
                    runs_stream: runs,
                    run_pos_stream: 0,
                    step_pos_stream: 0,
                    current_run_stream: None,
                },
            )
        }
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
    pairs: Vec<(usize, usize)>, // (run_idx, step_idx) when shuffled mode
    pos: usize,
    batch_size: usize,
    // Small decoded-run cache to avoid re-decoding the same run across batches
    run_cache: std::collections::HashMap<usize, Arc<ser::RunV2>>, // run_idx -> run
    lru: std::collections::VecDeque<usize>, // FIFO eviction for simplicity
    cache_cap: usize,
    // Streaming (non-shuffled) mode state
    streaming: bool,
    runs_stream: Vec<usize>,
    run_pos_stream: usize,
    step_pos_stream: usize,
    current_run_stream: Option<Arc<ser::RunV2>>,
}

#[pymethods]
impl PyStepBatchesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    /// Yield next batch as (pre_boards, chosen_dirs, branch_evs)
    /// - pre_boards: List[List[int]] with 16 exponents in row-major order
    /// - chosen_dirs: List[int] with 0..3 for Up,Down,Left,Right
    /// - branch_evs: List[List[BranchV2]] with 4 entries [Up, Down, Left, Right]
    fn __next__(&mut self, py: Python<'_>) -> Option<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        if self.streaming {
            return self.next_streaming(py);
        }
        if self.pos >= self.pairs.len() { return None; }
        let start = self.pos;
        let end = (self.pos + self.batch_size).min(self.pairs.len());
        self.pos = end;

        let count = end - start;
        let mut pre_boards: Vec<Vec<u8>> = Vec::with_capacity(count);
        let mut chosen_dirs: Vec<u8> = Vec::with_capacity(count);
        let mut branches_vec: Vec<[ser::BranchV2; 4]> = Vec::with_capacity(count);

        for i in start..end {
            let (ri, si) = self.pairs[i];
            // Fetch from small cache or decode
            let run = match self.fetch_run(py, ri) {
                Some(r) => r,
                None => return None,
            };
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

            pre_boards.push(exps);
            chosen_dirs.push(dir_idx);
            branches_vec.push(branches);
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

    fn next_streaming(&mut self, py: Python<'_>) -> Option<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        if self.run_pos_stream >= self.runs_stream.len() { return None; }
        let mut produced = 0usize;
        let mut pre_boards: Vec<Vec<u8>> = Vec::with_capacity(self.batch_size);
        let mut chosen_dirs: Vec<u8> = Vec::with_capacity(self.batch_size);
        let mut branches_vec: Vec<[ser::BranchV2; 4]> = Vec::with_capacity(self.batch_size);

        while produced < self.batch_size {
            if self.run_pos_stream >= self.runs_stream.len() { break; }
            // Ensure current run is loaded
            if self.current_run_stream.is_none() {
                let ri = self.runs_stream[self.run_pos_stream];
                self.current_run_stream = self.fetch_run(py, ri);
                if self.current_run_stream.is_none() { return None; }
                self.step_pos_stream = 0;
            }
            let run = self.current_run_stream.as_ref().unwrap().clone();
            let total_steps = run.steps.len();
            while produced < self.batch_size && self.step_pos_stream < total_steps {
                let st = &run.steps[self.step_pos_stream];
                let exps: Vec<u8> = RsBoard::from_raw(st.pre_board).to_vec();
                let dir_idx: u8 = match st.chosen { RsMove::Up => 0, RsMove::Down => 1, RsMove::Left => 2, RsMove::Right => 3 };
                let mut branches = if let Some(arr) = st.branches { arr } else {
                    let mut tmp = [ser::BranchV2::Illegal; 4];
                    tmp[dir_idx as usize] = ser::BranchV2::Legal(1.0);
                    tmp
                };
                if let ser::BranchV2::Legal(v) = branches[dir_idx as usize] {
                    if v > 1.0 - 1e-6 { branches[dir_idx as usize] = ser::BranchV2::Legal(1.0); }
                }
                pre_boards.push(exps);
                chosen_dirs.push(dir_idx);
                branches_vec.push(branches);
                self.step_pos_stream += 1;
                produced += 1;
            }
            if self.step_pos_stream >= total_steps {
                self.run_pos_stream += 1;
                self.current_run_stream = None;
            }
        }
        if produced == 0 { return None; }

        let py_pre = pyo3::types::PyList::new_bound(py, &pre_boards).unbind().into_any();
        let py_dirs = pyo3::types::PyList::new_bound(py, &chosen_dirs).unbind().into_any();
        let outer = pyo3::types::PyList::empty_bound(py);
        for arr in branches_vec.into_iter() {
            let inner = pyo3::types::PyList::empty_bound(py);
            for b in arr {
                match Py::new(py, PyBranchV2::from(b)) { Ok(obj) => { let _ = inner.append(obj); }, Err(_) => return None }
            }
            let _ = outer.append(inner);
        }
        Some((py_pre, py_dirs, outer.unbind().into_any()))
    }
}

// Add to existing PyPackReader methods block above

// ------------------------
// PackView: a view over a subset
// ------------------------

#[pyclass(module = "ai_2048", name = "PackView")]
pub struct PyPackView {
    reader: Arc<ser::PackReader>,
    // if by_runs == true, `runs` is the selected run indices and iter_step_batches will expand them to step pairs
    // if by_runs == false (step-based split), we still keep selected runs and build step pairs from them on demand
    runs: Vec<usize>,
    by_runs: bool,
}

impl PyPackView {
    pub fn new(reader: Arc<ser::PackReader>, by_runs: bool, runs: Vec<usize>) -> Self {
        Self { reader, runs, by_runs }
    }
}

#[pymethods]
impl PyPackView {
    fn __len__(&self) -> usize { self.runs.len() }

    /// Iterate step-level batches restricted to this view.
    /// Parameters mirror `PackReader::iter_step_batches` but operate only on the selected runs.
    #[pyo3(signature = (batch_size, shuffle=None, seed=None))]
    pub fn iter_step_batches(
        &self,
        py: Python<'_>,
        batch_size: usize,
        shuffle: Option<bool>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyStepBatchesIter>> {
        if shuffle.unwrap_or(false) {
            // Shuffled: materialize (run_idx, step_idx) pairs for selected runs
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            for &ri in &self.runs {
                let reader = self.reader.clone();
                let steps = py.allow_threads(move || reader.decode_auto_v2(ri).map_err(map_pack_err))?.meta.steps as usize;
                for si in 0..steps { pairs.push((ri, si)); }
            }
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0));
            pairs.as_mut_slice().shuffle(&mut rng);
            Py::new(
                py,
                PyStepBatchesIter {
                    reader: self.reader.clone(),
                    pairs,
                    pos: 0,
                    batch_size,
                    run_cache: std::collections::HashMap::with_capacity(16),
                    lru: std::collections::VecDeque::with_capacity(16),
                    cache_cap: 16,
                    streaming: false,
                    runs_stream: Vec::new(),
                    run_pos_stream: 0,
                    step_pos_stream: 0,
                    current_run_stream: None,
                },
            )
        } else {
            // Non-shuffled: stream within selected runs without upfront decode
            Py::new(
                py,
                PyStepBatchesIter {
                    reader: self.reader.clone(),
                    pairs: Vec::new(),
                    pos: 0,
                    batch_size,
                    run_cache: std::collections::HashMap::with_capacity(16),
                    lru: std::collections::VecDeque::with_capacity(16),
                    cache_cap: 16,
                    streaming: true,
                    runs_stream: self.runs.clone(),
                    run_pos_stream: 0,
                    step_pos_stream: 0,
                    current_run_stream: None,
                },
            )
        }
    }

    /// Iterate run-level batches restricted to this view.
    #[pyo3(signature = (batch_size, shuffle=None, seed=None))]
    pub fn iter_batches(
        &self,
        py: Python<'_>,
        batch_size: usize,
        shuffle: Option<bool>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyPackBatchesIter>> {
        let mut idxs = self.runs.clone();
        if shuffle.unwrap_or(false) {
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0));
            idxs.as_mut_slice().shuffle(&mut rng);
        }
        Py::new(
            py,
            PyPackBatchesIter { reader: self.reader.clone(), idxs, pos: 0, batch_size },
        )
    }

    /// Simple iterator over runs in this view.
    pub fn iter(&self, py: Python<'_>) -> PyResult<Py<PyPackIter>> {
        Py::new(
            py,
            PyPackIter { reader: self.reader.clone(), idxs: self.runs.clone(), pos: 0 },
        )
    }

    /// Return the run indices backing this view.
    pub fn indices(&self) -> Vec<usize> { self.runs.clone() }
}

impl PyStepBatchesIter {
    #[inline]
    fn fetch_run(&mut self, py: Python<'_>, idx: usize) -> Option<Arc<ser::RunV2>> {
        if let Some(r) = self.run_cache.get(&idx) {
            return Some(r.clone());
        }
        let reader = self.reader.clone();
        let run = match py.allow_threads(move || reader.decode_auto_v2(idx).map_err(map_pack_err)) {
            Ok(r) => Arc::new(r),
            Err(_) => return None,
        };
        if self.run_cache.len() >= self.cache_cap {
            if let Some(old) = self.lru.pop_front() {
                self.run_cache.remove(&old);
            }
        }
        self.lru.push_back(idx);
        self.run_cache.insert(idx, run.clone());
        Some(run)
    }
}
