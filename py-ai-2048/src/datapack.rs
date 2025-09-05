//! PyO3 bindings for the RAM-friendly DataPack (.dat)

use pyo3::prelude::*;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use ai_2048_lib::engine::state::Board as RsBoard;
use ai_2048_lib::serialization::{DataPack, DataPackError};
use ai_2048_lib::serialization as ser;

#[pyclass(module = "ai_2048", name = "Dataset")]
pub struct PyDataset {
    pack: Arc<DataPack>,
    // Selected step indices into pack.steps (identity if full view)
    indices: Vec<usize>,
}

#[pymethods]
impl PyDataset {
    /// Load a dataset pack (.dat) into RAM. Decoding is parallelized.
    #[new]
    pub fn new(path: PathBuf) -> PyResult<Self> {
        let pack = DataPack::load(&path).map_err(map_err)?;
        let indices: Vec<usize> = (0..pack.steps.len()).collect();
        Ok(Self { pack: Arc::new(pack), indices })
    }

    fn __len__(&self) -> usize { self.indices.len() }

    /// Return a tuple (pre_boards, chosen_dirs, branch_evs) for the provided local indices.
    /// - pre_boards: List[List[int]] with shape (N, 16) exponents (0=empty, 1->2, ...)
    /// - chosen_dirs: List[int] with 0:Up, 1:Down, 2:Left, 3:Right
    /// - branch_evs: List[List[BranchV2]] with 4 entries [Up, Down, Left, Right]
    pub fn get_batch(&self, py: Python<'_>, idxs: Vec<usize>) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let mut pre_boards: Vec<Vec<u8>> = Vec::with_capacity(idxs.len());
        let mut dirs: Vec<u8> = Vec::with_capacity(idxs.len());
        let mut branches_vec: Vec<[ser::BranchV2; 4]> = Vec::with_capacity(idxs.len());
        for i in idxs {
            if i >= self.indices.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
            }
            let gi = self.indices[i];
            let st = &self.pack.steps[gi];
            let exps = RsBoard::from_raw(st.board).to_vec();
            pre_boards.push(exps);
            dirs.push(st.move_dir);
            // Reconstruct BranchV2 from quantized EVs and mask
            let mut arr = [ser::BranchV2::Illegal; 4];
            for bi in 0..4 {
                let legal = (st.ev_mask & (1u8 << bi)) != 0;
                if legal {
                    let v = (st.ev_q[bi] as f32) / 65535.0;
                    arr[bi] = ser::BranchV2::Legal(v);
                }
            }
            branches_vec.push(arr);
        }
        let py_pre = pyo3::types::PyList::new_bound(py, &pre_boards).unbind().into_any();
        let py_dirs = pyo3::types::PyList::new_bound(py, &dirs).unbind().into_any();
        // Convert to nested list of BranchV2 objects
        let outer = pyo3::types::PyList::empty_bound(py);
        for arr in branches_vec.into_iter() {
            let inner = pyo3::types::PyList::empty_bound(py);
            for b in arr {
                match Py::new(py, crate::serialization::PyBranchV2::from(b)) {
                    Ok(obj) => { let _ = inner.append(obj); },
                    Err(e) => return Err(e),
                }
            }
            let _ = outer.append(inner);
        }
        Ok((py_pre, py_dirs, outer.unbind().into_any()))
    }

    /// Filter by run max score (inclusive range). Returns a new Dataset view.
    pub fn filter_by_score(&self, min_score: u64, max_score: u64) -> Self {
        let valid_runs: HashSet<u32> = self.pack
            .runs
            .iter()
            .filter(|r| r.max_score >= min_score && r.max_score <= max_score)
            .map(|r| r.id)
            .collect();
        let indices: Vec<usize> = self
            .indices
            .iter()
            .copied()
            .filter(|&gi| valid_runs.contains(&self.pack.steps[gi].run_id))
            .collect();
        Self { pack: Arc::clone(&self.pack), indices }
    }
}

fn map_err(err: DataPackError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("datapack error: {}", err))
}
