use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

/// Convert a batch of packed `u64` boards into 16-exponent rows as a Python bytearray.
///
/// - Input `boards` must be any Python object exposing a contiguous buffer of `uint64`
///   (e.g., `numpy.ndarray` with `dtype=np.uint64` and C-contiguous layout).
/// - Returns a Python `bytearray` of length `len(boards) * 16`, where each row is
///   16 exponents (0 for empty, 1 for 2, ...), row-major.
/// - Zero-copy into NumPy: `np.frombuffer(out, dtype=np.uint8).reshape(-1, 16)`.
/// - If `parallel` is True, conversion runs without the GIL and uses all cores.
#[pyfunction]
#[pyo3(signature = (boards, parallel = true))]
pub fn exps_from_boards(py: Python<'_>, boards: Bound<'_, PyAny>, parallel: bool) -> PyResult<PyObject> {
    // Borrow a 1-D contiguous u64 buffer from Python (numpy arrays support this)
    let buf = PyBuffer::<u64>::get_bound(&boards)?;
    if buf.dimensions() != 1 {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "boards must be a 1-D buffer of uint64",
        ));
    }
    // Materialize into a Vec<u64> if needed (contiguous copy). Simplicity and safety over zero-copy here.
    let boards_vec: Vec<u64> = buf.to_vec(py)?;
    let n = boards_vec.len();
    let out_len = n.checked_mul(16).ok_or_else(|| {
        pyo3::exceptions::PyOverflowError::new_err("too many boards to convert")
    })?;

    // Allocate Python-owned bytearray (mutable, buffer-compatible)
    let zeros = vec![0u8; out_len];
    let bytearr = PyByteArray::new_bound(py, &zeros);
    // SAFETY: We own the bytearray and hold the GIL; returning object keeps it alive.
    let out_bytes = unsafe { bytearr.as_bytes_mut() };

    // Convert with GIL released. Implementation lives in the Rust library for performance.
    py.allow_threads(move || {
        ai_2048_lib::engine::state::boards_to_exponents_into(out_bytes, &boards_vec, parallel)
    });

    Ok(bytearr.into_py(py))
}

/// Convenience: slice a structured NumPy `steps` array by `idxs` and return a triple
/// (exps_bytearray, dirs_ndarray, evs_ndarray).
///
/// - `steps`: numpy structured array with fields ['board', 'move', 'ev_values'].
/// - `idxs`: 1-D integer numpy array of indices.
/// - Returns:
///   - bytearray of shape N*16 (view as (N,16) uint8 via `np.frombuffer`),
///   - numpy array for moves (shape (N,), integer dtype),
///   - numpy array for ev_values (shape (N,4), float32).
#[pyfunction]
#[pyo3(signature = (steps, idxs, parallel = true))]
pub fn batch_from_steps(
    py: Python<'_>,
    steps: Bound<'_, PyAny>,
    idxs: Bound<'_, PyAny>,
    parallel: bool,
) -> PyResult<(PyObject, PyObject, PyObject)> {
    // Slice boards, dirs, evs using NumPy field selection and advanced indexing
    let boards_arr = steps.get_item("board")?.get_item(&idxs)?;
    let dirs_arr = steps.get_item("move")?.get_item(&idxs)?;
    let evs_arr = steps.get_item("ev_values")?.get_item(&idxs)?;

    // Convert boards to exponents via the fast path
    let exps = exps_from_boards(py, boards_arr, parallel)?;
    Ok((exps, dirs_arr.into_py(py), evs_arr.into_py(py)))
}
