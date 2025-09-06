# 2048 RL Dataset Architecture: A KISS Design

## Executive Summary

We need to serve 3,072-4,096 random steps per batch to PyTorch, thousands of times per epoch, without stalling the GPU. The solution is embarrassingly simple: use a single NumPy structured array loaded into RAM.

**The entire architecture:**
1. Store steps as a structured NumPy array (`.npy` file)
2. Store run metadata in SQLite
3. Load the array into RAM at startup
4. Serve batches via direct array indexing

That's it. No sharding. No memory mapping. No column stores. No "block-shuffle" algorithms.

## Problem Statement

### Access Pattern
- **Primary operation:** Fetch 3,072-4,096 random rows from a pool of 1M-1B steps
- **Frequency:** ~1000 batches per epoch × many epochs
- **Shuffle behavior:** PyTorch generates one random permutation per epoch, then iterates through it
- **Latency requirement:** Must keep GPU fed (batch fetch must be < GPU forward pass time)

### Data Scale
- **Dataset size:** 1M-1B steps (32MB-32GB)
- **Available RAM:** 64GB DDR5
- **Growth rate:** Incremental updates hourly/daily with 100K-10M new steps

### Why Existing Solutions Fail
- **SQLite:** 3,072 B-tree lookups + Python object creation = death by overhead
- **Column stores (Parquet/Arrow):** Multiple scattered reads for random rows
- **Memory-mapped files:** 4KB page size × 32-byte records = 128x read amplification
- **Current custom format:** Unknown implementation issues causing RAM thrashing

## Proposed Architecture

### Core Design: Single Structured Array

```python
# The entire data format
step_dtype = np.dtype([
    ('board', np.uint64),        # 8 bytes: packed 4×4 board
    ('move', np.uint8),           # 1 byte: 0-3 for direction
    ('ev_legal', np.uint8),       # 1 byte: bitmask of legal moves
    ('ev_values', np.float32, 4), # 16 bytes: EVs for 4 moves
    ('run_id', np.uint32),        # 4 bytes: which run
    ('step_index', np.uint16),    # 2 bytes: position in run
])  # Total: 32 bytes aligned
```

**Why structured array beats column arrays:**
- **Single gather operation** vs 6+ separate gathers
- **Cache-friendly:** One row = one contiguous memory region
- **NumPy optimized:** `array[indices]` uses SIMD gather instructions
- **No reassembly:** Returns ready-to-use records

### File Layout

```
dataset/
├── steps.npy           # THE data: structured array of all steps
├── metadata.db         # Run metadata and indices (SQLite)
└── manifest.json       # Version, checksums, stats (optional)
```

**Why this layout:**
- **Two files cover 99% of operations** (steps.npy + metadata.db)
- **Clear separation:** Bulk data (npy) vs queryable metadata (SQLite)
- **Standard formats:** Tools exist everywhere

### Runtime Architecture

```python
class Dataset:
    def __init__(self, dataset_dir: str):
        # Load ONCE at startup
        self.steps = np.load(f"{dataset_dir}/steps.npy")  # ~2 sec for 5GB
        self.metadata = sqlite3.connect(f"{dataset_dir}/metadata.db")

        # Pre-compute for filtering (optional)
        self.run_ranges = self._build_run_ranges()

    def __len__(self):
        return len(self.steps)

    def get_batch(self, indices: List[int]) -> np.ndarray:
        # This is the entire hot path
        return self.steps[indices]  # <1ms for 4096 rows

    def filter_by_score(self, min_score: int) -> 'Dataset':
        # Returns view with filtered indices
        run_ids = self.metadata.execute(
            "SELECT id FROM runs WHERE max_score >= ?",
            (min_score,)
        ).fetchall()
        # ... compute valid step indices ...
        return FilteredDataset(self, valid_indices)
```

### Build Pipeline

```python
def build_dataset(runs_dir: Path, output_dir: Path):
    """Convert .a2run2 files to dataset format"""

    # 1. Load all runs (parallel)
    runs = load_all_runs(runs_dir)  # existing code

    # 2. Flatten to steps array
    steps = []
    run_metadata = []

    for run_id, run in enumerate(runs):
        for step_idx, (board, move) in enumerate(zip(run.boards, run.moves)):
            steps.append((
                board,
                move,
                compute_ev_legal(run, step_idx),
                compute_ev_values(run, step_idx),
                run_id,
                step_idx
            ))
        run_metadata.append({
            'id': run_id,
            'num_steps': len(run.moves),
            'max_score': run.max_score,
            'highest_tile': run.highest_tile,
            'engine': run.engine
        })

    # 3. Save as structured array
    steps_array = np.array(steps, dtype=step_dtype)
    np.save(f"{output_dir}/steps.npy", steps_array)

    # 4. Save metadata
    create_metadata_db(f"{output_dir}/metadata.db", run_metadata)
```

### Incremental Updates

```python
def append_runs(dataset_dir: Path, new_runs_dir: Path):
    """Append new runs to existing dataset"""

    # Load existing
    old_steps = np.load(f"{dataset_dir}/steps.npy")
    old_run_count = get_max_run_id(f"{dataset_dir}/metadata.db")

    # Process new runs
    new_runs = load_all_runs(new_runs_dir)
    new_steps = flatten_runs(new_runs, run_id_offset=old_run_count+1)

    # Concatenate and save atomically
    combined = np.concatenate([old_steps, new_steps])
    np.save(f"{dataset_dir}/steps.npy.tmp", combined)
    os.rename(f"{dataset_dir}/steps.npy.tmp", f"{dataset_dir}/steps.npy")

    # Update metadata
    append_to_metadata_db(f"{dataset_dir}/metadata.db", new_runs)
```

## Performance Analysis

### Batch Fetch Performance

For 4,096 random indices:

**Our approach (structured array in RAM):**
- Single `np.take(array, indices)` call
- ~0.5ms for 4096 rows
- CPU: ~130KB of actual data movement
- Zero system calls

**Alternative approaches (why they fail):**

| Approach | Time | Why It's Slow |
|----------|------|--------------|
| SQLite in RAM | ~50ms | 4096 B-tree traversals + Python objects |
| Column arrays | ~5ms | 6 separate gathers + reassembly |
| Memory-mapped | ~10ms | Page faults (even if cached) |
| Parquet | ~100ms | Decompression + column reassembly |

### Memory Footprint

- **1B steps:** 32GB (fits in 64GB RAM with room for PyTorch)
- **100M steps:** 3.2GB (trivial)
- **Startup time:** ~6 seconds for 1B steps (5GB/s SSD read)
- **Working set:** Equal to dataset size (everything in RAM)

### Scaling Limits

This design works until:
- Dataset exceeds ~40GB (leaving 24GB for PyTorch/OS)
- At that point, implement sharding (see Future Work)

## Design Justifications

### Why Not Memory Mapping?

Memory mapping seems attractive but fails for this access pattern:
- **Page amplification:** Reading 32 bytes triggers 4KB page read (128x amplification)
- **Random access pattern:** Cache thrashing on every batch
- **No real benefit:** Dataset fits in RAM anyway

### Why Not Column Storage?

Column stores optimize for the wrong thing:
- **Built for scans:** Reading all rows, few columns
- **We need:** Random rows, all columns
- **Result:** Multiple scattered reads per batch

### Why Not SQLite for Steps?

Databases add unnecessary overhead:
- **B-tree traversal:** O(log n) per row vs O(1) array access
- **Python marshaling:** Creating 4096 Python objects per batch
- **SQL parsing:** Overhead on every query
- **Page structure:** Even in RAM, still page-based access

### Why Structured Arrays?

Structured arrays are optimal for this exact pattern:
- **Single operation:** One gather instruction
- **Cache friendly:** Row data is contiguous
- **NumPy optimized:** Uses SIMD/vectorization
- **Zero-copy compatible:** Can wrap in PyTorch tensors

## Rust Implementation

### Required Crates

```toml
[dependencies]
# Core
numpy = "0.21"           # NPY file format support
ndarray = "0.15"         # N-dimensional arrays (compatible with numpy crate)
rusqlite = "0.31"        # SQLite for metadata
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"       # For manifest.json

# Utils
rayon = "1.7"            # Parallel processing of runs
bytemuck = "1.14"        # Zero-copy casting for structured data
memmap2 = "0.9"          # IF you need memory mapping later (not recommended)
anyhow = "1.0"           # Error handling

# CLI
clap = { version = "4.0", features = ["derive"] }

# Optional (you probably already have these)
crc32c = "0.6"           # Checksums if needed
```

### Structured Array in Rust

```rust
use bytemuck::{Pod, Zeroable};
use numpy::{PyArray1, PyArrayMethods};

// Step struct matching NumPy dtype EXACTLY
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Step {
    board: u64,           // 8 bytes
    move_dir: u8,         // 1 byte
    ev_legal: u8,         // 1 byte
    _pad1: [u8; 2],       // 2 bytes padding for alignment
    ev_values: [f32; 4],  // 16 bytes
    run_id: u32,          // 4 bytes
    step_index: u16,      // 2 bytes
    _pad2: [u8; 2],       // 2 bytes padding to reach 32 bytes
}

impl Step {
    fn to_bytes(&self) -> &[u8; 32] {
        bytemuck::bytes_of(self).try_into().unwrap()
    }
}
```

### Writing NPY Files

```rust
use ndarray::{Array1, ArrayView1};
use numpy::npz::{NpzWriter, WriteOptions};
use std::fs::File;
use std::io::BufWriter;

fn write_steps_npy(steps: Vec<Step>, path: &Path) -> Result<()> {
    // Convert to bytes for NPY format
    let bytes: Vec<u8> = steps.iter()
        .flat_map(|s| s.to_bytes().iter())
        .copied()
        .collect();

    // Write as NPY with proper header
    // Note: numpy crate expects you to specify dtype in Python-compatible format
    let mut file = BufWriter::new(File::create(path)?);

    // Alternative: Use ndarray + numpy crate
    let array = Array1::from_vec(steps);
    numpy::write_npy(&mut file, &array)?;

    Ok(())
}

// Alternative using raw NPY format (more control)
fn write_npy_raw(steps: &[Step], path: &Path) -> Result<()> {
    use std::io::Write;

    let mut file = BufWriter::new(File::create(path)?);

    // NPY format header (v1.0)
    let header = format!(
        "{{'descr': [('board','<u8'),('move','u1'),('ev_legal','u1'),\
         ('ev_values','<f4',(4,)),('run_id','<u4'),('step_index','<u2')], \
         'fortran_order': False, 'shape': ({},)}}",
        steps.len()
    );

    // Write magic and header
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1u8, 0u8])?;  // Version 1.0

    // Header length (little endian u16)
    let header_len = header.len() + 1;  // +1 for newline
    let padding = (16 - (10 + header_len) % 16) % 16;
    let total_header_len = header_len + padding;
    file.write_all(&(total_header_len as u16).to_le_bytes())?;

    // Header string + padding
    file.write_all(header.as_bytes())?;
    file.write_all(b"\n")?;
    file.write_all(&vec![b' '; padding - 1])?;
    file.write_all(b"\n")?;

    // Write data as raw bytes
    let bytes = bytemuck::cast_slice::<Step, u8>(steps);
    file.write_all(bytes)?;

    Ok(())
}
```

### SQLite Metadata

```rust
use rusqlite::{Connection, params};

fn create_metadata_db(path: &Path, runs: &[RunMetadata]) -> Result<()> {
    let conn = Connection::open(path)?;

    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            first_step_idx INTEGER NOT NULL,
            num_steps INTEGER NOT NULL,
            max_score INTEGER NOT NULL,
            highest_tile INTEGER NOT NULL,
            engine TEXT,
            start_time INTEGER,
            elapsed_s REAL
        );

        CREATE INDEX IF NOT EXISTS idx_score ON runs(max_score);
        CREATE INDEX IF NOT EXISTS idx_steps ON runs(num_steps);
    ")?;

    let mut stmt = conn.prepare("
        INSERT INTO runs (id, first_step_idx, num_steps, max_score,
                         highest_tile, engine, start_time, elapsed_s)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
    ")?;

    let mut first_step_idx = 0;
    for (id, run) in runs.iter().enumerate() {
        stmt.execute(params![
            id as i64,
            first_step_idx,
            run.num_steps,
            run.max_score,
            run.highest_tile,
            run.engine,
            run.start_time,
            run.elapsed_s,
        ])?;
        first_step_idx += run.num_steps as i64;
    }

    Ok(())
}
```

### Complete Builder

```rust
use rayon::prelude::*;

pub struct DatasetBuilder {
    steps: Vec<Step>,
    runs: Vec<RunMetadata>,
}

impl DatasetBuilder {
    pub fn from_runs_dir(dir: &Path) -> Result<Self> {
        // Parallel load all .a2run2 files
        let run_files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension() == Some("a2run2"))
            .map(|e| e.path())
            .collect();

        let runs: Vec<Run> = run_files
            .par_iter()
            .filter_map(|path| parse_run_file(path).ok())
            .collect();

        // Flatten to steps
        let mut steps = Vec::new();
        let mut run_metadata = Vec::new();

        for (run_id, run) in runs.iter().enumerate() {
            let first_step = steps.len();

            for (step_idx, (&board, &move_dir)) in
                run.boards.iter().zip(&run.moves).enumerate()
            {
                steps.push(Step {
                    board,
                    move_dir,
                    ev_legal: compute_ev_legal(&run, step_idx),
                    ev_values: compute_ev_values(&run, step_idx),
                    run_id: run_id as u32,
                    step_index: step_idx as u16,
                    _pad1: [0; 2],
                    _pad2: [0; 2],
                });
            }

            run_metadata.push(RunMetadata {
                num_steps: run.moves.len(),
                max_score: run.max_score,
                // ... etc
            });
        }

        Ok(DatasetBuilder { steps, runs: run_metadata })
    }

    pub fn write_to(&self, output_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;

        // Write steps.npy
        write_steps_npy(&self.steps, &output_dir.join("steps.npy"))?;

        // Write metadata.db
        create_metadata_db(&output_dir.join("metadata.db"), &self.runs)?;

        // Optional: Write manifest.json
        let manifest = json!({
            "version": 1,
            "num_steps": self.steps.len(),
            "num_runs": self.runs.len(),
            "created": chrono::Utc::now().to_rfc3339(),
        });

        std::fs::write(
            output_dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest)?
        )?;

        Ok(())
    }
}
```

## Implementation Checklist

### Phase 1: Core (2-3 hours)
- [ ] Define Step struct with `#[repr(C)]` and proper alignment
- [ ] Implement NPY writer using `numpy` or raw format
- [ ] Write `DatasetBuilder::from_runs_dir()`
- [ ] Test NPY files are readable from Python

### Phase 2: Metadata (1-2 hours)
- [ ] Create SQLite schema using `rusqlite`
- [ ] Implement `create_metadata_db()`
- [ ] Add filtering queries

### Phase 3: Operations (2-3 hours)
- [ ] Implement incremental append
- [ ] Add CRC checks with `crc32c` crate
- [ ] Create CLI with `clap`

### Phase 4: Python Integration (1 hour)
- [ ] Verify NumPy dtype matches Rust struct exactly
- [ ] Custom PyTorch Dataset class
- [ ] Benchmark against current implementation
- [ ] Documentation

## Future Work (Only If Needed)

### Sharding for >40GB Datasets

If datasets grow beyond RAM:
```python
# Shard structure (NOT NEEDED NOW)
dataset/
├── manifest.json
├── metadata.db
├── shards/
│   ├── steps_00000.npy  # 5GB each
│   ├── steps_00001.npy
│   └── ...
```

Implementation sketch:
- Load 1-2 shards in RAM at a time
- Simple LRU cache
- Still structured arrays per shard

### Distributed Training

If multi-node needed:
- Each node gets shard assignments
- Deterministic shuffle with shared seed
- No modification to core design

## Migration Plan

1. **Week 1:** Implement core Dataset class
2. **Week 2:** Parallel testing with existing pipeline
3. **Week 3:** Full cutover
4. **Deprecate:** Remove `.dat` and `.a2pack` code

## Non-Goals (Explicitly Not Doing)

- **Compression:** Costs CPU, saves space we don't need
- **Columnar storage:** Wrong access pattern
- **Memory mapping:** Causes page amplification
- **Complex sampling algorithms:** Shuffle works fine
- **Transactional updates:** Rebuild is fast enough
- **Distributed filesystem support:** Single node is sufficient
- **Version control:** Just timestamp directories

## Success Metrics

- **Batch latency:** <1ms for 4096 samples
- **GPU utilization:** >95% (never waiting for data)
- **Code complexity:** <500 lines total
- **Startup time:** <10 seconds for 1B steps

## Summary

This design is intentionally boring. It uses NumPy arrays exactly as intended, for exactly what they're good at: fast random access to structured data in RAM.

No clever algorithms. No distributed systems. No compression. Just an array in memory and the fastest possible batch fetching.

**Total implementation: ~300 lines of Python/Rust. Total complexity: none.**
