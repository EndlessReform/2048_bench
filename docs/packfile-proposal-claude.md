# A2Run2 Packfile Format Proposal

## Overview

A readonly packfile abstraction for efficiently accessing large collections of a2run2 files. Addresses filesystem overhead when working with 5k-100k individual run files (~60KB each) by consolidating them into a single binary packfile with O(1) random access.

## Key Requirements

### Performance Goals
- **Eliminate filesystem overhead**: Single file I/O instead of thousands of small file operations
- **Fast random access**: O(1) lookup by run index, sub-millisecond access times
- **Bulk operations**: Efficient sequential iteration for training data preparation
- **Parallel deserialization**: Internal threadpool for concurrent run parsing
- **Memory efficiency**: Memory-mapped I/O with lazy deserialization

### Use Cases
1. **Summary statistics**: Fast computation over entire dataset (max scores, run lengths, etc.)
2. **Training data**: DataLoader-style batching with random sampling
3. **Run lookup**: Direct access by run ID/index for debugging/analysis  
4. **Export compatibility**: Bulk conversion to JSONL/Parquet for external tools
5. **Dataset exploration**: Quick filtering and subsetting operations

## File Format Specification

### Binary Layout
```
[Header: 32 bytes]
[Run Index Table: run_count * 16 bytes]
[Run Data: concatenated postcard-serialized RunV2 structs]
```

### Header Structure (32 bytes)
```rust
struct PackfileHeader {
    magic: [u8; 8],        // b"A2PACK01"
    version: u32,          // Format version (1)
    run_count: u32,        // Number of runs in packfile
    total_steps: u64,      // Sum of all run step counts
    max_score: u64,        // Maximum score across all runs
    max_run_length: u32,   // Maximum run length (steps)
    _reserved: [u8; 4],    // Future expansion
}
```

### Run Index Entry (16 bytes)
```rust
struct RunIndexEntry {
    offset: u64,           // Byte offset to run data
    length: u32,           // Byte length of serialized run
    step_count: u32,       // Cached step count for quick filtering
}
```

## Core API Design

### Packfile Reader
```rust
pub struct A2PackfileReader {
    mmap: Mmap,
    header: PackfileHeader,
    index: Vec<RunIndexEntry>,
    thread_pool: rayon::ThreadPool,
}

impl A2PackfileReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, PackfileError>;
    
    // Metadata access
    pub fn run_count(&self) -> u32;
    pub fn total_steps(&self) -> u64;
    pub fn max_score(&self) -> u64;
    pub fn max_run_length(&self) -> u32;
    
    // Single run access
    pub fn get_run(&self, index: usize) -> Result<RunV2, PackfileError>;
    pub fn get_run_bytes(&self, index: usize) -> Result<&[u8], PackfileError>;
    
    // Batch access
    pub fn get_runs(&self, indices: &[usize]) -> Result<Vec<RunV2>, PackfileError>;
    pub fn get_runs_parallel(&self, indices: &[usize]) -> Result<Vec<RunV2>, PackfileError>;
    
    // Iteration
    pub fn iter(&self) -> PackfileIterator;
    pub fn iter_parallel(&self) -> PackfileParallelIterator;
    
    // DataLoader-style sampling
    pub fn random_batch(&self, batch_size: usize, rng: &mut impl Rng) -> Result<Vec<RunV2>, PackfileError>;
    pub fn random_batch_indices(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<usize>;
    
    // Filtering
    pub fn filter_by_score(&self, min_score: u64, max_score: u64) -> Vec<usize>;
    pub fn filter_by_length(&self, min_steps: u32, max_steps: u32) -> Vec<usize>;
}
```

### Iterator Types
```rust
pub struct PackfileIterator<'a> {
    reader: &'a A2PackfileReader,
    current: usize,
}

pub struct PackfileParallelIterator {
    reader: Arc<A2PackfileReader>,
    chunk_size: usize,
}

pub struct PackfileBatchIterator<'a> {
    reader: &'a A2PackfileReader,
    batch_size: usize,
    current: usize,
    shuffle: bool,
    rng: Option<Box<dyn RngCore>>,
}
```

## Binary Tool: a2pack

### CLI Interface
```bash
# Create packfile from directory
a2pack create --input ./runs_directory --output dataset.a2pack

# Validate packfile
a2pack validate dataset.a2pack

# Extract specific runs
a2pack extract --packfile dataset.a2pack --indices 0,100,200 --output extracted/

# Convert to JSONL
a2pack to-jsonl --packfile dataset.a2pack --output dataset.jsonl

# Summary statistics
a2pack stats dataset.a2pack
```

### Implementation
```rust
// src/bin/a2pack.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Create { input: PathBuf, output: PathBuf },
    Validate { packfile: PathBuf },
    Extract { packfile: PathBuf, indices: String, output: PathBuf },
    ToJsonl { packfile: PathBuf, output: PathBuf },
    Stats { packfile: PathBuf },
}
```

## PyO3 Integration

### Python API
```python
from ai_2048 import PackfileReader, RunV2

# Open packfile
reader = PackfileReader("dataset.a2pack")

# Basic properties
print(f"Runs: {reader.run_count}")
print(f"Max score: {reader.max_score}")
print(f"Total steps: {reader.total_steps}")

# Single run access
run = reader.get_run(0)  # Returns RunV2
raw_bytes = reader.get_run_bytes(0)  # Returns bytes

# Batch access
runs = reader.get_runs([0, 1, 2])  # List[RunV2]
runs_parallel = reader.get_runs_parallel(range(1000))  # Parallel deserialization

# Iteration
for run in reader:
    process(run)

# DataLoader-style batching
for batch in reader.batches(batch_size=32, shuffle=True, seed=42):
    train_step(batch)  # batch is List[RunV2]

# Random sampling
batch = reader.random_batch(64)  # Returns List[RunV2]

# Filtering
high_score_indices = reader.filter_by_score(min_score=10000)
long_run_indices = reader.filter_by_length(min_steps=1000)

# Export compatibility
reader.to_jsonl("output.jsonl", parallel=True)
reader.to_parquet("output.parquet")  # If arrow integration added
```

### PyO3 Implementation
```rust
#[pyclass(module = "ai_2048", name = "PackfileReader")]
pub struct PyPackfileReader {
    inner: A2PackfileReader,
}

#[pymethods]
impl PyPackfileReader {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self>;
    
    #[getter]
    fn run_count(&self) -> u32;
    #[getter] 
    fn max_score(&self) -> u64;
    #[getter]
    fn total_steps(&self) -> u64;
    #[getter]
    fn max_run_length(&self) -> u32;
    
    fn get_run(&self, index: usize) -> PyResult<PyRunV2>;
    fn get_run_bytes(&self, index: usize) -> PyResult<Vec<u8>>;
    fn get_runs(&self, indices: Vec<usize>) -> PyResult<Vec<PyRunV2>>;
    fn get_runs_parallel(&self, indices: Vec<usize>) -> PyResult<Vec<PyRunV2>>;
    
    fn random_batch(&self, batch_size: usize, seed: Option<u64>) -> PyResult<Vec<PyRunV2>>;
    fn random_batch_indices(&self, batch_size: usize, seed: Option<u64>) -> Vec<usize>;
    
    fn filter_by_score(&self, min_score: Option<u64>, max_score: Option<u64>) -> Vec<usize>;
    fn filter_by_length(&self, min_steps: Option<u32>, max_steps: Option<u32>) -> Vec<usize>;
    
    fn to_jsonl(&self, path: PathBuf, parallel: bool) -> PyResult<()>;
    
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyPackfileIterator>;
    fn __len__(&self) -> usize;
    
    fn batches(&self, batch_size: usize, shuffle: bool, seed: Option<u64>) -> PyPackfileBatchIterator;
}
```

## Implementation Phases

### Phase 1: Core Packfile Format
- [ ] Define binary format and header structure
- [ ] Implement `A2PackfileReader` with memory mapping
- [ ] Basic indexing and single-run access
- [ ] Unit tests for format validation

### Phase 2: Batch Operations & CLI
- [ ] Parallel deserialization using rayon
- [ ] `a2pack` binary tool with create/validate commands
- [ ] Iterator implementations
- [ ] Filtering and sampling methods

### Phase 3: PyO3 Integration
- [ ] `PyPackfileReader` wrapper class
- [ ] Python iterator protocol implementation
- [ ] Batch access methods and DataLoader-style API
- [ ] Export functions (JSONL, optionally Parquet)

### Phase 4: Optimization & Features
- [ ] Optional compression (LZ4/Zstd) for packfile data section
- [ ] Async I/O variants for very large packfiles
- [ ] Memory usage optimization for huge datasets
- [ ] Advanced filtering (by timestamp, engine type, etc.)

## Error Handling

```rust
#[derive(thiserror::Error, Debug)]
pub enum PackfileError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid packfile format: {0}")]
    InvalidFormat(String),
    #[error("Run index out of bounds: {index}, max: {max}")]
    IndexOutOfBounds { index: usize, max: usize },
    #[error("Deserialization error: {0}")]
    Deserialization(#[from] postcard::Error),
    #[error("Memory mapping error: {0}")]
    Mmap(String),
}
```

## Performance Characteristics

### Expected Improvements
- **File open time**: ~1ms vs ~500ms-1s for 5k individual files
- **Random access**: ~10μs vs ~200μs per run
- **Sequential iteration**: ~50% faster due to better cache locality
- **Memory usage**: Constant overhead vs O(n) file handles
- **Parallel processing**: Near-linear scaling with thread count

### Memory Usage
- Base overhead: ~32 bytes + (run_count * 16) bytes for index
- Per-run cost: ~0 bytes (memory-mapped, lazy deserialization)
- Example: 100k runs = ~1.6MB index overhead

## Compatibility & Migration

### Backward Compatibility
- Original a2run2 files remain supported via existing API
- Packfiles can be created from existing directory structures
- No changes to core `RunV2` format or postcard serialization

### Migration Path
1. Use `a2pack create` to convert existing directories
2. Update Python scripts to use `PackfileReader` instead of individual file loading  
3. Gradual adoption - both APIs can coexist

This design provides the foundation for efficient large-scale dataset processing while maintaining the existing API contracts and serialization format.