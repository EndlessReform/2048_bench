# PyO3 Implementation Proposal for ai-2048

**Author:** Claude  
**Date:** 2025-08-23  
**Target:** Python bindings for the ai-2048 Rust crate

## Executive Summary

This document proposes a comprehensive PyO3 implementation to expose the core ai-2048 functionality to Python users. The design prioritizes ergonomic Python APIs while preserving the performance characteristics and deterministic behavior of the underlying Rust implementation.

## Architecture Overview

### Core Design Principles

1. **Zero-Copy Where Possible**: Leverage Rust's `Copy` types (Board, Move) for minimal overhead
2. **Ergonomic Python APIs**: Provide both raw access and convenience methods following Python conventions
3. **Preserve Determinism**: Maintain exact behavioral compatibility with Rust implementation
4. **Performance First**: Expose parallel algorithms and optional GIL release for long operations
5. **Comprehensive Coverage**: Surface all key abstractions without leaking internal complexity

### Module Structure

```
src/
├── python/
│   ├── mod.rs          # PyO3 module definition and initialization
│   ├── board.rs        # PyBoard wrapper and methods
│   ├── expectimax.rs   # PyExpectimax and PyExpectimaxConfig
│   ├── serialization.rs # PyRunV2, PyStepV2, and I/O operations
│   └── types.rs        # PyMove, PyBranchEval, PySearchStats
```

## Public Python API Specification

### 1. Board Management (`ai_2048.Board`)

```python
class Board:
    # Construction
    @staticmethod
    def empty() -> Board
    @staticmethod 
    def from_raw(raw: int) -> Board
    
    # Game mechanics
    def shift(self, direction: Move) -> Board
    def make_move(self, direction: Move, *, rng_seed: Optional[int] = None) -> Board
    def with_random_tile(self, *, rng_seed: Optional[int] = None) -> Board
    
    # State inspection
    def score(self) -> int
    def is_game_over(self) -> bool
    def highest_tile(self) -> int
    def count_empty(self) -> int
    def tile_value(self, index: int) -> int
    def to_list(self) -> List[int]  # 16 tile values
    
    # Raw access (performance-critical code)
    def raw(self) -> int
    
    # Python protocols
    def __str__(self) -> str      # Visual grid display
    def __repr__(self) -> str     # Board(0x...)
    def __iter__(self) -> Iterator[int]  # Tile exponents
    def __eq__(self, other: Board) -> bool
    def __hash__(self) -> int
```

### 2. AI Policy (`ai_2048.Expectimax`)

```python
class ExpectimaxConfig:
    def __init__(self, 
                 prob_cutoff: float = 1e-4,
                 depth_cap: Optional[int] = None,
                 cache_enabled: bool = True,
                 parallel_thresholds: Optional[ParallelThresholds] = None)

class Expectimax:
    def __init__(self, config: Optional[ExpectimaxConfig] = None)
    
    @staticmethod
    def parallel(config: Optional[ExpectimaxConfig] = None) -> Expectimax
    
    # Core decision making
    def best_move(self, board: Board) -> Optional[Move]
    def branch_evals(self, board: Board) -> List[BranchEval]
    def state_value(self, board: Board) -> float
    
    # Performance monitoring
    def last_stats(self) -> SearchStats
    def reset_stats(self) -> None

class BranchEval:
    direction: Move
    expected_value: float
    is_legal: bool

class SearchStats:
    nodes_visited: int
    peak_nodes: int
```

### 3. Serialization (`ai_2048.RunV2`)

```python
class StepV2:
    # Convenience properties (recommended usage)
    @property
    def pre_board(self) -> Board
    @property 
    def chosen(self) -> Move
    @property
    def branches(self) -> Optional[List[BranchEval]]
    
    # Raw access (performance-critical code)
    @property
    def pre_board_raw(self) -> int

class RunV2:
    def __init__(self, meta: Meta, steps: List[StepV2], final_board: Board)
    
    # File I/O
    @staticmethod
    def load_from_file(path: str) -> RunV2
    def save_to_file(self, path: str) -> None
    
    # Serialization
    @staticmethod
    def from_bytes(data: bytes) -> RunV2
    def to_bytes(self) -> bytes
    
    # V1 compatibility
    @staticmethod
    def from_v1_file(path: str) -> RunV2
    
    # Properties
    @property
    def meta(self) -> Meta
    @property 
    def steps(self) -> List[StepV2]
    @property
    def final_board(self) -> Board
    @property
    def final_board_raw(self) -> int

class Meta:
    steps: int
    start_unix_s: int
    elapsed_s: float
    max_score: int
    highest_tile: int
    engine_str: Optional[str]
```

### 4. Utility Types

```python
class Move(Enum):
    UP = "up"
    DOWN = "down" 
    LEFT = "left"
    RIGHT = "right"

# Module-level constants
ai_2048.MOVE_UP: Move
ai_2048.MOVE_DOWN: Move
ai_2048.MOVE_LEFT: Move
ai_2048.MOVE_RIGHT: Move
```

## Usage Examples

### Basic Game Loop

```python
import ai_2048

# Initialize game
board = ai_2048.Board.empty()
board = board.with_random_tile().with_random_tile()
ai = ai_2048.Expectimax()

# Game loop
while not board.is_game_over():
    print(board)
    move = ai.best_move(board)
    if move is None:
        break
    board = board.make_move(move)
    
print(f"Final score: {board.score()}")
print(f"AI stats: {ai.last_stats()}")
```

### Advanced Analysis

```python
# Branch analysis
branches = ai.branch_evals(board)
for branch in branches:
    if branch.is_legal:
        print(f"{branch.direction}: EV = {branch.expected_value:.2f}")

# Deterministic replay with seeded RNG
board = ai_2048.Board.empty()
board = board.with_random_tile(rng_seed=42)
board = board.with_random_tile(rng_seed=43)

# Performance comparison
import time
single_ai = ai_2048.Expectimax()
parallel_ai = ai_2048.Expectimax.parallel()

start = time.time()
move1 = single_ai.best_move(complex_board)
single_time = time.time() - start

start = time.time()  
move2 = parallel_ai.best_move(complex_board)
parallel_time = time.time() - start

print(f"Speedup: {single_time / parallel_time:.1f}x")
```

### Run Recording and Analysis

```python
# Record a game
steps = []
board = ai_2048.Board.empty().with_random_tile().with_random_tile()
ai = ai_2048.Expectimax()

start_time = time.time()
while not board.is_game_over():
    branches = ai.branch_evals(board)
    move = ai.best_move(board) 
    if move is None:
        break
        
    step = ai_2048.StepV2(
        pre_board=board,
        chosen=move,
        branches=branches
    )
    steps.append(step)
    board = board.make_move(move)

# Save run
meta = ai_2048.Meta(
    steps=len(steps),
    start_unix_s=int(start_time),
    elapsed_s=time.time() - start_time,
    max_score=board.score(),
    highest_tile=board.highest_tile(),
    engine_str="expectimax-python"
)

run = ai_2048.RunV2(meta, steps, board)
run.save_to_file("analysis.a2run2")

# Load and analyze
loaded_run = ai_2048.RunV2.load_from_file("analysis.a2run2")
print(f"Game took {loaded_run.meta.elapsed_s:.1f}s")
print(f"Final score: {loaded_run.final_board.score()}")

for i, step in enumerate(loaded_run.steps):
    print(f"Step {i}: chose {step.chosen} from:")
    print(step.pre_board)
```

## Implementation Strategy

### 1. Cargo Configuration

```toml
# Add to Cargo.toml
[features]
python = ["dep:pyo3"]

[dependencies]
pyo3 = { version = "0.20", optional = true }

# New crate type for Python extension
[lib]
crate-type = ["rlib", "cdylib"]
```

### 2. Key Implementation Decisions

**Error Handling:**
- Convert Rust errors to appropriate Python exceptions
- Use `PyResult<T>` consistently
- Provide meaningful error messages for invalid board states, file I/O failures

**Memory Management:**
- Board uses `Copy` semantics - no ownership concerns
- RunV2/StepV2 use reference counting for large data structures
- Avoid unnecessary cloning in hot paths

**Thread Safety:**
- Release GIL during long-running expectimax searches
- Parallel expectimax can utilize multiple cores without GIL contention
- Cache structures use thread-safe collections from Rust side

**Initialization:**
- Call `engine::new()` automatically in Python module init
- Lazy initialization of heuristic tables preserved
- No explicit setup required from Python side

### 3. Performance Considerations

**Hot Paths:**
- Direct delegation to Rust implementations
- Minimal Python object allocation in game loops
- Raw board access bypasses object creation

**Cold Paths:**
- File I/O uses efficient postcard serialization
- Lazy property evaluation for convenience methods
- String formatting only when requested

**Memory Usage:**
- Board: 8 bytes (u64)
- Move: 1 byte (enum)
- BranchEval: ~16 bytes
- SearchStats: ~16 bytes

## Testing Strategy

### 1. Behavioral Parity
- Comprehensive comparison tests between Python and Rust APIs
- Deterministic game replay verification
- Serialization round-trip testing

### 2. Performance Benchmarks
- Python vs Rust expectimax performance comparison
- Memory usage profiling
- GIL release effectiveness measurement

### 3. Integration Testing
- End-to-end game simulation
- File format compatibility with existing tools
- Error handling and edge cases

## Migration Path

### Phase 1: Core Functionality
- Board manipulation and display
- Basic expectimax integration
- Move generation and validation

### Phase 2: Advanced Features
- Parallel expectimax support
- Configuration and performance tuning
- Statistics and monitoring

### Phase 3: Serialization & Analysis
- RunV2 serialization format
- V1 compatibility layer
- Analysis and visualization helpers

### Phase 4: Documentation & Polish
- Comprehensive docstrings
- Usage examples and tutorials
- Performance optimization

## Risk Assessment

**Low Risk:**
- Core Board/Move abstractions (direct Rust delegation)
- Serialization (existing battle-tested implementation)

**Medium Risk:**
- GIL management in parallel expectimax
- Error handling and exception translation
- Memory management for large runs

**High Risk:**
- Thread safety in concurrent usage
- Performance degradation vs pure Rust
- API design decisions (hard to change post-release)

## Success Metrics

1. **Performance**: <10% overhead vs pure Rust for single-threaded workloads
2. **Usability**: Complete game implementation in <50 lines of Python
3. **Compatibility**: 100% serialization format compatibility with existing tools
4. **Adoption**: Enable new use cases (Jupyter analysis, ML training data generation)

## Conclusion

This PyO3 implementation would provide comprehensive Python access to ai-2048's core functionality while preserving its performance characteristics. The design balances ergonomic Python APIs with efficient Rust delegation, enabling new use cases in data analysis, machine learning, and interactive development environments.

The modular implementation strategy allows for incremental development and testing, with clear success metrics and risk mitigation strategies. The resulting Python package would serve as a powerful tool for 2048 research and analysis while maintaining the deterministic behavior crucial for reproducible experiments.