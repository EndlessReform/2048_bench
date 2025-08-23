# PyO3 Python Bindings for ai-2048

This directory contains Python bindings for the ai-2048 Rust library, providing high-performance 2048 game simulation and AI capabilities in Python.

## Current Implementation Status

**Completed:**
- Cargo workspace setup with separate library crate (`ai-2048`) and Python bindings crate (`py-ai-2048`)
- PyO3 dependencies and feature flags configured
- `Board` struct with all basic methods (construction, movement, scoring, inspection)
- `Move` enum with all directions (UP, DOWN, LEFT, RIGHT)
- `Rng` helper class for deterministic random number generation
- Full Python protocol support (`__str__`, `__repr__`, `__eq__`, `__hash__`, `__iter__`)

## Building and Testing

### Prerequisites

- Rust toolchain (cargo)
- Python 3.10+
- UV package manager (recommended)

### Building the Extension

**Build and install in development mode:**
```bash
uv run maturin develop
```

### Testing Basic Functionality

```python
import ai_2048
from ai_2048 import Board, Move, Rng

# Create empty board
board = Board.empty()

# Add random tiles deterministically
rng = Rng(42)
board = board.with_random_tile(rng=rng).with_random_tile(rng=rng)

# Inspect board
print(f"Score: {board.score()}")
print(f"Empty cells: {board.count_empty()}")
print(f"Highest tile: {board.highest_tile()}")
print(f"Game over: {board.is_game_over()}")

# Make moves
left_board = board.shift(Move.LEFT)
full_move = board.make_move(Move.LEFT, rng=rng)  # includes random tile insertion

# Display board
print(board)  # Pretty-printed grid
```

## API Reference

### Board Class

**Construction:**
- `Board.empty()`  Board - Create empty board
- `Board.from_raw(raw: int)`  Board - Create from packed representation
- `board.raw`  int - Get packed representation

**Game Operations:**
- `board.shift(direction: Move)`  Board - Shift without random tile
- `board.make_move(direction: Move, rng=None, *, seed=None)`  Board - Move + random tile
- `board.with_random_tile(rng=None, *, seed=None)`  Board - Add random tile

**Inspection:**
- `board.score()`  int - Current score
- `board.is_game_over()`  bool - No legal moves remain
- `board.highest_tile()`  int - Highest tile value
- `board.count_empty()`  int - Empty cell count
- `board.tile_value(index: int)`  int - Tile value at index (0-15)
- `board.to_values()`  List[int] - All tile values
- `board.to_exponents()`  List[int] - All tile exponents (internal representation)

### Move Enum

- `Move.UP`, `Move.DOWN`, `Move.LEFT`, `Move.RIGHT`
- Supports equality, hashing, string representation

### Rng Class

- `Rng(seed: int)` - Deterministic random number generator
- `rng.clone()`  Rng - Independent copy

## Next Steps / Remaining Work

### High Priority

2. **Serialization Support (v2 format)**
   - `RunV2`, `StepV2`, `Meta`, `BranchV2` classes
   - File I/O methods (`load`/`save`)
   - `RunBuilder` for ergonomic recording

3. **Enhanced Testing**
   - Comprehensive test suite
   - Doctests in Python docstrings
   - Performance benchmarks vs Rust

### Medium Priority

4. **Documentation**
   - Complete API documentation
   - Usage examples and tutorials
   - Migration guide from v1 format

5. **Distribution**
   - Wheel building for multiple platforms
   - PyPI publishing setup
   - CI/CD pipeline

6. **Advanced Features**
   - Async support for long-running operations
   - Progress callbacks for search
   - Custom heuristics interface

## Development Notes

- The Rust library uses a packed u64 representation for the 4x4 board (16 nibbles)
- Random tile generation follows 2048 rules: 90% chance of 2, 10% chance of 4
- All operations preserve Rust semantics (immutable Board, cheap copying)
- PyO3 handles memory management and Python object conversion automatically

## Architecture Decision

The codebase is structured as a Cargo workspace with:
- `ai-2048/`: Core Rust library (engine, expectimax, serialization)
- `py-ai-2048/`: Python bindings using PyO3

This separation allows:
- Clean dependency management
- Independent versioning
- Easier testing and maintenance
- Potential for other language bindings (e.g., WebAssembly)

The Python API closely mirrors the Rust API design documented in `docs/pyo3_proposal.md` for consistency and familiarity.
