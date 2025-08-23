import pytest
from ai_2048 import Board, Move, Rng

def test_board_empty():
    """Test creation of an empty board."""
    board = Board.empty()
    assert board.raw == 0
    assert board.score() == 0
    # The rust implementation returns 1 for the highest tile of an empty board.
    assert board.highest_tile() == 1
    assert board.count_empty() == 16
    assert all(v == 0 for v in board.to_values())

def test_board_from_raw():
    """Test creation from a raw u64 value."""
    # Represents a board with a 2 at (0,0) and a 4 at (3,3)
    # Exponent of 2 is 1, exponent of 4 is 2.
    # Index 0 is the most significant nibble, index 15 is the least significant.
    raw_val = 0x1000000000000002
    board = Board.from_raw(raw_val)
    assert board.raw == raw_val
    assert board.to_values()[0] == 2
    assert board.to_exponents()[0] == 1
    assert board.to_values()[15] == 4
    assert board.to_exponents()[15] == 2
    assert board.count_empty() == 14


def test_board_properties():
    """Test board properties."""
    board = Board.from_raw(0x1234123412341234)
    # Score is the sum of the values of the merged tiles.
    # This is hard to test without a reference implementation.
    # We know the highest tile is 4 (exponent 4), so value is 16.
    assert board.highest_tile() == 16
    assert board.count_empty() == 0

def test_board_iteration():
    """Test iterating over board tiles."""
    board = Board.from_raw(0x4321432143214321)
    values = list(board)
    assert len(values) == 16
    assert values[0] == 16
    assert values[1] == 8
    assert values[2] == 4
    assert values[3] == 2

def test_board_representation():
    """Test string and repr representations."""
    board = Board.empty()
    # Empty board is represented by spaces
    assert " " in str(board)
    assert "Board(0x0000000000000000)" == repr(board)
    raw_val = 0x1000000000000000
    board = Board.from_raw(raw_val)
    assert "Board(0x1000000000000000)" == repr(board)

def test_board_equality():
    """Test board equality."""
    board1 = Board.from_raw(0x1234)
    board2 = Board.from_raw(0x1234)
    board3 = Board.from_raw(0x5678)
    assert board1 == board2
    assert board1 != board3
    assert not (board1 == "not a board")

def test_shift_and_make_move():
    """Test board shifting and making moves."""
    # Board with one tile that can be shifted
    board = Board.from_raw(0x1000000000000000) # A 2 in the top-left corner
    
    # Shift right
    shifted_right = board.shift(Move.RIGHT)
    assert shifted_right.raw != board.raw
    assert shifted_right.to_values()[3] == 2

    # Shift down
    shifted_down = board.shift(Move.DOWN)
    assert shifted_down.raw != board.raw
    assert shifted_down.to_values()[12] == 2

    # Make move (includes adding a random tile)
    rng = Rng(42)
    moved_board = board.make_move(Move.RIGHT, rng=rng)
    assert moved_board.raw != shifted_right.raw
    assert moved_board.count_empty() == 14 # Shifted and a new tile added

def test_with_random_tile():
    """Test adding a random tile."""
    board = Board.empty()
    rng = Rng(42)
    
    board_with_tile = board.with_random_tile(rng=rng)
    assert board_with_tile.count_empty() == 15
    
    # Ensure determinism
    rng1 = Rng(123)
    rng2 = Rng(123)
    board1 = Board.empty().with_random_tile(rng=rng1)
    board2 = Board.empty().with_random_tile(rng=rng2)
    assert board1 == board2

def test_is_game_over():
    """Test game over condition."""
    # An empty board is considered game over by the rust implementation
    # because no move can change the board.
    assert Board.empty().is_game_over()

    # A full board with no possible moves is game over
    # A board with unique, non-mergeable tiles
    raw_val = 0x123456789abcdef1
    full_board = Board.from_raw(raw_val)
    assert full_board.count_empty() == 0
    assert full_board.is_game_over()

    # A board with possible moves is not game over
    raw_val = 0x1100000000000000 # Two 2s that can be merged
    mergeable_board = Board.from_raw(raw_val)
    assert not mergeable_board.is_game_over()

def test_tile_inspection():
    """Test tile_value, to_values, and to_exponents."""
    raw_val = 0x4321000000000000
    board = Board.from_raw(raw_val)

    # to_values
    values = board.to_values()
    assert values[0] == 16
    assert values[1] == 8
    assert values[2] == 4
    assert values[3] == 2
    assert values[4] == 0

    # to_exponents
    exponents = board.to_exponents()
    assert exponents[0] == 4
    assert exponents[1] == 3
    assert exponents[2] == 2
    assert exponents[3] == 1
    assert exponents[4] == 0

    # tile_value
    assert board.tile_value(0) == 16
    assert board.tile_value(1) == 8
    assert board.tile_value(15) == 0

def test_integration_game_loop():
    """A simple integration test simulating a few moves."""
    rng = Rng(99)
    board = Board.empty().with_random_tile(rng=rng).with_random_tile(rng=rng)
    
    moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT]
    
    for move in moves:
        if board.is_game_over():
            break
        # A real AI would be smarter, here we just cycle moves
        board = board.make_move(move, rng=rng)

    assert board.score() > 0
    assert board.count_empty() < 16