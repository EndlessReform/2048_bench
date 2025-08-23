import pytest
from ai_2048 import Board, Move, Expectimax, ExpectimaxConfig

def test_expectimax_instantiation():
    """Test that the Expectimax AI can be instantiated."""
    ai = Expectimax()
    assert ai is not None

def test_expectimax_best_move():
    """Test that the AI can find a best move."""
    # A board where the best move is clearly down
    board = Board.from_raw(0x1100000000000000) # two 2s in the top row
    ai = Expectimax()
    best_move = ai.best_move(board)
    # The best move is to merge the two 2s. This can be done by moving up or down.
    # The AI should prefer down as it keeps the top rows clear.
    assert best_move == Move.DOWN

def test_expectimax_with_config():
    """Test instantiating Expectimax with a config."""
    config = ExpectimaxConfig()
    ai = Expectimax(config)
    assert ai is not None

def test_expectimax_stats():
    """Test search statistics."""
    board = Board.from_raw(0x1234000000000000)
    ai = Expectimax()
    _ = ai.best_move(board)
    stats = ai.last_stats()
    assert stats.nodes > 0
    assert stats.peak_nodes > 0
    ai.reset_stats()
    stats = ai.last_stats()
    assert stats.nodes == 0
    assert stats.peak_nodes == 0