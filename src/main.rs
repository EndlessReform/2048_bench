use ai_2048::engine as GameEngine;
use ai_2048::expectimax::{Expectimax};

fn main() {
    GameEngine::new();
    let mut expectimax = Expectimax::new();
    let mut board = GameEngine::insert_random_tile(0);
    board = GameEngine::insert_random_tile(board);
    println!("{}", GameEngine::to_str(board));
    let mut move_count = 0;
    while !GameEngine::is_game_over(board) {
        let direction = expectimax.get_next_move(board);
        if direction.is_none() {
            break;
        }
        move_count += 1;
        board = GameEngine::make_move(board, direction.unwrap());
        println!("{}", GameEngine::to_str(board));
    }
    println!("Moves made: {}, States considered: {}, Max states considered for a move: {}", move_count, expectimax.0, expectimax.1)
}
