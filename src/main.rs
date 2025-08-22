use ai_2048::engine as GameEngine;
use ai_2048::engine::Board;
use ai_2048::expectimax::Expectimax;

fn main() {
    GameEngine::new();
    let mut expectimax = Expectimax::new();
    let mut rng = rand::thread_rng();
    let mut board = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);
    println!("{}", board);
    let mut move_count = 0;
    while !GameEngine::is_game_over(board) {
        let direction = expectimax.get_next_move(board);
        if direction.is_none() {
            break;
        }
        move_count += 1;
        board = board.make_move(direction.unwrap(), &mut rng);
        println!("{}", board);
    }
    println!("Moves made: {}, States considered: {}, Max states considered for a move: {}", move_count, expectimax.0, expectimax.1)
}
