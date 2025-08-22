use ai_2048::engine as GameEngine;
use ai_2048::engine::Board;
use ai_2048::expectimax::Expectimax;

fn main() {
    GameEngine::new();
    let mut expectimax = Expectimax::new();
    let mut rng = rand::thread_rng();
    let mut board = Board::EMPTY
        .with_random_tile(&mut rng)
        .with_random_tile(&mut rng);
    println!("{}", board);
    let mut move_count = 0;
    let mut total_states: u64 = 0;
    let mut peak_states: u64 = 0;
    while !GameEngine::is_game_over(board) {
        let direction = expectimax.get_next_move(board);
        if direction.is_none() {
            break;
        }
        move_count += 1;
        board = board.make_move(direction.unwrap(), &mut rng);
        println!("{}", board);
        let stats = expectimax.last_stats();
        total_states = total_states.saturating_add(stats.nodes);
        peak_states = peak_states.max(stats.nodes);
    }
    println!(
        "Moves made: {}, States considered: {}, Max states considered for a move: {}",
        move_count, total_states, peak_states
    );
}
