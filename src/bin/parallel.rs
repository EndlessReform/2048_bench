use ai_2048::engine as GameEngine;
use ai_2048::engine::Board;
use ai_2048::expectimax::ExpectimaxMultithread;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    GameEngine::new();
    let start = Instant::now();

    let mut expectimax = ExpectimaxMultithread::new();
    let mut board: Board = GameEngine::insert_random_tile(0);
    board = GameEngine::insert_random_tile(board);

    // Status line: global moves/sec via indicatif
    let moves = Arc::new(AtomicU64::new(0));
    let moves_for_status = moves.clone();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = stop.clone();
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner} Moves: {msg}")
            .unwrap()
            .tick_chars("⠁⠃⠇⠧⠷⠿⠻⠟⠯⠷⠧⠇⠃"),
    );
    pb.enable_steady_tick(Duration::from_millis(120));
    let pb_bg = pb.clone();
    let status_handle = thread::spawn(move || {
        let start = Instant::now();
        while !stop_flag.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(250));
            let m = moves_for_status.load(Ordering::Relaxed);
            let elapsed = start.elapsed().as_secs_f64().max(1e-6);
            let rate = (m as f64) / elapsed;
            pb_bg.set_message(format!("{} | moves/sec: {:.1}", m, rate));
        }
    });

    let mut move_count: u64 = 0;
    // No board printing in parallel mode

    while !GameEngine::is_game_over(board) {
        let direction = expectimax.get_next_move(board);
        if direction.is_none() {
            break;
        }
        move_count += 1;
        board = GameEngine::make_move(board, direction.unwrap());
        moves.fetch_add(1, Ordering::Relaxed);
    }

    // Stop status thread and print final line
    stop.store(true, Ordering::Relaxed);
    let _ = status_handle.join();
    let final_moves = moves.load(Ordering::Relaxed);
    pb.finish_and_clear();
    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    println!("Moves: {} | moves/sec: {:.1}", final_moves, (final_moves as f64) / elapsed);
}
