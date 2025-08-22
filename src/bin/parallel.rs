use ai_2048::engine as GameEngine;
use ai_2048::engine::{Board, get_score, get_highest_tile_val, Move};
use ai_2048::trace::{self, Meta, encode_run};
use clap::{Parser, Subcommand};
use ai_2048::expectimax::ExpectimaxMultithread;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use std::fs;

use std::path::PathBuf;

fn main() {
    let args = Args::parse();
    GameEngine::new();

    // Subcommand-driven continuous generator mode
    if let Some(Cmd::Forever { out_dir, max_gb, quiet, steps, stop_score, stop_tile }) = &args.cmd {
        let gb = max_gb.unwrap_or(10.0);
        let max_bytes: u64 = if gb.is_finite() && gb > 0.0 { (gb * 1e9) as u64 } else { 10_000_000_000 };
        if let Err(e) = run_generator_mode(out_dir, max_bytes, *quiet, *steps, *stop_score, *stop_tile) {
            eprintln!("Generator mode failed: {e:?}");
        }
        return;
    }
    let start = Instant::now();
    let start_wall = trace::now_unix_seconds();

    let mut expectimax = ExpectimaxMultithread::new();
    let mut rng = rand::thread_rng();
    let mut board: Board = Board::EMPTY.with_random_tile(&mut rng).with_random_tile(&mut rng);

    // In-memory trace buffers (zero overhead aside from simple pushes)
    let mut states: Vec<u64> = Vec::with_capacity(1024);
    let mut moves_vec: Vec<u8> = Vec::with_capacity(1024);
    states.push(board.raw());

    // Status line: global moves/sec via indicatif
    let moves = Arc::new(AtomicU64::new(0));
    let score_atomic = Arc::new(AtomicU64::new(get_score(board)));
    let stop = Arc::new(AtomicBool::new(false));

    let mut status_handle: Option<thread::JoinHandle<()>> = None;
    let mut pb_opt: Option<ProgressBar> = None;
    if !args.quiet {
        let moves_for_status = moves.clone();
        let score_for_status = score_atomic.clone();
        let stop_flag = stop.clone();
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner} {elapsed_precise} | Moves: {msg}")
                .unwrap()
                .tick_chars("⠁⠃⠇⠧⠷⠿⠻⠟⠯⠷⠧⠇⠃"),
        );
        pb.enable_steady_tick(Duration::from_millis(120));
        let pb_bg = pb.clone();
        status_handle = Some(thread::spawn(move || {
            let start = Instant::now();
            while !stop_flag.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(250));
                let m = moves_for_status.load(Ordering::Relaxed);
                let elapsed = start.elapsed().as_secs_f64().max(1e-6);
                let rate = (m as f64) / elapsed;
                let s = score_for_status.load(Ordering::Relaxed);
                pb_bg.set_message(format!("{} | moves/sec: {:.1} | score: {}", m, rate, s));
            }
        }));
        pb_opt = Some(pb);
    }

    let mut move_count: u64 = 0;
    // No board printing in parallel mode

    while !board.is_game_over() {
        let direction = expectimax.get_next_move(board);
        if direction.is_none() {
            break;
        }
        move_count += 1;
        // Record move and resulting state
        let dir = direction.unwrap();
        moves_vec.push(move_to_u8(dir));
        board = board.make_move(dir, &mut rng);
        states.push(board.raw());
        moves.fetch_add(1, Ordering::Relaxed);
        // Tile cap: cheap check every move
        if let Some(tile_target) = args.stop_tile {
            let t = board.highest_tile();
            if t >= tile_target { break; }
        }
        if let Some(limit) = args.steps {
            if move_count >= limit {
                break;
            }
        }
        if move_count % 100 == 0 {
            let s = board.score();
            score_atomic.store(s, Ordering::Relaxed);
            if let Some(target) = args.stop_score {
                if s >= target {
                    break;
                }
            }
        }
    }

    // Stop status thread and print final line
    stop.store(true, Ordering::Relaxed);
    if let Some(h) = status_handle { let _ = h.join(); }
    let final_moves = moves.load(Ordering::Relaxed);
    if let Some(pb) = pb_opt { pb.finish_and_clear(); }
    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let final_score = board.score();
    if !args.quiet {
        println!(
            "Moves: {} | moves/sec: {:.1} | score: {}",
            final_moves,
            (final_moves as f64) / elapsed,
            final_score
        );
    }

    // Optionally write trace to disk after run completes
    if let Some(out_path) = args.out {
        // Compute metadata in a post-pass to avoid per-move overhead
        let highest_tile = states
            .iter()
            .map(|&b| Board::from_raw(b).highest_tile() as u32)
            .max()
            .unwrap_or(0);
        let max_score = states
            .iter()
            .map(|&b| Board::from_raw(b).score())
            .max()
            .unwrap_or(0);
        let meta = Meta {
            steps: moves_vec.len() as u32,
            start_unix_s: start_wall,
            elapsed_s: elapsed as f32,
            max_score,
            highest_tile,
            engine_str: None,
        };
        if let Err(e) = trace::write_run_to_path(out_path, &meta, &states, &moves_vec) {
            eprintln!("Failed to write trace: {e}");
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "parallel", about = "Parallel 2048 expectimax runner")]
struct Args {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Suppress status line output
    #[arg(long)]
    quiet: bool,

    /// Stop once score >= this value (checked every 100 moves)
    #[arg(long)]
    stop_score: Option<u64>,

    /// Stop after this many moves
    #[arg(long)]
    steps: Option<u64>,

    /// Stop once highest tile >= this value (checked every 100 moves)
    #[arg(long)]
    stop_tile: Option<u64>,

    /// Write a binary trace of the run to this path
    #[arg(long)]
    out: Option<PathBuf>,

}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Continuously generate runs into a directory until stopped or size cap reached
    Forever {
        /// Output directory for generated runs
        #[arg(long)]
        out_dir: PathBuf,
        /// Maximum total GB allowed in out_dir (default 10.0)
        #[arg(long)]
        max_gb: Option<f64>,
        /// Suppress the spinner status line
        #[arg(long)]
        quiet: bool,
        /// Per-run: stop after this many moves
        #[arg(long)]
        steps: Option<u64>,
        /// Per-run: stop once score ≥ this value (checked every 100 moves)
        #[arg(long)]
        stop_score: Option<u64>,
        /// Per-run: stop once highest tile ≥ this value (checked every 100 moves)
        #[arg(long)]
        stop_tile: Option<u64>,
    },
}

#[inline]
fn move_to_u8(m: Move) -> u8 {
    match m {
        Move::Up => 0,
        Move::Down => 1,
        Move::Left => 2,
        Move::Right => 3,
    }
}

fn run_single_game(steps: Option<u64>, stop_score: Option<u64>, stop_tile: Option<u64>) -> anyhow::Result<(Vec<u64>, Vec<u8>, f64, u64, u32, u64)> {
    let start = Instant::now();
    let start_wall = trace::now_unix_seconds();
    let mut expectimax = ExpectimaxMultithread::new();
    let mut board: Board = GameEngine::insert_random_tile(Board::EMPTY);
    board = GameEngine::insert_random_tile(board);
    let mut states: Vec<u64> = Vec::with_capacity(1024);
    let mut moves_vec: Vec<u8> = Vec::with_capacity(1024);
    states.push(board.raw());
    let mut move_count: u64 = 0;
    while !GameEngine::is_game_over(board) {
        let direction = expectimax.get_next_move(board);
        if direction.is_none() { break; }
        move_count += 1;
        let dir = direction.unwrap();
        moves_vec.push(move_to_u8(dir));
        board = GameEngine::make_move(board, dir);
        states.push(board.raw());
        if let Some(limit) = steps { if move_count >= limit { break; } }
        if move_count % 100 == 0 {
            if let Some(target) = stop_score { if get_score(board) >= target { break; } }
            if let Some(tile_target) = stop_tile { if get_highest_tile_val(board) >= tile_target { break; } }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let highest_tile = states
        .iter()
        .map(|&b| get_highest_tile_val(Board::from_raw(b)) as u32)
        .max()
        .unwrap_or(0);
    let max_score = states
        .iter()
        .map(|&b| get_score(Board::from_raw(b)))
        .max()
        .unwrap_or(0);
    Ok((states, moves_vec, elapsed, max_score, highest_tile, start_wall))
}

fn run_generator_mode(dir: &PathBuf, max_bytes: u64, quiet: bool, steps: Option<u64>, stop_score: Option<u64>, stop_tile: Option<u64>) -> anyhow::Result<()> {
    fs::create_dir_all(dir)?;
    let mut runs_written: u64 = 0;
    let mut bytes_written: u64 = directory_size_bytes(dir)?;
    let pb = if !quiet {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner} {elapsed_precise} | Runs: {pos} | Size: {msg}")?
                .tick_chars("⠁⠃⠇⠧⠷⠿⠻⠟⠯⠷⠧⠇⠃"),
        );
        pb.enable_steady_tick(Duration::from_millis(120));
        Some(pb)
    } else {
        None
    };

    loop {
        if bytes_written >= max_bytes { break; }
        let (states, moves_vec, elapsed_s, max_score, highest_tile, start_wall) = run_single_game(steps, stop_score, stop_tile)?;
        let meta = Meta {
            steps: moves_vec.len() as u32,
            start_unix_s: start_wall,
            elapsed_s: elapsed_s as f32,
            max_score,
            highest_tile,
            engine_str: None,
        };
        let bytes = encode_run(&meta, &states, &moves_vec);
        let path = autoname(dir, start_wall);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(&path, &bytes)?;
        runs_written += 1;
        bytes_written = bytes_written.saturating_add(bytes.len() as u64);
        if let Some(pb) = &pb {
            pb.set_position(runs_written);
            pb.set_message(format!("{:.2} GB", (bytes_written as f64) / 1e9));
        }
    }

    if let Some(pb) = pb { pb.finish_and_clear(); }
    eprintln!("Generator stopped. Runs: {}, Size: {:.2} GB", runs_written, (bytes_written as f64)/1e9);
    Ok(())
}

fn autoname(dir: &PathBuf, start_unix_s: u64) -> PathBuf {
    // shard by day number since epoch to keep dirs lighter
    let day = start_unix_s / 86_400;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let subdir = dir.join(format!("d{:08}", day));
    subdir.join(format!("run-{}-{:09}.a2run", start_unix_s, nanos))
}

fn directory_size_bytes(dir: &PathBuf) -> anyhow::Result<u64> {
    let mut total = 0u64;
    if !dir.exists() { return Ok(0); }
    for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            if let Ok(md) = entry.metadata() { total = total.saturating_add(md.len()); }
        }
    }
    Ok(total)
}
