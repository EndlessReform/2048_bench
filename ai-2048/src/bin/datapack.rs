use std::path::PathBuf;

use ai_2048::serialization::{DataPack, DataPackError, PackBuilder};
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "datapack",
    version,
    about = "Build and inspect the RAM-friendly 2048 dataset pack (.dat)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build a .dat pack from a directory of .a2run2 files
    Build {
        /// Input directory to scan recursively
        #[arg(long, value_name = "DIR")]
        input: PathBuf,
        /// Output .dat file path
        #[arg(long, value_name = "FILE")]
        output: PathBuf,
    },
    /// Validate a .dat file and print a brief summary
    Validate {
        /// .dat pack path
        #[arg(long, value_name = "FILE")]
        pack: PathBuf,
    },
    /// Print stats (runs, steps, min/max/mean length)
    Stats {
        /// .dat pack path
        #[arg(long, value_name = "FILE")]
        pack: PathBuf,
    },
    /// Inspect a single run's metadata
    Inspect {
        /// .dat pack path
        #[arg(long, value_name = "FILE")]
        pack: PathBuf,
        /// Run index to inspect
        #[arg(long, value_name = "IDX")]
        index: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Build { input, output } => {
            let builder = PackBuilder::from_directory(&input)?;
            builder.write_to_file(&output)?;
            eprintln!("Built DataPack: {}", output.display());
        }
        Command::Validate { pack } => {
            match DataPack::load(&pack) {
                Ok(dp) => {
                    eprintln!(
                        "OK: {} ({} runs, {} steps)",
                        pack.display(),
                        dp.runs.len(),
                        dp.steps.len()
                    );
                }
                Err(e) => {
                    eprintln!("INVALID: {} ({})", pack.display(), e);
                    std::process::exit(2);
                }
            }
        }
        Command::Stats { pack } => {
            let dp = DataPack::load(&pack)?;
            let count = dp.runs.len() as u64;
            let total_steps = dp.steps.len() as u64;
            let mut lens: Vec<u32> = dp.runs.iter().map(|r| r.num_steps).collect();
            lens.sort_unstable();
            let min_len = *lens.first().unwrap_or(&0);
            let max_len = *lens.last().unwrap_or(&0);
            let mean_len = if count == 0 { 0.0 } else { total_steps as f64 / count as f64 };
            println!("pack: {}", pack.display());
            println!("runs: {}", count);
            println!("total_steps: {}", total_steps);
            println!("min_len: {}", min_len);
            println!("max_len: {}", max_len);
            println!("mean_len: {:.3}", mean_len);
        }
        Command::Inspect { pack, index } => {
            let dp = DataPack::load(&pack)?;
            if index >= dp.runs.len() {
                return Err(Box::new(DataPackError::Malformed("index out of range")));
            }
            let r = &dp.runs[index];
            println!("pack: {}", pack.display());
            println!("index: {}", index);
            println!("first_step_idx: {}", r.first_step_idx);
            println!("num_steps: {}", r.num_steps);
            println!("max_score: {}", r.max_score);
            println!("highest_tile: {}", r.highest_tile);
            println!("engine: {}", r.engine);
            println!("start_unix_s: {}", r.start_time);
            println!("elapsed_s: {:.3}", r.elapsed_s);
        }
    }
    Ok(())
}

