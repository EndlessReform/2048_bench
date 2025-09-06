use std::path::PathBuf;

use ai_2048::serialization::dataset::build_dataset;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "dataset",
    version,
    about = "Build 2048 training datasets (steps.npy + metadata.db)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build a dataset directory from .a2run2 files
    Build {
        /// Input directory to scan recursively
        #[arg(long, value_name = "DIR")]
        input: PathBuf,
        /// Output dataset directory
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
    },
    /// Append new runs to an existing dataset directory
    Append {
        /// Existing dataset directory (must contain steps.npy and metadata.db)
        #[arg(long, value_name = "DIR")]
        dataset: PathBuf,
        /// Input directory with .a2run2 runs to append
        #[arg(long, value_name = "DIR")]
        input: PathBuf,
    },
    /// Print dataset stats (rows and runs)
    Stats {
        /// Dataset directory
        #[arg(long, value_name = "DIR")]
        dataset: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Build { input, out } => {
            let rep = build_dataset(&input, &out)?;
            eprintln!("Built dataset: runs={}, steps={}, dir={}", rep.runs, rep.steps, out.display());
        }
        Command::Append { dataset, input } => {
            let rep = ai_2048::serialization::dataset::append_runs(&dataset, &input)?;
            eprintln!("Appended: runs={}, steps={}, dir={}", rep.runs, rep.steps, dataset.display());
        }
        Command::Stats { dataset } => {
            let steps_path = dataset.join("steps.npy");
            let rows = ai_2048::serialization::dataset::npy_row_count(&steps_path)?;
            let conn = rusqlite::Connection::open(dataset.join("metadata.db"))?;
            let (runs, total_steps): (i64, i64) = {
                let mut stmt = conn.prepare("SELECT COUNT(*), COALESCE(SUM(num_steps),0) FROM runs")?;
                stmt.query_row([], |r| Ok((r.get(0)?, r.get(1)?)))?
            };
            println!("dataset: {}", dataset.display());
            println!("steps: {}", rows);
            println!("runs: {}", runs);
            println!("sum_steps_meta: {}", total_steps);
        }
    }
    Ok(())
}
