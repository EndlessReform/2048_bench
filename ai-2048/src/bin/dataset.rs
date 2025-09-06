use std::path::PathBuf;

use ai_2048::serialization::dataset::{build_dataset, build_dataset_from_runs};
use ai_2048::serialization as ser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
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
        #[arg(short = 'i', long = "input", value_name = "DIR")]
        input: PathBuf,
        /// Output dataset directory
        #[arg(short = 'o', long = "out", value_name = "DIR")]
        out: PathBuf,
        /// Show a progress bar while decoding runs
        #[arg(long)]
        progress: bool,
    },
    /// Append new runs to an existing dataset directory
    Append {
        /// Existing dataset directory (must contain steps.npy and metadata.db)
        #[arg(short = 'd', long = "dataset", value_name = "DIR")]
        dataset: PathBuf,
        /// Input directory with .a2run2 runs to append
        #[arg(short = 'i', long = "input", value_name = "DIR")]
        input: PathBuf,
    },
    /// Print dataset stats (rows and runs)
    Stats {
        /// Dataset directory
        #[arg(short = 'd', long = "dataset", value_name = "DIR")]
        dataset: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Build { input, out, progress } => {
            let rep = if progress {
                // Discover files and decode with a progress bar
                let mut files: Vec<std::path::PathBuf> = Vec::new();
                for e in walkdir::WalkDir::new(&input).into_iter().filter_map(Result::ok) {
                    if e.file_type().is_file() {
                        let p = e.path();
                        if p.extension().and_then(|s| s.to_str()) == Some("a2run2") {
                            files.push(p.to_path_buf());
                        }
                    }
                }
                files.sort();
                if files.is_empty() {
                    return Err(std::io::Error::new(std::io::ErrorKind::Other, "no .a2run2 files found").into());
                }

                let pb = ProgressBar::new(files.len() as u64);
                pb.set_style(
                    ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} runs ({eta})")
                        .unwrap()
                        .progress_chars("=>-"),
                );
                let runs: Vec<ser::RunV2> = files
                    .par_iter()
                    .map(|p| {
                        let r = ser::read_postcard_from_path(p);
                        pb.inc(1);
                        r.map_err(|e| anyhow::anyhow!(e))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                pb.finish_with_message("decoded runs");
                build_dataset_from_runs(&runs, &out)?
            } else {
                build_dataset(&input, &out)?
            };
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
