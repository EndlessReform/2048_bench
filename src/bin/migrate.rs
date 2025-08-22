use ai_2048::{trace, serialization};
use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Parser)]
#[command(name = "migrate", about = "Migrate v1 (.a2run) trace files to v2 (.a2run2) format")]
struct Args {
    /// Input path: either a single .a2run file or directory containing .a2run files
    input: PathBuf,
    
    /// Output path: directory for migrated files, or specific output file if input is a single file
    #[arg(short, long)]
    output: Option<PathBuf>,
    
    /// Dry run: show what would be migrated without actually converting files
    #[arg(long)]
    dry_run: bool,
    
    /// Overwrite existing .a2run2 files
    #[arg(long)]
    force: bool,
    
    /// Recursive directory traversal (default: true for directories)
    #[arg(long)]
    recursive: Option<bool>,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    if args.input.is_file() {
        migrate_single_file(&args)?;
    } else if args.input.is_dir() {
        migrate_directory(&args)?;
    } else {
        anyhow::bail!("Input path '{}' is neither a file nor directory", args.input.display());
    }
    
    Ok(())
}

fn migrate_single_file(args: &Args) -> anyhow::Result<()> {
    let input_path = &args.input;
    
    // Validate input file extension
    let ext = input_path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if ext != "a2run" {
        anyhow::bail!("Input file must have .a2run extension, got: {}", input_path.display());
    }
    
    // Determine output path
    let output_path = if let Some(ref out) = args.output {
        out.clone()
    } else {
        input_path.with_extension("a2run2")
    };
    
    migrate_file(input_path, &output_path, args)?;
    Ok(())
}

fn migrate_directory(args: &Args) -> anyhow::Result<()> {
    let input_dir = &args.input;
    let output_dir = args.output.as_ref().unwrap_or(input_dir);
    let recursive = args.recursive.unwrap_or(true);
    
    if args.verbose {
        println!("Scanning directory: {}", input_dir.display());
        println!("Output directory: {}", output_dir.display());
        println!("Recursive: {}", recursive);
    }
    
    // Collect all .a2run files
    let mut v1_files = Vec::new();
    
    let walker = if recursive {
        WalkDir::new(input_dir).into_iter()
    } else {
        WalkDir::new(input_dir).max_depth(1).into_iter()
    };
    
    for entry in walker {
        let entry = entry?;
        if entry.file_type().is_file() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "a2run" {
                    v1_files.push(path.to_path_buf());
                }
            }
        }
    }
    
    if v1_files.is_empty() {
        println!("No .a2run files found in {}", input_dir.display());
        return Ok(());
    }
    
    println!("Found {} .a2run files to migrate", v1_files.len());
    
    if args.dry_run {
        println!("\n--- DRY RUN ---");
        for v1_file in &v1_files {
            let rel_path = v1_file.strip_prefix(input_dir).unwrap_or(v1_file);
            let output_file = output_dir.join(rel_path).with_extension("a2run2");
            println!("Would migrate: {} -> {}", v1_file.display(), output_file.display());
        }
        return Ok(());
    }
    
    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir)?;
    
    let mut successful = 0;
    let mut failed = 0;
    
    for v1_file in &v1_files {
        let rel_path = v1_file.strip_prefix(input_dir).unwrap_or(v1_file);
        let output_file = output_dir.join(rel_path).with_extension("a2run2");
        
        // Ensure parent directory exists
        if let Some(parent) = output_file.parent() {
            fs::create_dir_all(parent)?;
        }
        
        match migrate_file(v1_file, &output_file, args) {
            Ok(()) => {
                successful += 1;
                if args.verbose {
                    println!("✓ Migrated: {} -> {}", v1_file.display(), output_file.display());
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("✗ Failed to migrate {}: {}", v1_file.display(), e);
            }
        }
    }
    
    println!("\nMigration complete: {} successful, {} failed", successful, failed);
    Ok(())
}

fn migrate_file(input_path: &Path, output_path: &Path, args: &Args) -> anyhow::Result<()> {
    // Check if output already exists
    if output_path.exists() && !args.force {
        anyhow::bail!("Output file already exists: {} (use --force to overwrite)", output_path.display());
    }
    
    if args.verbose {
        println!("Reading v1 file: {}", input_path.display());
    }
    
    // Read v1 format
    let v1_run = trace::parse_run_file(input_path)
        .map_err(|e| anyhow::anyhow!("Failed to parse v1 file: {}", e))?;
    
    if args.verbose {
        println!("Parsed v1 run: {} steps, {} moves, score: {}", 
                 v1_run.meta.steps, v1_run.moves.len(), v1_run.meta.max_score);
    }
    
    // Convert to v2 format
    let v2_run = serialization::from_v1(v1_run);
    
    if args.verbose {
        println!("Converted to v2 format");
    }
    
    // Write v2 format
    serialization::write_postcard_to_path(output_path, &v2_run)
        .map_err(|e| anyhow::anyhow!("Failed to write v2 file: {}", e))?;
    
    if args.verbose {
        println!("Written v2 file: {}", output_path.display());
    }
    
    // Compare file sizes
    if args.verbose {
        let v1_size = fs::metadata(input_path)?.len();
        let v2_size = fs::metadata(output_path)?.len();
        let ratio = (v2_size as f64) / (v1_size as f64);
        println!("Size comparison: v1={} bytes, v2={} bytes (ratio: {:.2}x)", 
                 v1_size, v2_size, ratio);
    }
    
    Ok(())
}