use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use ai_2048::serialization::{DataPack, DataPackError, PackBuilder};
use clap::{Parser, Subcommand};
use std::time::Instant;
use rand::{rngs::StdRng, Rng, SeedableRng};

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
    /// Time loading and index rebuild costs for a .dat file
    TimeLoad {
        /// .dat pack path
        #[arg(long, value_name = "FILE")]
        pack: PathBuf,
    },
    /// Benchmark random batch reads over a .dat file
    BenchBatches {
        /// .dat pack path
        #[arg(long, value_name = "FILE")]
        pack: PathBuf,
        /// Batch size (default 768)
        #[arg(long, default_value_t = 768usize)]
        batch: usize,
        /// Number of batches to measure
        #[arg(long, default_value_t = 500usize)]
        iters: usize,
        /// Seed for RNG to make runs deterministic
        #[arg(long, default_value_t = 1337u64)]
        seed: u64,
        /// Pre-generate all indices to exclude RNG cost from the measurement
        #[arg(long, default_value_t = true)]
        pregenerate: bool,
    },
    /// Append new runs from a directory of .a2run2 files to an existing .dat
    Append {
        /// Existing .dat pack path (input)
        #[arg(long, value_name = "FILE")]
        pack: PathBuf,
        /// Directory containing new .a2run2 runs
        #[arg(long, value_name = "DIR")]
        input: PathBuf,
        /// Output .dat path (written atomically via a temp file then renamed)
        #[arg(long, value_name = "FILE")]
        output: PathBuf,
    },
    /// Merge two .dat files (A then B) into a new .dat
    Merge {
        /// First .dat pack path (A)
        #[arg(long, value_name = "FILE")]
        a: PathBuf,
        /// Second .dat pack path (B)
        #[arg(long, value_name = "FILE")]
        b: PathBuf,
        /// Output .dat path (written atomically via a temp file then renamed)
        #[arg(long, value_name = "FILE")]
        output: PathBuf,
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
        Command::TimeLoad { pack } => {
            // Measure total load time
            let t0 = Instant::now();
            let dp = DataPack::load(&pack)?;
            let dt_load = t0.elapsed();

            // Measure index rebuild separately (runs-based derived indices)
            let mut dp_for_index = dp.clone();
            let t1 = Instant::now();
            dp_for_index.rebuild_indices();
            let dt_index = t1.elapsed();

            // File size if accessible
            let file_size = std::fs::metadata(&pack).map(|m| m.len()).unwrap_or(0);

            println!("pack: {}", pack.display());
            println!("file_bytes: {}", file_size);
            println!("runs: {}", dp_for_index.runs.len());
            println!("steps: {}", dp_for_index.steps.len());
            println!("load_total_ms: {:.3}", dt_load.as_secs_f64() * 1e3);
            println!("reindex_ms: {:.3}", dt_index.as_secs_f64() * 1e3);
        }
        Command::BenchBatches { pack, batch, iters, seed, pregenerate } => {
            if batch == 0 || iters == 0 {
                return Err("batch and iters must be > 0".into());
            }
            let dp = DataPack::load(&pack)?;
            let total_steps = dp.steps.len();
            if total_steps == 0 {
                return Err("datapack has zero steps".into());
            }

            // Build a pool of indices for sampling
            let mut rng = StdRng::seed_from_u64(seed);
            let total_samples = batch.checked_mul(iters).ok_or_else(||
                Box::<dyn std::error::Error>::from("overflow computing total_samples")
            )?;

            let indices: Vec<usize> = if pregenerate {
                (0..total_samples).map(|_| rng.gen_range(0..total_steps)).collect()
            } else {
                Vec::new()
            };

            // Warm up cache minimally by touching one element per page-ish
            let mut warm_sum: u64 = 0;
            let stride = (total_steps / 1024).max(1);
            for i in (0..total_steps).step_by(stride) {
                warm_sum = warm_sum.wrapping_add(dp.steps[i].board);
            }
            std::hint::black_box(warm_sum);

            // Benchmark loop: read batches and accumulate a trivial reduction to avoid DCE
            let t0 = Instant::now();
            let mut acc: u64 = 0;
            if pregenerate {
                let mut off = 0;
                for _ in 0..iters {
                    let slice = &indices[off..off + batch];
                    off += batch;
                    for &idx in slice {
                        // Touch a couple of fields to simulate typical access
                        let s = unsafe { dp.steps.get_unchecked(idx) };
                        acc = acc.wrapping_add(s.board).wrapping_add(s.run_id as u64);
                    }
                }
            } else {
                for _ in 0..iters {
                    for _ in 0..batch {
                        let idx = rng.gen_range(0..total_steps);
                        let s = unsafe { dp.steps.get_unchecked(idx) };
                        acc = acc.wrapping_add(s.board).wrapping_add(s.run_id as u64);
                    }
                }
            }
            let dt = t0.elapsed();
            std::hint::black_box(acc);

            let total_batches = iters as f64;
            let total_items = (iters as u128) * (batch as u128);
            let secs = dt.as_secs_f64();
            let batches_per_s = total_batches / secs;
            let items_per_s = (total_items as f64) / secs;

            println!("pack: {}", pack.display());
            println!("steps: {}", total_steps);
            println!("batch: {}", batch);
            println!("iters: {}", iters);
            println!("pregenerate: {}", pregenerate);
            println!("elapsed_ms: {:.3}", secs * 1e3);
            println!("batches_per_s: {:.2}", batches_per_s);
            println!("steps_per_s: {:.0}", items_per_s);
            // Approx bytes/s (touching ~16 bytes per Step: board + run_id + index_in_run)
            let approx_bytes_per_item = 8.0 + 4.0 + 4.0; // 16 bytes
            println!(
                "approx_read_gbps: {:.3}",
                (items_per_s * approx_bytes_per_item) / 1e9
            );
        }
        Command::Append { pack, input, output } => {
            append_runs(&pack, &input, &output)?;
        }
        Command::Merge { a, b, output } => {
            merge_packs(&a, &b, &output)?;
        }
    }
    Ok(())
}

const DAT_MAGIC: &[u8; 4] = b"2048";
const DAT_VERSION: u32 = 1;

struct DatView {
    num_runs: u32,
    num_steps: u64,
    metadata_offset: u64,
    content_len: usize, // excludes trailer CRC (file_len - 4)
}

fn read_dat_header(f: &mut File) -> Result<DatView, Box<dyn std::error::Error>> {
    let len = f.metadata()?.len() as usize;
    if len < 4 + 4 + 4 + 8 + 8 + 4 { return Err("file too small".into()); }
    let mut head = [0u8; 28];
    f.seek(SeekFrom::Start(0))?;
    f.read_exact(&mut head)?;
    if &head[0..4] != DAT_MAGIC { return Err("bad magic".into()); }
    let ver = u32::from_le_bytes(head[4..8].try_into().unwrap());
    if ver != DAT_VERSION { return Err("unsupported version".into()); }
    let num_runs = u32::from_le_bytes(head[8..12].try_into().unwrap());
    let num_steps = u64::from_le_bytes(head[12..20].try_into().unwrap());
    let metadata_offset = u64::from_le_bytes(head[20..28].try_into().unwrap());
    let content_len = len - 4; // strip trailer crc
    if (metadata_offset as usize) > content_len || (metadata_offset as usize) < 28 {
        return Err("metadata_offset out of range".into());
    }
    // Steps section length must be consistent
    let expected_steps_bytes = (num_steps as usize).checked_mul(32).ok_or("steps size overflow")?;
    let actual_steps_bytes = (metadata_offset as usize) - 28;
    if expected_steps_bytes != actual_steps_bytes { return Err("steps length mismatch".into()); }
    Ok(DatView { num_runs, num_steps, metadata_offset, content_len })
}

fn crc32c_update(mut state: u32, bytes: &[u8]) -> u32 {
    state = crc32c::crc32c_append(state, bytes);
    state
}

fn write_header(out: &mut File, hasher: &mut u32, total_runs: u32, total_steps: u64, metadata_offset: u64) -> std::io::Result<()> {
    let mut buf = [0u8; 28];
    buf[0..4].copy_from_slice(DAT_MAGIC);
    buf[4..8].copy_from_slice(&DAT_VERSION.to_le_bytes());
    buf[8..12].copy_from_slice(&total_runs.to_le_bytes());
    buf[12..20].copy_from_slice(&total_steps.to_le_bytes());
    buf[20..28].copy_from_slice(&metadata_offset.to_le_bytes());
    *hasher = crc32c_update(*hasher, &buf);
    out.write_all(&buf)
}

fn copy_region(out: &mut File, hasher: &mut u32, f: &mut File, offset: u64, len: usize) -> std::io::Result<()> {
    f.seek(SeekFrom::Start(offset))?;
    let mut rem = len;
    let mut buf = vec![0u8; 256 * 1024];
    while rem > 0 {
        let to_read = buf.len().min(rem);
        f.read_exact(&mut buf[..to_read])?;
        out.write_all(&buf[..to_read])?;
        *hasher = crc32c_update(*hasher, &buf[..to_read]);
        rem -= to_read;
    }
    Ok(())
}

fn write_step_adjusted(out: &mut File, hasher: &mut u32, s: &ai_2048::serialization::Step, run_id_offset: u32) -> std::io::Result<()> {
    let mut buf = [0u8; 32];
    // Layout: board[8], run_id[4], index_in_run[4], move_dir[1], ev_mask[1], ev_q[8], pad[6]
    buf[0..8].copy_from_slice(&s.board.to_le_bytes());
    let rid = s.run_id.wrapping_add(run_id_offset);
    buf[8..12].copy_from_slice(&rid.to_le_bytes());
    buf[12..16].copy_from_slice(&s.index_in_run.to_le_bytes());
    buf[16] = s.move_dir;
    buf[17] = s.ev_mask;
    buf[18..20].copy_from_slice(&s.ev_q[0].to_le_bytes());
    buf[20..22].copy_from_slice(&s.ev_q[1].to_le_bytes());
    buf[22..24].copy_from_slice(&s.ev_q[2].to_le_bytes());
    buf[24..26].copy_from_slice(&s.ev_q[3].to_le_bytes());
    // pad bytes remain zero
    out.write_all(&buf)?;
    *hasher = crc32c_update(*hasher, &buf);
    Ok(())
}

fn write_runmeta_adjusted(out: &mut File, hasher: &mut u32, r: &ai_2048::serialization::RunMeta, id_offset: u32, first_step_offset: u32) -> std::io::Result<()> {
    let mut tmp = Vec::with_capacity(4 + 4 + 4 + 8 + 4 + 2 + r.engine.len() + 8 + 4);
    let id = r.id.wrapping_add(id_offset);
    tmp.extend_from_slice(&id.to_le_bytes());
    let first = r.first_step_idx.wrapping_add(first_step_offset);
    tmp.extend_from_slice(&first.to_le_bytes());
    tmp.extend_from_slice(&r.num_steps.to_le_bytes());
    tmp.extend_from_slice(&r.max_score.to_le_bytes());
    tmp.extend_from_slice(&r.highest_tile.to_le_bytes());
    let eng = r.engine.as_bytes();
    let eng_len = u16::try_from(eng.len()).unwrap_or(0);
    tmp.extend_from_slice(&eng_len.to_le_bytes());
    tmp.extend_from_slice(eng);
    tmp.extend_from_slice(&r.start_time.to_le_bytes());
    tmp.extend_from_slice(&r.elapsed_s.to_bits().to_le_bytes());
    out.write_all(&tmp)?;
    *hasher = crc32c_update(*hasher, &tmp);
    Ok(())
}

fn append_runs(pack: &Path, input_dir: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Load new runs into a DataPack (keeps memory bounded to the delta)
    let builder = PackBuilder::from_directory(input_dir)?;
    let new_dp = builder.build();

    // Open existing pack and parse header
    let mut f = File::open(pack)?;
    let hdr = read_dat_header(&mut f)?;
    let old_steps_bytes = (hdr.num_steps as usize) * 32;
    let old_meta_bytes = hdr.content_len - (hdr.metadata_offset as usize);

    // Compute new header fields
    let new_runs_count = u32::try_from(new_dp.runs.len())?;
    let new_steps_count = u64::try_from(new_dp.steps.len())?;
    let total_runs = hdr.num_runs + new_runs_count;
    let total_steps = hdr.num_steps + new_steps_count;
    let new_metadata_offset = 28u64 + (total_steps as u64) * 32u64;

    // Prepare temp output and rolling CRC
    let out_tmp = output.with_extension("tmp");
    let mut out = File::create(&out_tmp)?;
    let mut hasher: u32 = 0;

    // Header
    write_header(&mut out, &mut hasher, total_runs, total_steps, new_metadata_offset)?;

    // Steps: old + new
    copy_region(&mut out, &mut hasher, &mut f, 28, old_steps_bytes)?;
    for s in &new_dp.steps {
        write_step_adjusted(&mut out, &mut hasher, s, hdr.num_runs)?;
    }

    // Metadata: old (verbatim) + new (adjusted)
    copy_region(&mut out, &mut hasher, &mut f, hdr.metadata_offset, old_meta_bytes)?;
    let mut cum_steps_new: u32 = 0;
    let first_step_offset = u32::try_from(hdr.num_steps)?;
    for r in &new_dp.runs {
        let adj = ai_2048::serialization::RunMeta {
            id: r.id,
            first_step_idx: r.first_step_idx + cum_steps_new,
            num_steps: r.num_steps,
            max_score: r.max_score,
            highest_tile: r.highest_tile,
            engine: r.engine.clone(),
            start_time: r.start_time,
            elapsed_s: r.elapsed_s,
        };
        write_runmeta_adjusted(&mut out, &mut hasher, &adj, hdr.num_runs, first_step_offset)?;
        cum_steps_new = cum_steps_new.saturating_add(r.num_steps);
    }

    // Trailer CRC over all preceding bytes
    out.write_all(&hasher.to_le_bytes())?;
    out.flush()?;

    // Atomic rename
    if output.exists() { fs::remove_file(output).ok(); }
    fs::rename(&out_tmp, output)?;
    eprintln!("Appended {} runs ({} steps): {} -> {}", new_runs_count, new_steps_count, pack.display(), output.display());
    Ok(())
}

fn merge_packs(a: &Path, b: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut fa = File::open(a)?;
    let mut fb = File::open(b)?;
    let ha = read_dat_header(&mut fa)?;
    let hb = read_dat_header(&mut fb)?;

    let a_steps_bytes = (ha.num_steps as usize) * 32;
    let b_steps_bytes = (hb.num_steps as usize) * 32;
    let a_meta_bytes = ha.content_len - (ha.metadata_offset as usize);
    let b_meta_bytes = hb.content_len - (hb.metadata_offset as usize);

    let total_runs = ha.num_runs + hb.num_runs;
    let total_steps = ha.num_steps + hb.num_steps;
    let new_metadata_offset = 28u64 + (total_steps as u64) * 32u64;

    let out_tmp = output.with_extension("tmp");
    let mut out = File::create(&out_tmp)?;
    let mut hasher: u32 = 0;
    write_header(&mut out, &mut hasher, total_runs, total_steps, new_metadata_offset)?;

    // Copy A steps
    copy_region(&mut out, &mut hasher, &mut fa, 28, a_steps_bytes)?;

    // Copy B steps with run_id += ha.num_runs
    fb.seek(SeekFrom::Start(28))?;
    let mut rem = b_steps_bytes;
    let mut buf = vec![0u8; 32 * 4096]; // 128 KiB, 32-byte aligned
    while rem > 0 {
        let to_read = buf.len().min(rem);
        fb.read_exact(&mut buf[..to_read])?;
        // Adjust run_id per 32-byte record
        for chunk in buf[..to_read].chunks_mut(32) {
            // run_id at bytes 8..12
            let mut rid = u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
            rid = rid.wrapping_add(ha.num_runs);
            let bytes = rid.to_le_bytes();
            chunk[8..12].copy_from_slice(&bytes);
        }
        out.write_all(&buf[..to_read])?;
        hasher = crc32c_update(hasher, &buf[..to_read]);
        rem -= to_read;
    }

    // Metadata: A verbatim, then B adjusted (id += a_runs, first_step_idx += a_steps)
    copy_region(&mut out, &mut hasher, &mut fa, ha.metadata_offset, a_meta_bytes)?;
    // For B metadata, parse entries and rewrite adjusted fields
    let mut meta_buf = vec![0u8; b_meta_bytes];
    fb.seek(SeekFrom::Start(hb.metadata_offset))?;
    fb.read_exact(&mut meta_buf)?;
    let mut off = 0usize;
    for _ in 0..hb.num_runs as usize {
        // Fixed-size portion check
        if off + 4 + 4 + 4 + 8 + 4 + 2 > meta_buf.len() { return Err("B run meta truncated".into()); }
        let id = u32::from_le_bytes(meta_buf[off..off + 4].try_into().unwrap()); off += 4;
        let first = u32::from_le_bytes(meta_buf[off..off + 4].try_into().unwrap()); off += 4;
        let num_steps_run = u32::from_le_bytes(meta_buf[off..off + 4].try_into().unwrap()); off += 4;
        let max_score = u64::from_le_bytes(meta_buf[off..off + 8].try_into().unwrap()); off += 8;
        let highest_tile = u32::from_le_bytes(meta_buf[off..off + 4].try_into().unwrap()); off += 4;
        let eng_len = u16::from_le_bytes(meta_buf[off..off + 2].try_into().unwrap()) as usize; off += 2;
        if off + eng_len + 8 + 4 > meta_buf.len() { return Err("B run meta string/fields truncated".into()); }
        let eng = &meta_buf[off..off + eng_len]; off += eng_len;
        let start_time = u64::from_le_bytes(meta_buf[off..off + 8].try_into().unwrap()); off += 8;
        let elapsed_bits = u32::from_le_bytes(meta_buf[off..off + 4].try_into().unwrap()); off += 4;
        let elapsed_s = f32::from_bits(elapsed_bits);

        // Write adjusted
        let mut tmp = Vec::with_capacity(4 + 4 + 4 + 8 + 4 + 2 + eng_len + 8 + 4);
        let new_id = id.wrapping_add(ha.num_runs);
        tmp.extend_from_slice(&new_id.to_le_bytes());
        let new_first = first.wrapping_add(u32::try_from(ha.num_steps).unwrap());
        tmp.extend_from_slice(&new_first.to_le_bytes());
        tmp.extend_from_slice(&num_steps_run.to_le_bytes());
        tmp.extend_from_slice(&max_score.to_le_bytes());
        tmp.extend_from_slice(&highest_tile.to_le_bytes());
        tmp.extend_from_slice(&(eng_len as u16).to_le_bytes());
        tmp.extend_from_slice(eng);
        tmp.extend_from_slice(&start_time.to_le_bytes());
        tmp.extend_from_slice(&elapsed_bits.to_le_bytes());
        out.write_all(&tmp)?;
        hasher = crc32c_update(hasher, &tmp);
    }

    out.write_all(&hasher.to_le_bytes())?;
    out.flush()?;

    if output.exists() { fs::remove_file(output).ok(); }
    fs::rename(&out_tmp, output)?;
    eprintln!("Merged A({} runs, {} steps) + B({} runs, {} steps) -> {}", ha.num_runs, ha.num_steps, hb.num_runs, hb.num_steps, output.display());
    Ok(())
}
