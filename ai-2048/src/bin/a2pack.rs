use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use ai_2048::serialization::PackReader;
use clap::{Parser, Subcommand};
use crc32c::crc32c;
use walkdir::WalkDir;

const MAGIC: &[u8; 8] = b"A2PACK\0\0";
const VERSION: u32 = 1;

#[derive(Parser, Debug)]
#[command(
    name = "a2pack",
    version,
    about = "Pack a directory of a2run2 files into a readonly packfile"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Pack a directory of .a2run2 files into a .a2pack
    Pack {
        /// Input directory to scan recursively
        #[arg(long, value_name = "DIR")]
        input: PathBuf,
        /// Output packfile path
        #[arg(long, value_name = "PACKFILE")]
        output: PathBuf,
        /// Align data region and entries to this many bytes (default 4096)
        #[arg(long, value_name = "BYTES", default_value_t = 4096)]
        align: usize,
        /// Include per-entry CRC32C for payload verification
        #[arg(long, default_value_t = true)]
        entry_crc: bool,
    },
    /// Validate a packfile (structure and checksums)
    Validate {
        /// Packfile path
        #[arg(long, value_name = "PACKFILE")]
        packfile: PathBuf,
    },
    /// Print summary statistics for a packfile
    Stats {
        /// Packfile path
        #[arg(long, value_name = "PACKFILE")]
        packfile: PathBuf,
    },
    /// Export all runs to JSONL via the fast Rust path
    ToJsonl {
        /// Packfile path
        #[arg(long, value_name = "PACKFILE")]
        packfile: PathBuf,
        /// Output JSONL file
        #[arg(long, value_name = "FILE")]
        output: PathBuf,
        /// Decode/serialize in parallel
        #[arg(long, default_value_t = true)]
        parallel: bool,
    },
    /// Extract specific runs by index
    Extract {
        /// Packfile path
        #[arg(long, value_name = "PACKFILE")]
        packfile: PathBuf,
        /// Comma-separated indices, e.g., 0,5,42
        #[arg(long, value_name = "LIST")]
        indices: String,
        /// Output directory to write .a2run2 files
        #[arg(long, value_name = "DIR")]
        output: PathBuf,
    },
    /// Inspect a single run: print decoded summary
    Inspect {
        /// Packfile path
        #[arg(long, value_name = "PACKFILE")]
        packfile: PathBuf,
        /// Index of the run to inspect
        #[arg(long, value_name = "IDX")]
        index: usize,
    },
}

#[derive(Clone, Debug)]
struct EntryMeta {
    path: PathBuf,
    len: u32,
    kind: u16,   // 1=v1, 2=v2
    flags: u16,  // reserved
    crc: u32,    // 0 if disabled
    offset: u64, // to be filled after layout
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Pack {
            input,
            output,
            align,
            entry_crc,
        } => {
            pack_dir(&input, &output, align, entry_crc)?;
        }
        Command::Validate { packfile } => {
            match PackReader::open(&packfile) {
                Ok(r) => {
                    eprintln!("OK: {} ({} runs)", packfile.display(), r.len());
                }
                Err(e) => {
                    eprintln!("INVALID: {} ({})", packfile.display(), e);
                    std::process::exit(2);
                }
            }
        }
        Command::Stats { packfile } => {
            let r = PackReader::open(&packfile)?;
            let s = r.stats()?;
            println!("packfile: {}", packfile.display());
            println!("runs: {}", s.count);
            println!("total_steps: {}", s.total_steps);
            println!("min_len: {}", s.min_len);
            println!("max_len: {}", s.max_len);
            println!("mean_len: {:.3}", s.mean_len);
        }
        Command::ToJsonl { packfile, output, parallel } => {
            let r = PackReader::open(&packfile)?;
            r.to_jsonl(&output, parallel)?;
            eprintln!("Wrote JSONL to {}", output.display());
        }
        Command::Extract { packfile, indices, output } => {
            let r = PackReader::open(&packfile)?;
            std::fs::create_dir_all(&output)?;
            let idxs = parse_indices(&indices)?;
            for i in idxs {
                let bytes = r.get_slice(i).map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                let out = output.join(format!("run-{:06}.a2run2", i));
                let mut f = File::create(&out)?;
                f.write_all(bytes)?;
            }
            eprintln!("Extracted indices [{}] to {}", indices, output.display());
        }
        Command::Inspect { packfile, index } => {
            let r = PackReader::open(&packfile)?;
            let run = r.decode_auto_v2(index).map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            println!("packfile: {}", packfile.display());
            println!("index: {}", index);
            println!("steps: {}", run.meta.steps);
            println!("max_score: {}", run.meta.max_score);
            println!("highest_tile: {}", run.meta.highest_tile);
            println!("start_unix_s: {}", run.meta.start_unix_s);
            println!("elapsed_s: {:.3}", run.meta.elapsed_s);
            println!("final_board: {:#018x}", run.final_board);
        }
    }
    Ok(())
}

fn pack_dir(
    input: &Path,
    output: &Path,
    align: usize,
    entry_crc: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !input.is_dir() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "input must be a directory",
        )));
    }

    // Pass 1: discover and collect metadata
    let mut entries: Vec<EntryMeta> = Vec::new();
    for e in WalkDir::new(input)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let p = e.path();
        if let Some(ext) = p.extension() {
            if ext != "a2run2" {
                continue;
            }
        } else {
            continue;
        }

        let (len, kind, crc) = compute_len_kind_crc(p, entry_crc)?;
        entries.push(EntryMeta {
            path: p.to_path_buf(),
            len,
            kind,
            flags: 0,
            crc,
            offset: 0,
        });
    }

    // Stable order: path ascending (deterministic)
    entries.sort_by(|a, b| a.path.cmp(&b.path));

    let count = entries.len();
    if count == 0 {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no .a2run2 files found",
        )));
    }

    // Compute layout
    let header_len = 28usize; // magic+ver+flags+count + header crc
    let index_len = count * 32;
    let mut data_start = header_len + index_len;
    if data_start % align != 0 {
        data_start += align - (data_start % align);
    }

    let mut cur = data_start;
    for ent in entries.iter_mut() {
        if cur % align != 0 {
            cur += align - (cur % align);
        }
        ent.offset = cur as u64;
        cur += ent.len as usize;
    }
    let data_end = cur;

    // Create and write file
    let mut out = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(output)?,
    );

    // Header
    out.write_all(MAGIC)?;
    out.write_all(&VERSION.to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())?; // flags
    out.write_all(&(count as u64).to_le_bytes())?;
    let mut header_buf = Vec::with_capacity(24);
    header_buf.extend_from_slice(MAGIC);
    header_buf.extend_from_slice(&VERSION.to_le_bytes());
    header_buf.extend_from_slice(&0u32.to_le_bytes());
    header_buf.extend_from_slice(&(count as u64).to_le_bytes());
    let header_crc = crc32c(&header_buf);
    out.write_all(&header_crc.to_le_bytes())?; // header crc

    // Index
    let mut index_bytes: Vec<u8> = Vec::with_capacity(index_len);
    for ent in &entries {
        index_bytes.extend_from_slice(&ent.offset.to_le_bytes());
        index_bytes.extend_from_slice(&ent.len.to_le_bytes());
        index_bytes.extend_from_slice(&ent.kind.to_le_bytes());
        index_bytes.extend_from_slice(&ent.flags.to_le_bytes());
        index_bytes.extend_from_slice(&ent.crc.to_le_bytes());
        index_bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved
        index_bytes.extend_from_slice(&[0u8; 8]); // pad to 32
    }
    out.write_all(&index_bytes)?;

    // Pad to alignment before data region
    let mut written = header_len + index_len;
    if written % align != 0 {
        let pad = align - (written % align);
        out.write_all(&vec![0u8; pad])?;
        written += pad;
    }

    // Data region: write entries in the same order
    for ent in &entries {
        let needed = ent.offset as usize - written;
        if needed > 0 {
            out.write_all(&vec![0u8; needed])?;
            written += needed;
        }
        copy_file(&ent.path, &mut out)?;
        written += ent.len as usize;
    }

    // Footer (fixed 32 bytes, with 4-byte pad at end)
    let index_crc = crc32c(&index_bytes);
    out.write_all(&(data_end as u64).to_le_bytes())?;
    out.write_all(&index_crc.to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())?; // data crc (optional, 0)
    out.write_all(&0u64.to_le_bytes())?; // stats offset (0 = none)
                                         // footer CRC over the previous 24 bytes
    let mut footer_tmp = Vec::with_capacity(24);
    footer_tmp.extend_from_slice(&(data_end as u64).to_le_bytes());
    footer_tmp.extend_from_slice(&index_crc.to_le_bytes());
    footer_tmp.extend_from_slice(&0u32.to_le_bytes());
    footer_tmp.extend_from_slice(&0u64.to_le_bytes());
    let footer_crc = crc32c(&footer_tmp);
    out.write_all(&footer_crc.to_le_bytes())?;
    out.write_all(&0u32.to_le_bytes())?; // pad to 32 bytes

    out.flush()?;
    eprintln!("Packed {} runs into {}", count, output.display());
    Ok(())
}

fn compute_len_kind_crc(path: &Path, entry_crc: bool) -> io::Result<(u32, u16, u32)> {
    let mut f = File::open(path)?;
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];
    // Try read first 6 bytes (magic+version?) for v1 detection
    let mut head = [0u8; 6];
    let mut read = 0usize;
    while read < head.len() {
        let n = f.read(&mut head[read..])?;
        if n == 0 {
            break;
        }
        read += n;
    }

    // Determine kind by v1 magic/version, else treat as v2
    let kind = if read >= 6 && &head[0..4] == b"A2T1" && head[4] == 1 {
        1u16
    } else {
        2u16
    };

    // Compute len and crc (optional) by streaming.
    let mut hasher: u32 = 0;
    let mut total: u64 = read as u64;
    if entry_crc {
        hasher = crc32c(&head[..read]);
    }
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if entry_crc {
            hasher = crc32c::crc32c_append(hasher, &buf[..n]);
        }
    }

    let len_u32 = u32::try_from(total)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "file too large"))?;
    let crc = if entry_crc { hasher } else { 0 };
    Ok((len_u32, kind, crc))
}

fn copy_file(src: &Path, out: &mut dyn Write) -> io::Result<()> {
    let mut f = File::open(src)?;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
    }
    Ok(())
}

fn parse_indices(s: &str) -> io::Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() { continue; }
        match part.parse::<usize>() {
            Ok(v) => out.push(v),
            Err(_) => return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("bad index: {}", part))),
        }
    }
    Ok(out)
}
