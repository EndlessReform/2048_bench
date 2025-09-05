//! RAM-friendly dataset pack per docs/2048-pack.md
//!
//! Binary layout (little-endian):
//! - Header:
//!   magic: [u8;4] = b"2048"
//!   version: u32 = 1
//!   num_runs: u32
//!   num_steps: u64
//!   metadata_offset: u64  // absolute offset where run metadata starts
//! - Steps section: num_steps entries of 32 bytes each (Step)
//! - Run metadata section: repeated for num_runs (see RunMeta on-disk format)
//! - Trailer: checksum u32 = CRC32C of all preceding bytes
//!
//! This module provides `DataPack` with save/load and minimal derived indices.

use std::fs;
use std::io;
use std::mem::{size_of, align_of};
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;

const MAGIC: &[u8; 4] = b"2048";
const VERSION: u32 = 1;

#[derive(thiserror::Error, Debug)]
pub enum DataPackError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("malformed pack: {0}")]
    Malformed(&'static str),
    #[error("checksum mismatch")]
    Checksum,
    #[error("utf8 error")] // for engine string, we only write valid UTF-8; we don't surface the specific error
    Utf8,
    #[error("decode error: {0}")]
    Decode(String),
}

/// A single flattened step (32 bytes).
///
/// On-disk and in-memory layout (little-endian):
/// - `board`: packed 4x4 board as `u64`
/// - `run_id`: zero-based run identifier
/// - `index_in_run`: step index within the run
/// - `move_dir`: 0=Up, 1=Down, 2=Left, 3=Right
/// - `_padding`: reserved to keep size at 32 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub board: u64,
    pub run_id: u32,
    pub index_in_run: u32,
    pub move_dir: u8,       // 0..=3
    pub _padding: [u8; 15],
}

// Safety: Step is a plain-old-data type composed of integers and a fixed-size byte array.
// It has a stable #[repr(C)] layout, no references, and no invalid bit patterns.
unsafe impl Zeroable for Step {}
unsafe impl Pod for Step {}

/// Metadata describing an original run (kept in memory as Rust types).
///
/// The `engine` field is a UTF-8 string; empty means “unknown”.
#[derive(Clone, Debug, PartialEq)]
pub struct RunMeta {
    pub id: u32,
    pub first_step_idx: u32,
    pub num_steps: u32,
    pub max_score: u64,
    pub highest_tile: u32,
    pub engine: String,   // UTF-8, empty string means unknown
    pub start_time: u64,
    pub elapsed_s: f32,
}

/// In-RAM dataset for fast randomized access and filtering.
///
/// Typical usage:
/// ```
/// use ai_2048::serialization::{DataPack, PackBuilder};
/// use std::path::Path;
///
/// // Build from a directory of .a2run2 files and save
/// // let builder = PackBuilder::from_directory(Path::new("runs/"))?;
/// // builder.write_to_file("dataset.dat")?;
///
/// // Load a dataset pack into memory (parallel step decode by default)
/// // let dp = DataPack::load(Path::new("dataset.dat"))?;
/// // assert!(dp.steps.len() > 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct DataPack {
    pub steps: Vec<Step>,
    pub runs: Vec<RunMeta>,
    pub runs_by_score: Vec<(u64, u32)>,
    pub runs_by_length: Vec<(u32, u32)>,
}

impl DataPack {
    /// Build derived indices over run metadata. Call after constructing or loading.
    pub fn rebuild_indices(&mut self) {
        self.runs_by_score = self
            .runs
            .iter()
            .map(|r| (r.max_score, r.id))
            .collect();
        self.runs_by_score.sort_unstable();

        self.runs_by_length = self
            .runs
            .iter()
            .map(|r| (r.num_steps, r.id))
            .collect();
        self.runs_by_length.sort_unstable();
    }

    /// Save this DataPack to a file using the on-disk layout.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), DataPackError> {
        // Header fields
        let num_runs = self.runs.len() as u32;
        let num_steps = self.steps.len() as u64;

        // Serialize into a buffer to compute CRC once
        let mut buf: Vec<u8> = Vec::with_capacity(4 + 4 + 4 + 8 + 8 + self.steps.len() * size_of::<Step>() + 1024);

        // Write header (placeholder for metadata_offset to be filled properly)
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&num_runs.to_le_bytes());
        buf.extend_from_slice(&num_steps.to_le_bytes());

        // We'll compute metadata_offset = current header (4+4+4+8+8) + steps_bytes_len
        let metadata_offset_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder

        // Steps section: contiguous 32-byte entries
        // Safety: Step is Pod; we can cast slice of Steps to &[u8]
        let steps_bytes: &[u8] = bytemuck::cast_slice(&self.steps);
        buf.extend_from_slice(steps_bytes);

        // Patch metadata_offset now that steps_bytes length is known
        let metadata_offset = (4 + 4 + 4 + 8 + 8 + steps_bytes.len()) as u64;
        buf[metadata_offset_pos..metadata_offset_pos + 8]
            .copy_from_slice(&metadata_offset.to_le_bytes());

        // Run metadata section
        for r in &self.runs {
            buf.extend_from_slice(&r.id.to_le_bytes());
            buf.extend_from_slice(&r.first_step_idx.to_le_bytes());
            buf.extend_from_slice(&r.num_steps.to_le_bytes());
            buf.extend_from_slice(&r.max_score.to_le_bytes());
            buf.extend_from_slice(&r.highest_tile.to_le_bytes());
            let eng = r.engine.as_bytes();
            let eng_len = u16::try_from(eng.len()).map_err(|_| DataPackError::Malformed("engine too long"))?;
            buf.extend_from_slice(&eng_len.to_le_bytes());
            buf.extend_from_slice(eng);
            buf.extend_from_slice(&r.start_time.to_le_bytes());
            buf.extend_from_slice(&r.elapsed_s.to_bits().to_le_bytes());
        }

        // Trailer: CRC32C over all preceding bytes
        let crc = crc32c::crc32c(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        fs::write(path, buf)?;
        Ok(())
    }

    /// Load a DataPack from a file, validating header, bounds, and checksum.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, DataPackError> {
        let data = fs::read(path)?;
        if data.len() < 4 + 4 + 4 + 8 + 8 + 4 {
            return Err(DataPackError::Malformed("file too small"));
        }

        // Split trailer checksum
        let content_len = data.len() - 4;
        let file_crc = u32::from_le_bytes([
            data[content_len],
            data[content_len + 1],
            data[content_len + 2],
            data[content_len + 3],
        ]);
        let calc_crc = crc32c::crc32c(&data[..content_len]);
        if file_crc != calc_crc {
            return Err(DataPackError::Checksum);
        }

        let mut off = 0usize;
        if &data[off..off + 4] != MAGIC {
            return Err(DataPackError::Malformed("bad magic"));
        }
        off += 4;
        let version = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        if version != VERSION {
            return Err(DataPackError::Malformed("unsupported version"));
        }
        off += 4;
        let num_runs = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let num_steps = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
        off += 8;
        let metadata_offset = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
        off += 8;

        // Steps section bounds
        let steps_bytes_len = num_steps
            .checked_mul(size_of::<Step>())
            .ok_or(DataPackError::Malformed("steps size overflow"))?;
        if metadata_offset < off || content_len < metadata_offset {
            return Err(DataPackError::Malformed("metadata_offset out of range"));
        }
        let steps_region = &data[off..metadata_offset];
        if steps_region.len() != steps_bytes_len {
            return Err(DataPackError::Malformed("steps length mismatch"));
        }

        // Parallel decode of steps into a preallocated vector.
        debug_assert!(align_of::<Step>() <= 8);
        let mut steps: Vec<Step> = vec![Step { board: 0, run_id: 0, index_in_run: 0, move_dir: 0, _padding: [0; 15] }; num_steps];
        steps
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, out)| {
                let p = i * 32;
                // Field order (LE): board[8], run_id[4], index_in_run[4], move_dir[1], padding[15]
                let board = u64::from_le_bytes(steps_region[p..p + 8].try_into().unwrap());
                let run_id = u32::from_le_bytes(steps_region[p + 8..p + 12].try_into().unwrap());
                let index_in_run = u32::from_le_bytes(steps_region[p + 12..p + 16].try_into().unwrap());
                let move_dir = steps_region[p + 16];
                *out = Step { board, run_id, index_in_run, move_dir, _padding: [0; 15] };
            });

        // Parse run metadata
        let mut runs: Vec<RunMeta> = Vec::with_capacity(num_runs);
        let mut meta_off = metadata_offset;
        for i in 0..num_runs {
            // Fixed-size portion without engine string
            if meta_off + 4 + 4 + 4 + 8 + 4 + 2 > content_len {
                return Err(DataPackError::Malformed("run meta truncated"));
            }
            let id = u32::from_le_bytes(data[meta_off..meta_off + 4].try_into().unwrap());
            meta_off += 4;
            let first_step_idx = u32::from_le_bytes(data[meta_off..meta_off + 4].try_into().unwrap());
            meta_off += 4;
            let num_steps_run = u32::from_le_bytes(data[meta_off..meta_off + 4].try_into().unwrap());
            meta_off += 4;
            let max_score = u64::from_le_bytes(data[meta_off..meta_off + 8].try_into().unwrap());
            meta_off += 8;
            let highest_tile = u32::from_le_bytes(data[meta_off..meta_off + 4].try_into().unwrap());
            meta_off += 4;
            let eng_len = u16::from_le_bytes(data[meta_off..meta_off + 2].try_into().unwrap()) as usize;
            meta_off += 2;
            if meta_off + eng_len + 8 + 4 > content_len {
                return Err(DataPackError::Malformed("run meta string/fields truncated"));
            }
            let eng_bytes = &data[meta_off..meta_off + eng_len];
            // enforce UTF-8 (engine string); allow empty
            let engine = if eng_len == 0 {
                String::new()
            } else {
                match std::str::from_utf8(eng_bytes) {
                    Ok(s) => s.to_string(),
                    Err(_) => return Err(DataPackError::Utf8),
                }
            };
            meta_off += eng_len;
            let start_time = u64::from_le_bytes(data[meta_off..meta_off + 8].try_into().unwrap());
            meta_off += 8;
            let elapsed_s_bits = u32::from_le_bytes(data[meta_off..meta_off + 4].try_into().unwrap());
            let elapsed_s = f32::from_bits(elapsed_s_bits);
            meta_off += 4;

            runs.push(RunMeta {
                id,
                first_step_idx,
                num_steps: num_steps_run,
                max_score,
                highest_tile,
                engine,
                start_time,
                elapsed_s,
            });

            // Sanity checks on per-run step ranges
            let fsi = first_step_idx as usize;
            let n = num_steps_run as usize;
            if fsi > num_steps || n > num_steps || fsi + n > num_steps {
                return Err(DataPackError::Malformed("run step range oob"));
            }
            if (i as u32) != id {
                // For now, require ids to be 0..num_runs-1 for simplicity
                return Err(DataPackError::Malformed("run id mismatch"));
            }
        }

        // Ensure we consumed exactly up to content_len
        if meta_off != content_len {
            return Err(DataPackError::Malformed("trailing bytes before checksum"));
        }

        let mut pack = DataPack {
            steps,
            runs,
            runs_by_score: Vec::new(),
            runs_by_length: Vec::new(),
        };
        pack.rebuild_indices();
        Ok(pack)
    }
}

/// Builder that loads `.a2run2` files from a directory and assembles a `DataPack`.
///
/// - Skips legacy v1 files by design.
/// - Decoding runs is parallelized with Rayon.
pub struct PackBuilder {
    runs: Vec<crate::serialization::RunV2>,
    total_steps: usize,
}

impl PackBuilder {
    /// Load all `.a2run2` files under `dir` (recursive), in parallel.
    pub fn from_directory(dir: &Path) -> Result<Self, DataPackError> {
        if !dir.is_dir() {
            return Err(DataPackError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "input must be a directory",
            )));
        }
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        for e in walkdir::WalkDir::new(dir).into_iter().filter_map(Result::ok) {
            if e.file_type().is_file() {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) == Some("a2run2") {
                    files.push(p.to_path_buf());
                }
            }
        }
        // Stable order: path ascending
        files.sort();
        if files.is_empty() {
            return Err(DataPackError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "no .a2run2 files found",
            )));
        }

        // Parallel read/decode
        let runs: Result<Vec<_>, _> = files
            .par_iter()
            .map(|p| {
                let bytes = fs::read(p).map_err(DataPackError::Io)?;
                crate::serialization::from_postcard_bytes(&bytes)
                    .map_err(|e| DataPackError::Decode(e.to_string()))
            })
            .collect();
        let runs = runs?;
        let total_steps = runs.iter().map(|r| r.meta.steps as usize).sum();
        Ok(Self { runs, total_steps })
    }

    /// Assemble an in-memory `DataPack` (steps flattened across runs) and build indices.
    pub fn build(self) -> DataPack {
        let mut steps: Vec<Step> = Vec::with_capacity(self.total_steps);
        let mut runs_meta: Vec<RunMeta> = Vec::with_capacity(self.runs.len());
        let mut next_first = 0u32;
        for (run_id, run) in self.runs.into_iter().enumerate() {
            let rid = run_id as u32;
            let first_step_idx = next_first;
            for (i, s) in run.steps.iter().enumerate() {
                let dir = match s.chosen {
                    crate::engine::Move::Up => 0u8,
                    crate::engine::Move::Down => 1u8,
                    crate::engine::Move::Left => 2u8,
                    crate::engine::Move::Right => 3u8,
                };
                steps.push(Step {
                    board: s.pre_board,
                    run_id: rid,
                    index_in_run: i as u32,
                    move_dir: dir,
                    _padding: [0; 15],
                });
            }
            let num_steps = run.meta.steps;
            runs_meta.push(RunMeta {
                id: rid,
                first_step_idx,
                num_steps,
                max_score: run.meta.max_score,
                highest_tile: run.meta.highest_tile,
                engine: run.meta.engine_str.clone().unwrap_or_default(),
                start_time: run.meta.start_unix_s,
                elapsed_s: run.meta.elapsed_s,
            });
            next_first = next_first.saturating_add(num_steps);
        }
        let mut dp = DataPack { steps, runs: runs_meta, runs_by_score: vec![], runs_by_length: vec![] };
        dp.rebuild_indices();
        dp
    }

    /// Convenience: build and write to a file.
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), DataPackError> {
        let cloned = Self { runs: self.runs.clone(), total_steps: self.total_steps };
        let dp = cloned.build();
        dp.save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn mk_step(run_id: u32, idx: u32, board: u64, dir: u8) -> Step {
        Step { board, move_dir: dir, run_id, index_in_run: idx, _padding: [0; 15] }
    }

    #[test]
    fn layout_and_sizes() {
        assert_eq!(size_of::<Step>(), 32);
    }

    #[test]
    fn roundtrip_small() {
        let steps = vec![
            mk_step(0, 0, 0x1000_0000_0000_0000, 0),
            mk_step(0, 1, 0x1100_0000_0000_0000, 3),
            mk_step(1, 0, 0x2000_0000_0000_0000, 2),
        ];
        let runs = vec![
            RunMeta { id: 0, first_step_idx: 0, num_steps: 2, max_score: 4096, highest_tile: 512, engine: "e1".into(), start_time: 1_700_000_000, elapsed_s: 2.5 },
            RunMeta { id: 1, first_step_idx: 2, num_steps: 1, max_score: 1024, highest_tile: 128, engine: String::new(), start_time: 1_700_000_100, elapsed_s: 1.0 },
        ];
        let mut dp = DataPack { steps, runs, runs_by_score: vec![], runs_by_length: vec![] };
        dp.rebuild_indices();

        let tmp = NamedTempFile::new().unwrap();
        dp.save(tmp.path()).unwrap();
        let loaded = DataPack::load(tmp.path()).unwrap();

        assert_eq!(loaded.steps.len(), 3);
        assert_eq!(loaded.runs.len(), 2);
        assert_eq!(loaded.steps[0].board, 0x1000_0000_0000_0000);
        assert_eq!(loaded.steps[1].move_dir, 3);
        assert_eq!(loaded.steps[2].run_id, 1);
        assert_eq!(loaded.runs[0].engine, "e1");
        assert_eq!(loaded.runs[1].engine, "");
        assert_eq!(loaded.runs_by_score.len(), 2);
        assert_eq!(loaded.runs_by_length.len(), 2);
    }

    #[test]
    fn checksum_and_bounds() {
        let steps = vec![mk_step(0, 0, 1, 0)];
        let runs = vec![RunMeta { id: 0, first_step_idx: 0, num_steps: 1, max_score: 0, highest_tile: 2, engine: String::new(), start_time: 0, elapsed_s: 0.0 }];
        let dp = DataPack { steps, runs, runs_by_score: vec![], runs_by_length: vec![] };
        let tmp = NamedTempFile::new().unwrap();
        dp.save(tmp.path()).unwrap();
        let mut bytes = fs::read(tmp.path()).unwrap();
        // Corrupt a byte in steps region
        let off = 4 + 4 + 4 + 8 + 8; // up to metadata_offset
        bytes[off] ^= 0xFF;
        fs::write(tmp.path(), &bytes).unwrap();
        let err = DataPack::load(tmp.path()).unwrap_err();
        matches!(err, DataPackError::Checksum);
    }
}
