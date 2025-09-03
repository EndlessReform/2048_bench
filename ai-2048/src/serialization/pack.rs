use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crc32c::crc32c;
use memmap2::Mmap;
use rayon::prelude::*;

use crate::serialization::{self as ser, RunV2};
use crate::trace;

const MAGIC: &[u8; 8] = b"A2PACK\0\0";
const VERSION: u32 = 1;

#[derive(thiserror::Error, Debug)]
pub enum PackError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("malformed packfile: {0}")]
    Malformed(&'static str),
    #[error("checksum mismatch: {0}")]
    Checksum(&'static str),
    #[error("index out of bounds")]
    Oob,
    #[error("decode error: {0}")]
    Decode(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunKind {
    V1,
    V2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct IndexEntry {
    offset: u64,
    length: u32,
    kind: u16,
    flags: u16,
    crc32c: u32,
    reserved: u32,
}

impl IndexEntry {
    fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < 32 {
            return None;
        }
        let offset = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        let length = u32::from_le_bytes([b[8], b[9], b[10], b[11]]);
        let kind = u16::from_le_bytes([b[12], b[13]]);
        let flags = u16::from_le_bytes([b[14], b[15]]);
        let crc32c = u32::from_le_bytes([b[16], b[17], b[18], b[19]]);
        let reserved = u32::from_le_bytes([b[20], b[21], b[22], b[23]]);
        let _pad = &b[24..32];
        Some(Self {
            offset,
            length,
            kind,
            flags,
            crc32c,
            reserved,
        })
    }
}

pub struct PackReader {
    mmap: Mmap,
    index_offset: usize,
    index_len: usize, // number of entries
    data_region_start: usize,
    data_region_end: usize,
    stats_offset: Option<usize>,
}

impl PackReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, PackError> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        let bytes = &mmap[..];
        if bytes.len() < 8 + 4 + 4 + 8 + 4 {
            return Err(PackError::Malformed("file too small"));
        }
        if &bytes[0..8] != MAGIC {
            return Err(PackError::Malformed("bad magic"));
        }
        let ver = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if ver != VERSION {
            return Err(PackError::Malformed("unsupported version"));
        }
        // flags
        let _flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let count = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        let header_crc = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let calc_crc = crc32c(&bytes[0..24]);
        if header_crc != calc_crc {
            return Err(PackError::Checksum("header"));
        }

        let index_offset = 28; // header size
        let index_bytes_len = count
            .checked_mul(32)
            .ok_or(PackError::Malformed("index size overflow"))?;
        let after_index = index_offset + index_bytes_len;
        if bytes.len() < after_index + 32 {
            return Err(PackError::Malformed("missing footer"));
        }

        // Footer is the last 32 bytes
        let footer_off = bytes.len() - 32;
        let data_end =
            u64::from_le_bytes(bytes[footer_off..footer_off + 8].try_into().unwrap()) as usize;
        let index_crc =
            u32::from_le_bytes(bytes[footer_off + 8..footer_off + 12].try_into().unwrap());
        let _data_crc =
            u32::from_le_bytes(bytes[footer_off + 12..footer_off + 16].try_into().unwrap());
        let stats_off_u64 =
            u64::from_le_bytes(bytes[footer_off + 16..footer_off + 24].try_into().unwrap());
        let stats_offset = if stats_off_u64 == 0 {
            None
        } else {
            Some(stats_off_u64 as usize)
        };
        let footer_crc =
            u32::from_le_bytes(bytes[footer_off + 24..footer_off + 28].try_into().unwrap());
        let calc_footer_crc = crc32c(&bytes[footer_off..footer_off + 24]);
        if footer_crc != calc_footer_crc {
            return Err(PackError::Checksum("footer"));
        }

        // Validate index CRC
        let index_slice = &bytes[index_offset..index_offset + index_bytes_len];
        let calc_index_crc = crc32c(index_slice);
        if index_crc != calc_index_crc {
            return Err(PackError::Checksum("index"));
        }

        // Determine data region start/end from index entries (min/max offsets)
        let mut min_off = usize::MAX;
        let mut max_end = 0usize;
        for i in 0..count {
            let ent = IndexEntry::from_bytes(&index_slice[i * 32..(i + 1) * 32])
                .ok_or(PackError::Malformed("bad index entry"))?;
            let off = ent.offset as usize;
            let len = ent.length as usize;
            min_off = min_off.min(off);
            max_end = max_end.max(off + len);
        }
        if min_off < after_index || max_end > footer_off {
            return Err(PackError::Malformed("data offsets out of range"));
        }
        if data_end != max_end {
            return Err(PackError::Malformed("data_end mismatch"));
        }

        Ok(Self {
            mmap,
            index_offset,
            index_len: count,
            data_region_start: min_off,
            data_region_end: max_end,
            stats_offset,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.index_len
    }

    #[inline]
    fn index_bytes(&self) -> &[u8] {
        &self.mmap[self.index_offset..self.index_offset + self.index_len * 32]
    }

    fn entry(&self, i: usize) -> Result<IndexEntry, PackError> {
        if i >= self.index_len {
            return Err(PackError::Oob);
        }
        IndexEntry::from_bytes(&self.index_bytes()[i * 32..(i + 1) * 32])
            .ok_or(PackError::Malformed("bad index entry"))
    }

    pub fn kind(&self, i: usize) -> Result<RunKind, PackError> {
        let ent = self.entry(i)?;
        Ok(match ent.kind {
            1 => RunKind::V1,
            2 => RunKind::V2,
            _ => return Err(PackError::Malformed("unknown kind")),
        })
    }

    pub fn get_slice(&self, i: usize) -> Result<&[u8], PackError> {
        let ent = self.entry(i)?;
        let off = ent.offset as usize;
        let len = ent.length as usize;
        let end = off
            .checked_add(len)
            .ok_or(PackError::Malformed("slice overflow"))?;
        if off < self.data_region_start || end > self.data_region_end {
            return Err(PackError::Malformed("slice oob"));
        }
        let slice = &self.mmap[off..end];
        if ent.crc32c != 0 {
            let calc = crc32c(slice);
            if calc != ent.crc32c {
                return Err(PackError::Checksum("entry"));
            }
        }
        Ok(slice)
    }

    pub fn decode_v2(&self, i: usize) -> Result<RunV2, PackError> {
        let bytes = self.get_slice(i)?;
        ser::from_postcard_bytes(bytes).map_err(|e| PackError::Decode(e.to_string()))
    }

    pub fn decode_v1(&self, i: usize) -> Result<trace::Run, PackError> {
        let bytes = self.get_slice(i)?;
        trace::parse_run_bytes(bytes).map_err(|e| PackError::Decode(e.to_string()))
    }

    pub fn decode_auto_v2(&self, i: usize) -> Result<RunV2, PackError> {
        match self.kind(i)? {
            RunKind::V2 => self.decode_v2(i),
            RunKind::V1 => {
                let v1 = self.decode_v1(i)?;
                Ok(ser::from_v1(v1))
            }
        }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = Result<RunV2, PackError>> + 'a {
        (0..self.len()).map(move |i| self.decode_auto_v2(i))
    }

    pub fn iter_indices<'a>(
        &'a self,
        idxs: impl IntoIterator<Item = usize> + 'a,
    ) -> impl Iterator<Item = Result<RunV2, PackError>> + 'a {
        idxs.into_iter().map(move |i| self.decode_auto_v2(i))
    }

    pub fn decode_batch_auto_v2(&self, idxs: &[usize]) -> Result<Vec<RunV2>, PackError> {
        let res: Result<Vec<_>, _> = idxs.par_iter().map(|&i| self.decode_auto_v2(i)).collect();
        res
    }

    /// Export the packfile to a JSONL file.
    /// If `by_step` is false, writes one line per run; if true, writes one line per step (flattened).
    pub fn to_jsonl<P: AsRef<Path>>(&self, out: P, parallel: bool, by_step: bool) -> Result<(), PackError> {
        let mut f = File::create(out)?;
        let indices: Vec<usize> = (0..self.len()).collect();
        if parallel {
            let chunks = indices
                .par_chunks(256)
                .map(|chunk| {
                    let mut buf: Vec<u8> = Vec::with_capacity(256 * 256);
                    for &i in chunk {
                        if by_step {
                            let slice = self.get_slice(i)?;
                            let run = self.decode_auto_v2(i)?;
                            let run_uuid = compute_run_uuid(i as u64, slice);
                            write_step_jsonl_lines(&mut buf, i as u64, &run_uuid, &run)
                                .map_err(|e| PackError::Decode(e.to_string()))?;
                        } else {
                            let run = self.decode_auto_v2(i)?;
                            write_run_jsonl_line(&mut buf, i as u64, &run)
                                .map_err(|e| PackError::Decode(e.to_string()))?;
                        }
                    }
                    Ok::<Vec<u8>, PackError>(buf)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for s in chunks {
                f.write_all(&s)?;
            }
        } else {
            let mut buf: Vec<u8> = Vec::with_capacity(256 * 256);
            for i in indices {
                if by_step {
                    let slice = self.get_slice(i)?;
                    let run = self.decode_auto_v2(i)?;
                    let run_uuid = compute_run_uuid(i as u64, slice);
                    write_step_jsonl_lines(&mut buf, i as u64, &run_uuid, &run)
                        .map_err(|e| PackError::Decode(e.to_string()))?;
                } else {
                    let run = self.decode_auto_v2(i)?;
                    write_run_jsonl_line(&mut buf, i as u64, &run)
                        .map_err(|e| PackError::Decode(e.to_string()))?;
                }
                if buf.len() > 1_000_000 {
                    f.write_all(&buf)?;
                    buf.clear();
                }
            }
            if !buf.is_empty() {
                f.write_all(&buf)?;
            }
        }
        Ok(())
    }

    pub fn stats(&self) -> Result<PackStats, PackError> {
        // Prefer reading cached stats block if present
        if let Some(off) = self.stats_offset {
            if off + 4 <= self.mmap.len() {
                let len = u32::from_le_bytes(self.mmap[off..off + 4].try_into().unwrap()) as usize;
                let start = off + 4;
                let end = start.saturating_add(len);
                if end <= self.mmap.len() {
                    let payload = &self.mmap[start..end];
                    if let Ok(sb) = postcard::from_bytes::<StatsBlock>(payload) {
                        return Ok(PackStats {
                            count: sb.count,
                            total_steps: sb.total_steps,
                            min_len: sb.min_len,
                            max_len: sb.max_len,
                            mean_len: sb.mean_len,
                            p50: sb.p50,
                            p90: sb.p90,
                            p99: sb.p99,
                        });
                    }
                }
            }
        }

        // Fallback: compute lazily in a single pass (parallel) and aggregate
        let indices: Vec<usize> = (0..self.len()).collect();
        let init = PartialStats {
            count: 0,
            total_steps: 0,
            min_len: u32::MAX,
            max_len: 0,
        };
        let reduced = indices
            .par_chunks(256)
            .map(|chunk| {
                let mut ps = init.clone();
                for &i in chunk {
                    let run = self.decode_auto_v2(i)?;
                    let steps = run.meta.steps;
                    ps.count += 1;
                    ps.total_steps += steps as u64;
                    ps.min_len = ps.min_len.min(steps);
                    ps.max_len = ps.max_len.max(steps);
                }
                Ok::<PartialStats, PackError>(ps)
            })
            .try_reduce(|| init.clone(), |a, b| Ok(a.merge(&b)))?;

        let mean_len = if reduced.count == 0 {
            0.0
        } else {
            reduced.total_steps as f64 / reduced.count as f64
        };
        Ok(PackStats {
            count: reduced.count as u64,
            total_steps: reduced.total_steps,
            min_len: if reduced.count == 0 {
                0
            } else {
                reduced.min_len
            },
            max_len: reduced.max_len,
            mean_len,
            p50: 0,
            p90: 0,
            p99: 0,
        })
    }
}

#[derive(Clone)]
struct PartialStats {
    count: u64,
    total_steps: u64,
    min_len: u32,
    max_len: u32,
}
impl PartialStats {
    fn merge(&self, o: &Self) -> Self {
        Self {
            count: self.count + o.count,
            total_steps: self.total_steps + o.total_steps,
            min_len: self.min_len.min(o.min_len),
            max_len: self.max_len.max(o.max_len),
        }
    }
}

pub struct PackStats {
    pub count: u64,
    pub total_steps: u64,
    pub min_len: u32,
    pub max_len: u32,
    pub mean_len: f64,
    pub p50: u32,
    pub p90: u32,
    pub p99: u32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StatsBlock {
    count: u64,
    total_steps: u64,
    min_len: u32,
    max_len: u32,
    mean_len: f64,
    p50: u32,
    p90: u32,
    p99: u32,
}

#[derive(serde::Serialize)]
struct RunJson<'a> {
    run_idx: u64,
    steps: u32,
    start_unix_s: u64,
    elapsed_s: f32,
    max_score: u64,
    highest_tile: u32,
    final_board: u64,
    states: &'a [u64],
    moves: &'a [&'static str],
}

fn write_run_jsonl_line(
    buf: &mut Vec<u8>,
    run_idx: u64,
    run: &RunV2,
) -> Result<(), serde_json::Error> {
    let m = &run.meta;
    // reconstruct states and moves
    let mut states = Vec::with_capacity(run.steps.len() + 1);
    for s in &run.steps {
        states.push(s.pre_board);
    }
    states.push(run.final_board);
    let mut moves = Vec::with_capacity(run.steps.len());
    for s in &run.steps {
        let dir = match s.chosen {
            crate::engine::Move::Up => "UP",
            crate::engine::Move::Down => "DOWN",
            crate::engine::Move::Left => "LEFT",
            crate::engine::Move::Right => "RIGHT",
        };
        moves.push(dir);
    }

    let rec = RunJson {
        run_idx,
        steps: m.steps,
        start_unix_s: m.start_unix_s,
        elapsed_s: m.elapsed_s,
        max_score: m.max_score,
        highest_tile: m.highest_tile,
        final_board: run.final_board,
        states: &states,
        moves: &moves,
    };
    serde_json::to_writer(&mut *buf, &rec)?;
    buf.push(b'\n');
    Ok(())
}

#[derive(serde::Serialize)]
struct BranchJson { legal: bool, value: f32 }

#[derive(serde::Serialize)]
struct StepJson<'a> {
    run_idx: u64,
    run_uuid: &'a str,
    step_idx: u32,
    // optional run context for convenience
    steps: u32,
    start_unix_s: u64,
    // step payload
    pre_board: u64,
    chosen: &'static str,
    branches: [BranchJson; 4], // Up, Down, Left, Right
}

fn compute_run_uuid(run_idx: u64, slice: &[u8]) -> String {
    // Stable identifier based on content + index; not a formal UUID but globally stable for the pack.
    // Format: a2r-<crc32c(slice)>-<run_idx>
    let crc = crc32c(slice);
    format!("a2r-{:08x}-{:08x}", crc, run_idx as u32)
}

fn write_step_jsonl_lines(
    buf: &mut Vec<u8>,
    run_idx: u64,
    run_uuid: &str,
    run: &RunV2,
) -> Result<(), serde_json::Error> {
    let m = &run.meta;
    for (si, s) in run.steps.iter().enumerate() {
        let chosen = match s.chosen {
            crate::engine::Move::Up => "UP",
            crate::engine::Move::Down => "DOWN",
            crate::engine::Move::Left => "LEFT",
            crate::engine::Move::Right => "RIGHT",
        };
        let branches_arr = if let Some(arr) = s.branches {
            arr
        } else {
            let mut tmp = [crate::serialization::BranchV2::Illegal; 4];
            let idx = match s.chosen { crate::engine::Move::Up => 0, crate::engine::Move::Down => 1, crate::engine::Move::Left => 2, crate::engine::Move::Right => 3 };
            tmp[idx] = crate::serialization::BranchV2::Legal(1.0);
            tmp
        };
        let branches = branches_arr.map(|b| match b {
            crate::serialization::BranchV2::Illegal => BranchJson { legal: false, value: 0.0 },
            crate::serialization::BranchV2::Legal(v) => BranchJson { legal: true, value: v },
        });
        let rec = StepJson {
            run_idx,
            run_uuid,
            step_idx: si as u32,
            steps: m.steps,
            start_unix_s: m.start_unix_s,
            pre_board: s.pre_board,
            chosen,
            branches,
        };
        serde_json::to_writer(&mut *buf, &rec)?;
        buf.push(b'\n');
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn build_v1_bytes(steps: u32) -> Vec<u8> {
        let mut states = Vec::with_capacity(steps as usize + 1);
        let mut moves = Vec::with_capacity(steps as usize);
        let mut s = 0x1000_0000_0000_0000u64;
        states.push(s);
        for i in 0..steps {
            s = s.wrapping_add(1);
            states.push(s);
            moves.push((i % 4) as u8);
        }
        let meta = trace::Meta {
            steps,
            start_unix_s: 1_700_000_000,
            elapsed_s: 1.0,
            max_score: 1000,
            highest_tile: 128,
            engine_str: None,
        };
        trace::encode_run(&meta, &states, &moves)
    }

    fn build_v2_bytes(steps: u32) -> Vec<u8> {
        let mut vsteps = Vec::new();
        for i in 0..steps {
            let pre = 0x2000_0000_0000_0000u64 + i as u64;
            let chosen = match i % 4 {
                0 => crate::engine::Move::Up,
                1 => crate::engine::Move::Down,
                2 => crate::engine::Move::Left,
                _ => crate::engine::Move::Right,
            };
            vsteps.push(ser::StepV2 {
                pre_board: pre,
                chosen,
                branches: None,
            });
        }
        let meta = trace::Meta {
            steps,
            start_unix_s: 1_700_000_100,
            elapsed_s: 2.5,
            max_score: 2048,
            highest_tile: 256,
            engine_str: None,
        };
        let run = RunV2 {
            meta,
            steps: vsteps,
            final_board: 0xDEAD_BEEF_DEAD_BEEFu64,
        };
        ser::to_postcard_bytes(&run).unwrap()
    }

    fn write_pack(v1s: &[Vec<u8>], v2s: &[Vec<u8>]) -> NamedTempFile {
        let mut entries: Vec<(u64, u32, u16, u16, u32, u32, Vec<u8>)> = Vec::new();
        let align = 4096usize;
        let mut offset = 0usize; // will fill after header+index size known
        let count = v1s.len() + v2s.len();
        let mut tmp = NamedTempFile::new().unwrap();

        // Reserve header + index + footer space in buffer logic by writing later.
        // We'll assemble into a Vec then write to file for clarity.
        let mut data_region: Vec<u8> = Vec::new();

        // Prepare entries (we will compute offsets after we know index size)
        for b in v1s {
            entries.push((0, b.len() as u32, 1, 0, crc32c(b), 0, b.clone()));
        }
        for b in v2s {
            entries.push((0, b.len() as u32, 2, 0, crc32c(b), 0, b.clone()));
        }

        // Layout calculations
        let header_len = 28usize;
        let index_len = count * 32;
        offset = header_len + index_len;
        // Align data start
        if offset % align != 0 {
            offset += align - (offset % align);
        }

        let mut cur = offset;
        for e in entries.iter_mut() {
            // align each entry to 4k
            if cur % align != 0 {
                cur += align - (cur % align);
            }
            e.0 = cur as u64; // set offset
            cur += e.1 as usize;
        }
        let data_end = cur;

        // Build file bytes
        let mut file: Vec<u8> = Vec::with_capacity(data_end + 32);
        // Header
        file.extend_from_slice(MAGIC);
        file.extend_from_slice(&VERSION.to_le_bytes());
        file.extend_from_slice(&0u32.to_le_bytes()); // flags
        file.extend_from_slice(&(count as u64).to_le_bytes());
        let crc = crc32c(&file[0..24]);
        file.extend_from_slice(&crc.to_le_bytes());

        // Index
        for e in &entries {
            file.extend_from_slice(&e.0.to_le_bytes());
            file.extend_from_slice(&e.1.to_le_bytes());
            file.extend_from_slice(&e.2.to_le_bytes());
            file.extend_from_slice(&e.3.to_le_bytes());
            file.extend_from_slice(&e.4.to_le_bytes());
            file.extend_from_slice(&e.5.to_le_bytes());
            file.extend_from_slice(&[0u8; 8]); // pad to 32 bytes
        }
        // Pad to alignment
        if file.len() % align != 0 {
            let pad = align - (file.len() % align);
            file.extend(std::iter::repeat(0u8).take(pad));
        }

        // Data region
        let current_len = file.len();
        assert_eq!(current_len, offset);
        for e in &entries {
            let needed = e.0 as usize - file.len();
            if needed > 0 {
                file.extend(std::iter::repeat(0u8).take(needed));
            }
            file.extend_from_slice(&e.6);
        }

        // Footer
        let index_crc = crc32c(&file[28..(28 + index_len)]);
        file.extend_from_slice(&(data_end as u64).to_le_bytes());
        file.extend_from_slice(&index_crc.to_le_bytes());
        file.extend_from_slice(&0u32.to_le_bytes()); // data crc optional
        file.extend_from_slice(&0u64.to_le_bytes()); // stats offset
        let footer_crc = crc32c(&file[file.len() - 24..]);
        file.extend_from_slice(&footer_crc.to_le_bytes());
        file.extend_from_slice(&0u32.to_le_bytes()); // pad to 32 bytes fixed footer

        // Write to temp file
        tmp.write_all(&file).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    #[test]
    fn open_and_len() {
        let v1 = build_v1_bytes(3);
        let v2 = build_v2_bytes(4);
        let tmp = write_pack(&[v1], &[v2]);
        let reader = PackReader::open(tmp.path()).unwrap();
        assert_eq!(reader.len(), 2);
        assert_eq!(reader.kind(0).unwrap(), RunKind::V1);
        assert_eq!(reader.kind(1).unwrap(), RunKind::V2);
    }

    #[test]
    fn decode_paths() {
        let v1 = build_v1_bytes(2);
        let v2 = build_v2_bytes(2);
        let tmp = write_pack(&[v1], &[v2]);
        let reader = PackReader::open(tmp.path()).unwrap();

        let r1 = reader.decode_v1(0).unwrap();
        assert_eq!(r1.meta.steps, 2);
        let r2 = reader.decode_v2(1).unwrap();
        assert_eq!(r2.meta.steps, 2);
        let r2auto = reader.decode_auto_v2(0).unwrap();
        assert_eq!(r2auto.meta.steps, 2);
    }

    #[test]
    fn iter_and_batch() {
        let mut v1s = vec![];
        for _ in 0..3 {
            v1s.push(build_v1_bytes(3));
        }
        let mut v2s = vec![];
        for _ in 0..2 {
            v2s.push(build_v2_bytes(4));
        }
        let tmp = write_pack(&v1s, &v2s);
        let reader = PackReader::open(tmp.path()).unwrap();
        let runs: Vec<_> = reader.iter().map(|r| r.unwrap()).collect();
        assert_eq!(runs.len(), 5);
        let batch = reader.decode_batch_auto_v2(&[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(batch.len(), 5);
        assert!(batch[0].meta.steps > 0);
    }

    #[test]
    fn stats_and_jsonl() {
        let v1 = build_v1_bytes(3);
        let v2 = build_v2_bytes(4);
        let tmp = write_pack(&[v1], &[v2]);
        let reader = PackReader::open(tmp.path()).unwrap();
        let st = reader.stats().unwrap();
        assert_eq!(st.count, 2);
        assert_eq!(st.min_len, 3);
        assert_eq!(st.max_len, 4);
        assert!(st.mean_len > 3.0 && st.mean_len < 4.1);
        // quantiles may be zero for fallback compute

        let out = NamedTempFile::new().unwrap();
        reader.to_jsonl(out.path(), true).unwrap();
        // sanity: file non-empty and contains two lines
        let mut s = String::new();
        File::open(out.path())
            .unwrap()
            .read_to_string(&mut s)
            .unwrap();
        let lines: Vec<_> = s.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"steps\""));
    }
}
