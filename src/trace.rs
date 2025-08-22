use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const MAGIC: &[u8; 4] = b"A2T1"; // ASCII magic
const VERSION: u8 = 1;
const ENDIAN_LE: u8 = 0; // 0 = little-endian

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    pub steps: u32,
    pub start_unix_s: u64,
    pub elapsed_s: f32,
    pub max_score: u64,
    pub highest_tile: u32,
    pub engine_str: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub meta: Meta,
    pub states: Vec<u64>, // length = steps + 1
    pub moves: Vec<u8>,   // length = steps
}

#[derive(thiserror::Error, Debug)]
pub enum TraceError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("invalid magic or version")] 
    MagicOrVersion,
    #[error("unsupported endianness")] 
    Endianness,
    #[error("file too short or malformed")] 
    Malformed,
    #[error("checksum mismatch")] 
    Checksum,
}

#[inline]
fn read_u16_le(bytes: &[u8]) -> Option<u16> {
    if bytes.len() < 2 { return None; }
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

#[inline]
fn read_u32_le(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 4 { return None; }
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[inline]
fn read_u64_le(bytes: &[u8]) -> Option<u64> {
    if bytes.len() < 8 { return None; }
    Some(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

#[inline]
fn read_f32_le(bytes: &[u8]) -> Option<f32> {
    read_u32_le(bytes).map(f32::from_bits)
}

pub fn encode_run(meta: &Meta, states: &[u64], moves: &[u8]) -> Vec<u8> {
    // Validate lengths consistent
    assert_eq!(states.len(), meta.steps as usize + 1);
    assert_eq!(moves.len(), meta.steps as usize);

    let engine_bytes = meta
        .engine_str
        .as_ref()
        .map(|s| s.as_bytes())
        .unwrap_or(&[]);
    let engine_len: u16 = engine_bytes
        .len()
        .try_into()
        .expect("engine_str too long for u16 length");

    // Header size:
    // 4 magic + 1 version + 1 endian + 4 steps + 8 start + 4 elapsed + 8 max_score + 4 highest_tile + 2 engine_len
    let header_len = 4 + 1 + 1 + 4 + 8 + 4 + 8 + 4 + 2;
    let states_len = states.len() * 8;
    let moves_len = moves.len();
    let payload_len = engine_len as usize + states_len + moves_len;
    let total_without_checksum = header_len + payload_len;
    let mut buf = Vec::with_capacity(total_without_checksum + 4);

    // Header
    buf.extend_from_slice(MAGIC);
    buf.push(VERSION);
    buf.push(ENDIAN_LE);
    buf.extend_from_slice(&meta.steps.to_le_bytes());
    buf.extend_from_slice(&meta.start_unix_s.to_le_bytes());
    buf.extend_from_slice(&meta.elapsed_s.to_bits().to_le_bytes());
    buf.extend_from_slice(&meta.max_score.to_le_bytes());
    buf.extend_from_slice(&meta.highest_tile.to_le_bytes());
    buf.extend_from_slice(&engine_len.to_le_bytes());

    // Variable metadata
    buf.extend_from_slice(engine_bytes);

    // Payload: states LE u64, then moves u8
    for &v in states { buf.extend_from_slice(&v.to_le_bytes()); }
    buf.extend_from_slice(moves);

    // Trailer: CRC32C of all preceding bytes
    let checksum = crc32c::crc32c(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf
}

pub fn write_run_to_path<P: AsRef<Path>>(path: P, meta: &Meta, states: &[u64], moves: &[u8]) -> Result<(), TraceError> {
    let data = encode_run(meta, states, moves);
    let mut f = fs::File::create(path)?;
    f.write_all(&data)?;
    Ok(())
}

pub fn parse_run_bytes(bytes: &[u8]) -> Result<Run, TraceError> {
    if bytes.len() < 4 + 1 + 1 + 4 + 8 + 4 + 8 + 4 + 2 + 4 { // header + checksum at minimum (no payload)
        return Err(TraceError::Malformed);
    }

    // Validate checksum first to avoid panics while reading fields
    if bytes.len() < 4 { return Err(TraceError::Malformed); }
    let (content, trailer) = bytes.split_at(bytes.len() - 4);
    let file_crc = read_u32_le(trailer).ok_or(TraceError::Malformed)?;
    let calc_crc = crc32c::crc32c(content);
    if file_crc != calc_crc { return Err(TraceError::Checksum); }

    // Fixed header
    if &content[..4] != MAGIC { return Err(TraceError::MagicOrVersion); }
    if content[4] != VERSION { return Err(TraceError::MagicOrVersion); }
    if content[5] != ENDIAN_LE { return Err(TraceError::Endianness); }

    let mut off = 6;
    let steps = read_u32_le(&content[off..]).ok_or(TraceError::Malformed)?; off += 4;
    let start_unix_s = read_u64_le(&content[off..]).ok_or(TraceError::Malformed)?; off += 8;
    let elapsed_s = read_f32_le(&content[off..]).ok_or(TraceError::Malformed)?; off += 4;
    let max_score = read_u64_le(&content[off..]).ok_or(TraceError::Malformed)?; off += 8;
    let highest_tile = read_u32_le(&content[off..]).ok_or(TraceError::Malformed)?; off += 4;
    let engine_len = read_u16_le(&content[off..]).ok_or(TraceError::Malformed)? as usize; off += 2;

    if content.len() < off + engine_len { return Err(TraceError::Malformed); }
    let engine_bytes = &content[off..off + engine_len];
    off += engine_len;
    let engine_str = if engine_len > 0 {
        match std::str::from_utf8(engine_bytes) {
            Ok(s) => Some(s.to_string()),
            Err(_) => None,
        }
    } else { None };

    let states_count = steps as usize + 1;
    let states_bytes_len = states_count.checked_mul(8).ok_or(TraceError::Malformed)?;
    let moves_len = steps as usize;

    if content.len() < off + states_bytes_len + moves_len {
        return Err(TraceError::Malformed);
    }

    let mut states = Vec::with_capacity(states_count);
    let mut i = 0;
    while i < states_bytes_len {
        let v = read_u64_le(&content[off + i..]).ok_or(TraceError::Malformed)?;
        states.push(v);
        i += 8;
    }
    off += states_bytes_len;

    let moves = content[off..off + moves_len].to_vec();

    let meta = Meta {
        steps,
        start_unix_s,
        elapsed_s,
        max_score,
        highest_tile,
        engine_str,
    };

    Ok(Run { meta, states, moves })
}

pub fn parse_run_file<P: AsRef<Path>>(path: P) -> Result<Run, TraceError> {
    let data = fs::read(path)?;
    parse_run_bytes(&data)
}

pub fn now_unix_seconds() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn round_trip_small() {
        let states = vec![0_u64, 0x1000_0000_0000_0000, 0x1100_0000_0000_0000];
        let moves = vec![2_u8, 0_u8];
        let meta = Meta {
            steps: moves.len() as u32,
            start_unix_s: 1_700_000_000,
            elapsed_s: 12.34,
            max_score: 12345,
            highest_tile: 2048,
            engine_str: Some("test-engine".to_string()),
        };

        let tmp = NamedTempFile::new().unwrap();
        write_run_to_path(tmp.path(), &meta, &states, &moves).unwrap();
        let run = parse_run_file(tmp.path()).unwrap();
        assert_eq!(run.meta, meta);
        assert_eq!(run.states, states);
        assert_eq!(run.moves, moves);
    }

    #[test]
    fn checksum_mismatch() {
        let states = vec![0_u64, 1_u64];
        let moves = vec![3_u8];
        let meta = Meta { steps: 1, start_unix_s: 0, elapsed_s: 0.0, max_score: 0, highest_tile: 0, engine_str: None };
        let mut bytes = encode_run(&meta, &states, &moves);
        // Flip one byte in the payload
        let idx = 4 + 1 + 1 + 4 + 8 + 4 + 8 + 4 + 2; // start of engine_len (0)
        let payload_start = idx + 2; // engine_len bytes
        bytes[payload_start] ^= 0xFF;
        // Parsing should fail on checksum
        let err = parse_run_bytes(&bytes).unwrap_err();
        matches!(err, TraceError::Checksum);
    }

    #[test]
    fn malformed_bounds() {
        let states = vec![0_u64, 1_u64, 2_u64];
        let moves = vec![3_u8, 1_u8];
        let meta = Meta { steps: 2, start_unix_s: 0, elapsed_s: 0.0, max_score: 0, highest_tile: 0, engine_str: None };
        let mut bytes = encode_run(&meta, &states, &moves);
        // Truncate last 5 bytes to simulate incomplete file
        bytes.truncate(bytes.len() - 5);
        let err = parse_run_bytes(&bytes).unwrap_err();
        matches!(err, TraceError::Malformed);
    }
}
