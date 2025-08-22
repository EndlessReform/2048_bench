use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::engine::Move;
use crate::expectimax::BranchEval as BranchEvalRaw;
use crate::trace::{Meta, Run};

/// Branch EV for a direction at a decision point.
///
/// Values are normalized to [0, 1) using per-step min–max across legal
/// branches; illegal branches are recorded explicitly.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BranchV2 {
    Legal(f32),
    Illegal,
}

/// A single decision step in a run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StepV2 {
    /// Board before the move.
    pub pre_board: u64,
    /// Chosen direction.
    pub chosen: Move,
    /// Normalized branch EVs for [Up, Down, Left, Right].
    ///
    /// Absent when converted from legacy v1 traces.
    pub branches: Option<[BranchV2; 4]>,
}

/// A full run in v2 representation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunV2 {
    pub meta: Meta,
    pub steps: Vec<StepV2>,
    pub final_board: u64,
}

#[derive(thiserror::Error, Debug)]
pub enum SerializationError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("postcard serialize error: {0}")]
    PostcardSer(#[from] postcard::Error),
}

/// Normalize raw branch EVs to [0, 1) with stable edge cases.
///
/// - If no legal moves: all `Illegal`.
/// - One legal move: that move gets `1.0 - f32::EPSILON`.
/// - Multiple legal with equal EVs: all legal get 0.5.
/// - Else: (ev - min) / (max - min), clamped to < 1.0.
pub fn normalize_branches(input: [BranchEvalRaw; 4]) -> [BranchV2; 4] {
    let mut legal_vals: Vec<(usize, f64)> = Vec::with_capacity(4);
    for (i, b) in input.iter().enumerate() {
        if b.legal { legal_vals.push((i, b.ev)); }
    }
    if legal_vals.is_empty() {
        return [BranchV2::Illegal, BranchV2::Illegal, BranchV2::Illegal, BranchV2::Illegal];
    }
    if legal_vals.len() == 1 {
        let mut out = [BranchV2::Illegal; 4];
        out[legal_vals[0].0] = BranchV2::Legal(1.0 - f32::EPSILON);
        return out;
    }
    let (mut min_ev, mut max_ev) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(_, ev) in &legal_vals {
        if ev < min_ev { min_ev = ev; }
        if ev > max_ev { max_ev = ev; }
    }
    let mut out = [BranchV2::Illegal; 4];
    if (max_ev - min_ev).abs() < f64::EPSILON {
        for (i, _) in legal_vals { out[i] = BranchV2::Legal(0.5); }
        return out;
    }
    let denom = (max_ev - min_ev) as f32;
    for (i, ev) in legal_vals {
        let mut v = ((ev - min_ev) as f32) / denom;
        if v >= 1.0 { v = 1.0 - f32::EPSILON; }
        out[i] = BranchV2::Legal(v);
    }
    out
}

/// Convert a legacy v1 binary run into a v2 struct, without branch EVs.
pub fn from_v1(run: Run) -> RunV2 {
    let steps_count = run.meta.steps as usize;
    let mut steps = Vec::with_capacity(steps_count);
    for i in 0..steps_count {
        steps.push(StepV2 {
            pre_board: run.states[i],
            chosen: u8_to_move(run.moves[i]),
            branches: None,
        });
    }
    let final_board = *run.states.last().unwrap_or(&0);
    RunV2 { meta: run.meta, steps, final_board }
}

#[inline]
fn u8_to_move(b: u8) -> Move {
    match b {
        0 => Move::Up,
        1 => Move::Down,
        2 => Move::Left,
        3 => Move::Right,
        _ => Move::Up, // fallback; v1 encoder only emits 0..=3
    }
}

/// Encode a v2 run to postcard bytes.
pub fn to_postcard_bytes(run: &RunV2) -> Result<Vec<u8>, SerializationError> {
    Ok(postcard::to_allocvec(run)?)
}

/// Decode a v2 run from postcard bytes.
pub fn from_postcard_bytes(bytes: &[u8]) -> Result<RunV2, SerializationError> {
    Ok(postcard::from_bytes(bytes)?)
}

/// Write postcard-encoded v2 run to a file.
pub fn write_postcard_to_path<P: AsRef<Path>>(path: P, run: &RunV2) -> Result<(), SerializationError> {
    let bytes = to_postcard_bytes(run)?;
    fs::write(path, bytes)?;
    Ok(())
}

/// Read postcard-encoded v2 run from a file.
pub fn read_postcard_from_path<P: AsRef<Path>>(path: P) -> Result<RunV2, SerializationError> {
    let bytes = fs::read(path)?;
    from_postcard_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn make_test_meta() -> Meta {
        Meta {
            steps: 3,
            start_unix_s: 1_700_000_000,
            elapsed_s: 5.25,
            max_score: 4096,
            highest_tile: 512,
            engine_str: Some("test-engine-v2".to_string()),
        }
    }

    fn make_test_branches() -> [BranchV2; 4] {
        [
            BranchV2::Legal(0.8),
            BranchV2::Legal(0.2),
            BranchV2::Illegal,
            BranchV2::Legal(0.5),
        ]
    }

    #[test]
    fn test_branch_serialization() {
        // Test individual BranchV2 enum variants
        let legal = BranchV2::Legal(0.5);
        let illegal = BranchV2::Illegal;
        
        let legal_bytes = postcard::to_allocvec(&legal).unwrap();
        let illegal_bytes = postcard::to_allocvec(&illegal).unwrap();
        
        let legal_loaded: BranchV2 = postcard::from_bytes(&legal_bytes).unwrap();
        let illegal_loaded: BranchV2 = postcard::from_bytes(&illegal_bytes).unwrap();
        
        assert_eq!(legal, legal_loaded);
        assert_eq!(illegal, illegal_loaded);
    }

    #[test]
    fn test_move_serialization() {
        // Test Move enum serialization
        let up = Move::Up;
        let bytes = postcard::to_allocvec(&up).unwrap();
        let loaded: Move = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(up, loaded);
    }

    #[test]
    fn test_meta_serialization() {
        let meta = Meta {
            steps: 1,
            start_unix_s: 1_700_000_000,
            elapsed_s: 1.0,
            max_score: 100,
            highest_tile: 4,
            engine_str: None,
        };
        let bytes = postcard::to_allocvec(&meta).unwrap();
        let loaded: Meta = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(meta, loaded);
    }

    #[test]
    fn test_stepv2_serialization() {
        let step = StepV2 {
            pre_board: 0x1000_0000_0000_0000,
            chosen: Move::Up,
            branches: None,
        };
        let bytes = postcard::to_allocvec(&step).unwrap();
        let loaded: StepV2 = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(step, loaded);
    }

    #[test]
    fn round_trip_basic() {
        // Test minimal run first
        let meta = Meta {
            steps: 1,
            start_unix_s: 1_700_000_000,
            elapsed_s: 1.0,
            max_score: 100,
            highest_tile: 4,
            engine_str: None,
        };
        let steps = vec![
            StepV2 {
                pre_board: 0x1000_0000_0000_0000,
                chosen: Move::Up,
                branches: None,
            },
        ];
        let final_board = 0x2000_0000_0000_0000;
        let run = RunV2 { meta: meta.clone(), steps: steps.clone(), final_board };

        let bytes = to_postcard_bytes(&run).unwrap();
        let loaded = from_postcard_bytes(&bytes).unwrap();
        assert_eq!(loaded, run);
    }

    #[test]
    fn round_trip_with_branches() {
        let meta = make_test_meta();
        let steps = vec![
            StepV2 {
                pre_board: 0x1000_0000_0000_0000,
                chosen: Move::Up,
                branches: Some(make_test_branches()),
            },
            StepV2 {
                pre_board: 0x1100_0000_0000_0000,
                chosen: Move::Right,
                branches: Some([BranchV2::Legal(0.9), BranchV2::Illegal, BranchV2::Legal(0.1), BranchV2::Legal(0.0)]),
            },
            StepV2 {
                pre_board: 0x1110_0000_0000_0000,
                chosen: Move::Down,
                branches: None, // Test mix of Some/None branches
            },
        ];
        let final_board = 0x2220_0000_0000_0000;
        let run = RunV2 { meta: meta.clone(), steps: steps.clone(), final_board };

        let tmp = NamedTempFile::new().unwrap();
        write_postcard_to_path(tmp.path(), &run).unwrap();
        let loaded = read_postcard_from_path(tmp.path()).unwrap();

        assert_eq!(loaded.meta, meta);
        assert_eq!(loaded.steps, steps);
        assert_eq!(loaded.final_board, final_board);
    }

    #[test]
    fn round_trip_empty_run() {
        let meta = Meta {
            steps: 0,
            start_unix_s: 1_600_000_000,
            elapsed_s: 0.1,
            max_score: 0,
            highest_tile: 2,
            engine_str: None,
        };
        let run = RunV2 { meta: meta.clone(), steps: vec![], final_board: 0x0010_0000_0000_0000 };

        let bytes = to_postcard_bytes(&run).unwrap();
        let loaded = from_postcard_bytes(&bytes).unwrap();
        assert_eq!(loaded, run);
    }

    #[test]
    fn test_branch_normalization() {
        // Test no legal moves
        let all_illegal = [
            BranchEvalRaw { dir: Move::Up, ev: 0.0, legal: false },
            BranchEvalRaw { dir: Move::Down, ev: 0.0, legal: false },
            BranchEvalRaw { dir: Move::Left, ev: 0.0, legal: false },
            BranchEvalRaw { dir: Move::Right, ev: 0.0, legal: false },
        ];
        let normalized = normalize_branches(all_illegal);
        assert_eq!(normalized, [BranchV2::Illegal; 4]);

        // Test single legal move
        let single_legal = [
            BranchEvalRaw { dir: Move::Up, ev: 100.0, legal: true },
            BranchEvalRaw { dir: Move::Down, ev: 0.0, legal: false },
            BranchEvalRaw { dir: Move::Left, ev: 0.0, legal: false },
            BranchEvalRaw { dir: Move::Right, ev: 0.0, legal: false },
        ];
        let normalized = normalize_branches(single_legal);
        assert_eq!(normalized[0], BranchV2::Legal(1.0 - f32::EPSILON));
        assert_eq!(normalized[1], BranchV2::Illegal);
        assert_eq!(normalized[2], BranchV2::Illegal);
        assert_eq!(normalized[3], BranchV2::Illegal);

        // Test equal EVs
        let equal_evs = [
            BranchEvalRaw { dir: Move::Up, ev: 50.0, legal: true },
            BranchEvalRaw { dir: Move::Down, ev: 50.0, legal: true },
            BranchEvalRaw { dir: Move::Left, ev: 0.0, legal: false },
            BranchEvalRaw { dir: Move::Right, ev: 50.0, legal: true },
        ];
        let normalized = normalize_branches(equal_evs);
        assert_eq!(normalized[0], BranchV2::Legal(0.5));
        assert_eq!(normalized[1], BranchV2::Legal(0.5));
        assert_eq!(normalized[2], BranchV2::Illegal);
        assert_eq!(normalized[3], BranchV2::Legal(0.5));

        // Test normal range
        let normal_range = [
            BranchEvalRaw { dir: Move::Up, ev: 100.0, legal: true },    // max -> 1.0 - epsilon
            BranchEvalRaw { dir: Move::Down, ev: 50.0, legal: true },   // mid -> 0.5
            BranchEvalRaw { dir: Move::Left, ev: 0.0, legal: true },    // min -> 0.0
            BranchEvalRaw { dir: Move::Right, ev: 0.0, legal: false },  // illegal
        ];
        let normalized = normalize_branches(normal_range);
        if let BranchV2::Legal(v) = normalized[0] { assert!((v - (1.0 - f32::EPSILON)).abs() < 1e-6); }
        if let BranchV2::Legal(v) = normalized[1] { assert!((v - 0.5).abs() < 1e-6); }
        if let BranchV2::Legal(v) = normalized[2] { assert!(v.abs() < 1e-6); }
        assert_eq!(normalized[3], BranchV2::Illegal);
    }

    #[test]
    fn test_from_v1_conversion() {
        let v1_meta = Meta {
            steps: 2,
            start_unix_s: 1_650_000_000,
            elapsed_s: 2.5,
            max_score: 1024,
            highest_tile: 128,
            engine_str: Some("legacy-engine".to_string()),
        };
        let v1_run = Run {
            meta: v1_meta.clone(),
            states: vec![0x1000_0000_0000_0000, 0x1100_0000_0000_0000, 0x1110_0000_0000_0000],
            moves: vec![0, 3], // Up, Right
        };

        let v2_run = from_v1(v1_run);
        assert_eq!(v2_run.meta, v1_meta);
        assert_eq!(v2_run.steps.len(), 2);
        assert_eq!(v2_run.steps[0].pre_board, 0x1000_0000_0000_0000);
        assert_eq!(v2_run.steps[0].chosen, Move::Up);
        assert_eq!(v2_run.steps[0].branches, None);
        assert_eq!(v2_run.steps[1].pre_board, 0x1100_0000_0000_0000);
        assert_eq!(v2_run.steps[1].chosen, Move::Right);
        assert_eq!(v2_run.steps[1].branches, None);
        assert_eq!(v2_run.final_board, 0x1110_0000_0000_0000);
    }

    #[test]
    fn test_u8_to_move_conversion() {
        assert_eq!(u8_to_move(0), Move::Up);
        assert_eq!(u8_to_move(1), Move::Down);
        assert_eq!(u8_to_move(2), Move::Left);
        assert_eq!(u8_to_move(3), Move::Right);
        assert_eq!(u8_to_move(255), Move::Up); // fallback
    }

    #[test]
    fn test_branches_serialization() {
        let run = RunV2 {
            meta: make_test_meta(),
            steps: vec![
                StepV2 {
                    pre_board: 0x1000_0000_0000_0000,
                    chosen: Move::Up,
                    branches: Some([
                        BranchV2::Legal(0.0),
                        BranchV2::Legal(1.0 - f32::EPSILON),
                        BranchV2::Illegal,
                        BranchV2::Legal(0.5),
                    ]),
                },
            ],
            final_board: 0x2000_0000_0000_0000,
        };

        let bytes = to_postcard_bytes(&run).unwrap();
        let loaded = from_postcard_bytes(&bytes).unwrap();
        assert_eq!(loaded, run);

        // Verify specific branch values survive serialization
        if let Some(branches) = loaded.steps[0].branches {
            if let BranchV2::Legal(v) = branches[0] { assert!(v.abs() < 1e-6); }
            if let BranchV2::Legal(v) = branches[1] { assert!((v - (1.0 - f32::EPSILON)).abs() < 1e-6); }
            assert_eq!(branches[2], BranchV2::Illegal);
            if let BranchV2::Legal(v) = branches[3] { assert!((v - 0.5).abs() < 1e-6); }
        } else {
            panic!("Expected branches to be present");
        }
    }

    #[test]
    fn test_large_run() {
        let meta = Meta {
            steps: 1000,
            start_unix_s: 1_700_000_000,
            elapsed_s: 60.0,
            max_score: 100000,
            highest_tile: 2048,
            engine_str: Some("stress-test".to_string()),
        };
        
        let mut steps = Vec::with_capacity(1000);
        for i in 0..1000 {
            steps.push(StepV2 {
                pre_board: 0x1000_0000_0000_0000 + i as u64,
                chosen: match i % 4 {
                    0 => Move::Up,
                    1 => Move::Down,
                    2 => Move::Left,
                    _ => Move::Right,
                },
                branches: if i % 10 == 0 { None } else { Some(make_test_branches()) },
            });
        }
        
        let run = RunV2 { meta: meta.clone(), steps: steps.clone(), final_board: 0xFFFF_FFFF_FFFF_FFFF };

        let tmp = NamedTempFile::new().unwrap();
        write_postcard_to_path(tmp.path(), &run).unwrap();
        let loaded = read_postcard_from_path(tmp.path()).unwrap();

        assert_eq!(loaded.meta, meta);
        assert_eq!(loaded.steps.len(), 1000);
        assert_eq!(loaded.final_board, 0xFFFF_FFFF_FFFF_FFFF);
        
        // Spot-check a few steps
        assert_eq!(loaded.steps[0].chosen, Move::Up);
        assert_eq!(loaded.steps[999].chosen, Move::Right);
        assert_eq!(loaded.steps[0].branches, None); // i % 10 == 0
        assert_eq!(loaded.steps[1].branches, Some(make_test_branches()));
    }
}

