//! Serialization surfaces for run traces.
//!
//! This module defines a postcard-based v2 format that records per-step
//! branch evaluations alongside the board and chosen move. It also provides
//! helpers to normalize branch EVs and to convert legacy v1 traces to v2
//! structures (with `branches: None`).

mod v2;

pub use v2::{
    BranchV2,
    StepV2,
    RunV2,
    SerializationError,
    normalize_branches,
    from_v1,
    to_postcard_bytes,
    from_postcard_bytes,
    write_postcard_to_path,
    read_postcard_from_path,
};

