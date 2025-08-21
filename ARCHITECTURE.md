**Storage**
- Goal: fast, read-heavy dataloading with simple, robust files. Favor maintainability over maximal compression; corruption of a single run should not jeopardize others.

- File-per-run, binary layout (v1):
  - Magic + version: 4 bytes magic `A2T1`, 1 byte version `1`.
  - Endianness: 1 byte (0 = little-endian). All integers are LE.
  - Metadata (fixed-size):
    - `u32 steps` (number of moves)
    - `u64 start_unix_s` (optional; 0 if unknown)
    - `f32 elapsed_s` (wall time seconds)
    - `u64 max_score`
    - `u32 highest_tile`
  - Variable metadata (optional):
    - `u16 engine_str_len` then UTF-8 bytes (e.g., git SHA, engine mode). Can be 0.
  - Payload:
    - States: `u64[steps + 1]` (initial state + one per move). Zero-copy friendly.
    - Moves: `u8[steps]` (0=Up, 1=Down, 2=Left, 3=Right).
  - Trailer:
    - `u32 checksum` = CRC32C of all preceding bytes in the file (i.e., file contents except the final 4 bytes). Reader recomputes and compares; on mismatch, skip the file.
  - No spawns stored: states already encode full trajectory; spawns can be derived offline if ever needed.

- Why this design
  - KISS: no indexes, no bit-packing, no container multiplexing. One run → one file.
  - Read speed: dataloader can memory-map the file, read fixed-size header, then take slices for states/moves. No serde per-element cost.
  - Space: ~8*(steps+1) + steps bytes. For ~20k steps/run ≈ ~160 KB (states) + ~20 KB (moves) ≈ ~180 KB/run. 100k runs ≈ ~18 GB; well within storage budget.
  - Failure isolation: a corrupted file only affects that single run. The loader can skip unreadable files. The checksum adds a cheap integrity check.

- Dataloader pattern (Rust)
  - Enumerate files (e.g., under `traces/`).
  - Open + mmap.
  - Validate magic/version, bounds check sizes.
  - Validate checksum: compute CRC32C over `data[..len-4]` and compare to the stored `u32` at `data[len-4..]`.
  - Cast payload to `&[u64]` and `&[u8]` (e.g., via `bytemuck`), respecting LE.
  - Yield `(states, moves, meta)` to tokenization.

- Filenames and sharding
  - Keep simple directories (e.g., `traces/YYYYMMDD/uuid.bin`).
  - If directory fanout becomes an issue, shard by first 2 bytes of UUID (`traces/ab/cd/<uuid>.bin`).
  - No in-file index; the filesystem is the index. A separate optional CSV/Parquet summary can be generated offline for analytics.

- Optional refinements (only if needed later)
  - Pack moves to 2 bits (4 per byte) to shave ~75% on the moves array; keep states as `u64` for zero-copy.
  - Transparent compression at the filesystem or archive level (zstd), not in the format.
  - If CRC32C isn’t available, CRC32 (IEEE) or xxHash32 are acceptable. Use a tiny crate (`crc32c`, `crc32fast`, or `xxhash-rust`) for speed and simplicity.

- Future compatibility
  - Bump the version if layout changes; keep the magic constant.
  - Add new fixed-size metadata fields by extending the header; older readers can skip via version gating.
