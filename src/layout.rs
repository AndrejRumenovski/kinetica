//! Zero-copy memory layout for the catalyst surface lattice and the
//! reaction-rate lookup table.
//!
//! Nothing in this module owns a heap-allocated `Vec` for simulation state.
//! The site matrix and the reaction LUT are both backed by `memmap2`
//! mappings: the former read-write over a scratch file that can be many
//! times larger than physical RAM (the "out-of-core" requirement), the
//! latter read-only over a prebuilt `reactions.lut` blob of OC20 rate
//! constants. In both cases the OS page cache is the only buffer between
//! the NVMe device and the CPU -- this module just hands out typed views
//! over that mapped memory.

use memmap2::{Mmap, MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;
use std::slice;

// ------------------------------------------------------------------------
// Bitflag occupancy states
// ------------------------------------------------------------------------

/// Site is unoccupied: no adsorbate bound to this lattice point.
pub const VACANT: u8 = 0x00;
/// Site occupied by an adsorbed oxygen atom (O*).
pub const ADS_O: u8 = 0x01;
/// Site occupied by an adsorbed carbon monoxide molecule (CO*).
pub const ADS_CO: u8 = 0x02;
/// Site occupied by an adsorbed hydrogen atom (H*).
pub const ADS_H: u8 = 0x04;

/// Union of every currently defined occupancy bit.
///
/// Hot-path code should only ever mask *with* this constant (`byte &
/// OCCUPANCY_MASK`) to read known state; a set bit outside this mask means
/// either a newer site type this build doesn't know about or a corrupted
/// lattice file, and callers that care should treat it as the latter.
pub const OCCUPANCY_MASK: u8 = ADS_O | ADS_CO | ADS_H;

// ------------------------------------------------------------------------
// Bit-packed, memory-mapped site matrix
// ------------------------------------------------------------------------

/// The catalyst surface: a flat, memory-mapped, row-major byte matrix of
/// `width * height` occupancy bytes.
///
/// Deliberately *not* a `Vec<Vec<u8>>` or any nested structure -- a single
/// contiguous mapped region means every worker's neighborhood scan is a
/// linear walk over adjacent bytes the prefetcher can see coming, and the
/// domain-decomposition split in `engine.rs` is just slicing this one
/// allocation, never copying it.
pub struct SiteLattice {
    mmap: MmapMut,
    pub width: usize,
    pub height: usize,
}

impl SiteLattice {
    /// Open (creating and zero-extending if necessary) the backing lattice
    /// file at `path` and map `width * height` bytes of it read-write.
    pub fn open(path: impl AsRef<Path>, width: usize, height: usize) -> io::Result<Self> {
        let len = width
            .checked_mul(height)
            .expect("lattice dimensions overflow usize");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            // Explicit about intent: a pre-existing lattice file (e.g. one
            // resumed from a prior out-of-core run) must NOT be wiped on
            // open -- `set_len` below only grows/shrinks it to the target
            // size without touching bytes that remain in range.
            .truncate(false)
            .open(path)?;
        // Extend (or truncate) the backing file to exactly the mapped
        // length up front, so the mapping below never straddles the file's
        // actual end-of-data.
        file.set_len(len as u64)?;

        // SAFETY: `file` was just opened/created and sized by this call, so
        // no other mapping in this process already covers its range, and we
        // hold the only handle to it at this point. `map_mut` requires the
        // caller to guarantee the file isn't concurrently truncated or
        // resized by another process for the lifetime of the mapping --
        // this tool's contract is single-writer ownership of the lattice
        // file, enforced at the process level by the CLI in `main.rs`.
        let mmap = unsafe { MmapOptions::new().len(len).map_mut(&file)? };

        Ok(Self { mmap, width, height })
    }

    /// Flat row-major byte view: `width * height` bytes, one per site,
    /// index `y * width + x`. No 2D indirection is exposed on purpose --
    /// callers compute the linear index themselves so the compiler keeps
    /// scans over it a single contiguous walk.
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap[..]
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.mmap[..]
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> u8 {
        self.mmap[idx]
    }

    #[inline(always)]
    pub fn set(&mut self, idx: usize, state: u8) {
        self.mmap[idx] = state;
    }

    /// Split the lattice into up to `n` disjoint, contiguous **row-band**
    /// domains for spatial decomposition (see `engine::Patch`). Each band
    /// spans full rows (`y0..y1`) across the whole width, so every patch
    /// stays grid-aligned to real (x, y) lattice coordinates rather than
    /// being an arbitrary byte offset that could split a row in half.
    /// Bands alias distinct regions of the same underlying mapping, so `n`
    /// worker threads can each hold one exclusively with no locking and no
    /// copy.
    pub fn split_row_bands_mut(&mut self, n: usize) -> Vec<(usize, usize, &mut [u8])> {
        let n = n.max(1).min(self.height.max(1));
        let rows_per_band = self.height.div_ceil(n).max(1);
        let width = self.width;

        let mut bands = Vec::with_capacity(n);
        let mut rest = &mut self.mmap[..];
        let mut y0 = 0usize;
        while !rest.is_empty() {
            let rows_here = rows_per_band.min(self.height - y0);
            let byte_len = rows_here * width;
            let (band, remainder) = rest.split_at_mut(byte_len);
            bands.push((y0, y0 + rows_here, band));
            rest = remainder;
            y0 += rows_here;
        }
        bands
    }

    /// Flush the OS's dirty pages for this mapping back to the backing
    /// file, giving durability of everything written so far.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

// ------------------------------------------------------------------------
// Cache-line-aligned reaction-rate lookup table (AoSOA)
// ------------------------------------------------------------------------

/// One 64-byte, cache-line-resident block of `LANES` reaction records laid
/// out Array-of-Structures-of-Arrays: each *field* is contiguous across the
/// block's lanes, but blocks themselves are addressed and cached as single
/// units.
///
/// `#[repr(C, align(64))]`: `align(64)` pins every block to a 64-byte
/// hardware cache-line boundary on all mainstream x86_64/AArch64 parts, so
/// a worker reading or updating one block's propensities touches exactly
/// one cache line -- no split loads, and no false sharing with a
/// neighboring block a different worker is writing. `repr(C)` is required
/// alongside it: without it the compiler is free to reorder these fields,
/// which would silently desync this struct's in-memory layout from the
/// byte-for-byte format `reactions.lut` is written in on disk. The four
/// field arrays are sized so the struct is *exactly* 64 bytes with no
/// trailing padding: `4*8 + 1*8 + 1*8 + 2*8 = 64`.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct ReactionLutBlock {
    /// Fixed-point Q16.16 propensity-contributing rate constant, one per
    /// lane. See `gillespie::FixedPoint` for the encoding this feeds.
    pub rate_q16: [u32; Self::LANES],
    /// Composition-rejection bin this lane's reaction currently belongs to.
    pub bin_id: [u8; Self::LANES],
    /// Packed `(reactant_mask << 4) | product_mask` site transition.
    pub transition: [u8; Self::LANES],
    /// Activation energy in milli-eV, retained for diagnostics/re-fitting;
    /// not read on the hot sampling path.
    pub e_act_mev: [u16; Self::LANES],
}

impl ReactionLutBlock {
    /// Reactions packed per 64-byte block. See the struct-level layout
    /// comment for how this size was chosen to fill exactly one cache line.
    pub const LANES: usize = 8;
}

// Compile-time invariants the `unsafe` reinterpretation in `ReactionLut`
// below depends on: if either ever drifts (e.g. a field is added), this
// fails the build instead of silently corrupting reads.
const _: () = assert!(std::mem::size_of::<ReactionLutBlock>() == 64);
const _: () = assert!(std::mem::align_of::<ReactionLutBlock>() == 64);

/// Split a flat global reaction id into its `(block_index, lane_index)`
/// coordinates inside a `ReactionLut`. Pure integer division/modulo by a
/// compile-time power-of-two-friendly constant -- LLVM lowers this to
/// shift/mask, not a division instruction, keeping it O(1) and
/// branch-free.
#[inline(always)]
pub fn block_and_lane(reaction_id: usize) -> (usize, usize) {
    (
        reaction_id / ReactionLutBlock::LANES,
        reaction_id % ReactionLutBlock::LANES,
    )
}

/// Read-only, memory-mapped view over a prebuilt `reactions.lut` file: a
/// flat run of `ReactionLutBlock`s with no header, so mapping it is just a
/// pointer cast over the file's bytes.
pub struct ReactionLut {
    // Kept only to hold the mapping alive for the lifetime of `blocks`;
    // never read directly after `open` validates and casts it.
    _mmap: Mmap,
    blocks: *const ReactionLutBlock,
    len: usize,
}

// SAFETY: `ReactionLut` is a read-only view over mapped file bytes with no
// interior mutability; sharing `&ReactionLut` (and thus `blocks`) across
// threads is exactly as sound as sharing an immutable slice reference,
// which `Sync`/`Send` for `&[T]` already presumes. The raw pointer only
// exists in place of a slice reference to let `as_slice` hand out a
// `'self`-scoped `&[ReactionLutBlock]` without fighting the borrow checker
// over the `Mmap` field it's derived from.
unsafe impl Send for ReactionLut {}
unsafe impl Sync for ReactionLut {}

impl ReactionLut {
    /// Map `path` and validate it as a whole number of correctly-aligned
    /// `ReactionLutBlock`s.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;

        // SAFETY: read-only mapping of a file opened read-only above; this
        // process does not concurrently write to it, and `reactions.lut` is
        // treated as a static build artifact for the lifetime of the run.
        let mmap = unsafe { Mmap::map(&file)? };

        let block_size = std::mem::size_of::<ReactionLutBlock>();
        if mmap.len() % block_size != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "reactions.lut length is not a whole number of 64-byte AoSOA blocks",
            ));
        }

        let ptr = mmap.as_ptr();
        if ptr.align_offset(std::mem::align_of::<ReactionLutBlock>()) != 0 {
            // mmap bases are page-aligned (>= 4096 bytes) on every platform
            // this project targets, which is always a multiple of 64; this
            // branch exists as a defensive check, not an expected path.
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "reactions.lut mapping is not 64-byte aligned",
            ));
        }

        let len = mmap.len() / block_size;
        let blocks = ptr as *const ReactionLutBlock;

        Ok(Self {
            _mmap: mmap,
            blocks,
            len,
        })
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The full LUT as a typed slice of cache-line blocks.
    #[inline(always)]
    pub fn as_slice(&self) -> &[ReactionLutBlock] {
        // SAFETY: `blocks` points at `self._mmap`'s first byte, and
        // `_mmap` is kept alive for at least as long as `self` (dropped
        // together), so the returned slice -- borrowed from `&self` -- can
        // never outlive the backing mapping. `open` verified `mmap.len()`
        // is an exact multiple of `size_of::<ReactionLutBlock>()`, so
        // `self.len` blocks fit entirely in-bounds with no partial
        // trailing block, and verified the base pointer's alignment
        // satisfies `align_of::<ReactionLutBlock>()` (64). Every field of
        // `ReactionLutBlock` is a plain fixed-width integer array with no
        // padding bytes (enforced by the `repr(C, align(64))` size
        // assertion above), so any bit pattern in the mapped file is a
        // valid `ReactionLutBlock` -- there is no uninitialized-padding or
        // invalid-enum-discriminant hazard in reinterpreting these bytes.
        unsafe { slice::from_raw_parts(self.blocks, self.len) }
    }

    /// Direct O(1) lookup of a single reaction record by its flat global
    /// id, returning the `(rate_q16, bin_id, transition)` triple without
    /// materializing an intermediate `ReactionRate` struct.
    #[inline(always)]
    pub fn rate_of(&self, reaction_id: usize) -> (u32, u8, u8) {
        let (block_idx, lane) = block_and_lane(reaction_id);
        let block = &self.as_slice()[block_idx];
        (block.rate_q16[lane], block.bin_id[lane], block.transition[lane])
    }
}

/// Pack `(rate_q16, bin_id, transition)` triples into `ReactionLutBlock`s,
/// sorting by `bin_id` first (the invariant `CompositionTable::build`
/// relies on to collapse bin membership to a `[start, count)` range). Used
/// by both the synthetic demo generator (`main.rs`) and the real OC20
/// ingestion tool (`bin/oc20_ingest.rs`) so the on-disk packing logic
/// exists in exactly one place.
pub fn pack_records_into_blocks(mut records: Vec<(u32, u8, u8)>) -> Vec<ReactionLutBlock> {
    records.sort_by_key(|&(_, bin_id, _)| bin_id);

    let block_count = records.len().div_ceil(ReactionLutBlock::LANES).max(1);
    let mut blocks = Vec::with_capacity(block_count);

    for chunk in records.chunks(ReactionLutBlock::LANES) {
        let mut rate_q16 = [0u32; ReactionLutBlock::LANES];
        let mut bin_id = [0u8; ReactionLutBlock::LANES];
        let mut transition = [0u8; ReactionLutBlock::LANES];
        let e_act_mev = [0u16; ReactionLutBlock::LANES];

        for (lane, &(rate, bin, trans)) in chunk.iter().enumerate() {
            rate_q16[lane] = rate;
            bin_id[lane] = bin;
            transition[lane] = trans;
        }

        blocks.push(ReactionLutBlock {
            rate_q16,
            bin_id,
            transition,
            e_act_mev,
        });
    }

    blocks
}

/// Write `blocks` verbatim to `path` as the raw bytes `ReactionLut::open`
/// expects to map back in.
pub fn write_lut(path: impl AsRef<Path>, blocks: &[ReactionLutBlock]) -> io::Result<()> {
    // SAFETY: `ReactionLutBlock` is `repr(C, align(64))`, `Copy`, and every
    // field is a plain fixed-width integer array with no padding bytes
    // (enforced by the `size_of::<ReactionLutBlock>() == 64` assertion
    // above), so reinterpreting `&[ReactionLutBlock]` as `&[u8]` for the
    // duration of this write is a sound, lossless byte-for-byte view --
    // there is no uninitialized padding to expose, and no lifetime hazard
    // since the byte slice does not outlive `blocks`.
    let bytes = unsafe {
        slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            std::mem::size_of_val(blocks),
        )
    };

    std::fs::File::create(path)?.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_path;

    #[test]
    fn block_and_lane_computes_correct_coordinates() {
        assert_eq!(block_and_lane(0), (0, 0));
        assert_eq!(block_and_lane(7), (0, 7));
        assert_eq!(block_and_lane(8), (1, 0));
        assert_eq!(block_and_lane(23), (2, 7));
    }

    #[test]
    fn pack_records_into_blocks_sorts_by_bin_id_and_pads_last_block() {
        let records = vec![(100u32, 3u8, 0x12u8), (50, 1, 0x21), (10, 0, 0x00)];
        let blocks = pack_records_into_blocks(records);
        assert_eq!(blocks.len(), 1);

        let b = &blocks[0];
        assert_eq!((b.rate_q16[0], b.bin_id[0]), (10, 0));
        assert_eq!((b.rate_q16[1], b.bin_id[1]), (50, 1));
        assert_eq!((b.rate_q16[2], b.bin_id[2]), (100, 3));
        assert_eq!(b.transition[2], 0x12);

        for lane in 3..ReactionLutBlock::LANES {
            assert_eq!(b.rate_q16[lane], 0);
            assert_eq!(b.bin_id[lane], 0);
            assert_eq!(b.transition[lane], 0);
        }
    }

    #[test]
    fn pack_records_into_blocks_empty_input_yields_no_blocks() {
        assert!(pack_records_into_blocks(Vec::new()).is_empty());
    }

    #[test]
    fn write_and_reopen_lut_round_trips_reaction_rates() {
        let records = vec![(10u32, 0u8, 0x01u8), (20, 1, 0x02), (30, 2, 0x04), (40, 31, 0x00)];
        let blocks = pack_records_into_blocks(records.clone());
        let path = temp_path("lut_roundtrip");
        write_lut(&path, &blocks).unwrap();

        let lut = ReactionLut::open(&path).unwrap();
        assert_eq!(lut.len(), blocks.len());
        for (i, &(rate, bin, trans)) in records.iter().enumerate() {
            assert_eq!(lut.rate_of(i), (rate, bin, trans));
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_file_with_length_not_multiple_of_block_size() {
        let path = temp_path("lut_bad_len");
        std::fs::write(&path, [0u8; 100]).unwrap(); // not a multiple of 64
        assert!(ReactionLut::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn site_lattice_get_set_round_trips() {
        let path = temp_path("lattice_get_set");
        let mut lattice = SiteLattice::open(&path, 4, 3).unwrap();
        assert_eq!(lattice.as_slice().len(), 12);

        lattice.set(5, ADS_O);
        assert_eq!(lattice.get(5), ADS_O);
        assert_eq!(lattice.get(0), VACANT);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn split_row_bands_mut_partitions_full_height_without_overlap() {
        let path = temp_path("lattice_bands");
        let mut lattice = SiteLattice::open(&path, 4, 10).unwrap();
        let bands = lattice.split_row_bands_mut(3);

        let mut covered = 0usize;
        let mut prev_y1 = 0usize;
        for (y0, y1, data) in &bands {
            assert_eq!(*y0, prev_y1);
            assert!(y1 > y0);
            assert_eq!(data.len(), (y1 - y0) * 4);
            covered += y1 - y0;
            prev_y1 = *y1;
        }
        assert_eq!(covered, 10);

        let _ = std::fs::remove_file(&path);
    }
}
