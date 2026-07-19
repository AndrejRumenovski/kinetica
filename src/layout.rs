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
/// Site occupied by an adsorbed hydroxyl (OH*) -- see `oc20_ingest`'s
/// water-dissociation reaction (`2* + H2O(g) <-> H* + OH*`) for the one
/// real reaction that currently forms/consumes it.
pub const ADS_OH: u8 = 0x08;
/// Site occupied by molecularly adsorbed water (H2O*) -- distinct from
/// the dissociated H*/OH* pair above. Formed/consumed by real Pd(111)
/// monomolecular adsorption/desorption chemistry (`star + H2O(g) <->
/// H2Ostar`, 3 real Catalysis-Hub records, BEP-estimated barrier like
/// O*/H*/CO*), gas-coupled via `occupancy::Pressures::h2o`.
pub const ADS_H2O: u8 = 0x10;

/// Union of every currently defined occupancy bit.
///
/// Hot-path code should only ever mask *with* this constant (`byte &
/// OCCUPANCY_MASK`) to read known state; a set bit outside this mask means
/// either a newer site type this build doesn't know about or a corrupted
/// lattice file, and callers that care should treat it as the latter.
pub const OCCUPANCY_MASK: u8 = ADS_O | ADS_CO | ADS_H | ADS_OH | ADS_H2O;

/// The original fixed 5-species Pd(111) adsorbate list this project shipped
/// with before the config-driven generalization arc. Every real-data
/// producer/consumer -- `oc20_ingest`, `main.rs`, `engine.rs`/
/// `occupancy.rs`, `coverage_report.rs` -- now derives its species identity
/// at runtime from a LUT's own self-described `SpeciesTable`
/// (`ReactionLut::species`) instead of this compile-time list, so a build
/// naming a different species set works without touching any of them. Kept
/// as a real, still-used constant (test fixtures across the crate build a
/// `SpeciesTable` that reproduces it exactly, and `--generate-lut`'s
/// synthetic demo data still exercises this bit-layout convention) rather
/// than removed outright.
///
/// **Eight is the practical ceiling for this array, not an arbitrary
/// round number.** `apply_transition` below packs a reaction's reactant
/// and product species into a `u16` as two 8-bit byte-masks
/// (`(reactant_mask << 8) | product_mask`); a one-hot species bit has to
/// fit inside one byte to be representable there, which caps this list at
/// `{0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80}`. This used to be a
/// 4-bit nibble (4-species ceiling) until the OH*-to-H2O* extension
/// needed a 5th slot; widening the *packing unit* (nibble -> byte) rather
/// than switching occupancy representation entirely (bitmask -> compact
/// index) was a deliberate choice: it preserves the exact "no site holds
/// more than one adsorbate bit" corruption check
/// (`engine.rs`'s end-to-end test, `OCCUPANCY_MASK`'s `count_ones()`
/// invariant) byte-for-byte, rather than swapping it for a different
/// verification strategy. The cost is `ReactionLutBlock`'s block size
/// (see its own doc comment): `transition_a`/`transition_b` are now `u16`
/// or per-reaction bytes roughly doubles.
pub const SPECIES_BITS: [u8; 5] = [ADS_O, ADS_H, ADS_CO, ADS_OH, ADS_H2O];

/// `SPECIES_BITS.len()`, named for readability at the (now test-only, see
/// `SPECIES_BITS`'s own doc comment) call sites still measuring it rather
/// than repeating the array just to do so.
pub const NUM_SPECIES: usize = SPECIES_BITS.len();

/// The architectural ceiling on how many species this lattice can ever
/// track -- the one-hot-byte-in-a-`u16`-mask packing `SPECIES_BITS`'s doc
/// comment describes caps any species list at 8 (`0x01..=0x80`), not just
/// today's 5. Per-species arrays that need to stay valid across a future
/// runtime-configurable species set (rather than only today's fixed 5)
/// are sized to this constant instead of `NUM_SPECIES`, so widening the
/// *active* species count later never requires resizing them again --
/// only entries `NUM_SPECIES..MAX_SPECIES` go unused meanwhile.
pub const MAX_SPECIES: usize = 8;

/// Apply a packed `(reactant_mask << 8) | product_mask` transition to a
/// site's current occupancy byte: clear the reactant's bits and OR in the
/// product's. Shared by `engine.rs` (which also handles trajectory
/// logging/migration around this) and `occupancy.rs` (which only needs
/// the byte-level transition itself), so this one-line rule exists in
/// exactly one place.
#[inline(always)]
pub fn apply_transition(current: u8, transition: u16) -> u8 {
    let reactant_mask = (transition >> 8) as u8;
    let product_mask = (transition & 0xFF) as u8;
    (current & !reactant_mask) | product_mask
}

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
    /// Lattice width in sites.
    pub width: usize,
    /// Lattice height in sites.
    pub height: usize,
}

impl SiteLattice {
    /// Open (creating and zero-extending if necessary) the backing lattice
    /// file at `path` and map `width * height` bytes of it read-write.
    pub fn open(path: impl AsRef<Path>, width: usize, height: usize) -> io::Result<Self> {
        let len = width.checked_mul(height).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("lattice of {width}x{height} overflows usize"),
            )
        })?;
        // `engine.rs`'s `TrajectoryRecord.site_idx` and the global
        // reaction-id math both narrow a site's flat index to `u32` --
        // fine on a 64-bit `usize`, which comfortably exceeds `u32::MAX`
        // (~4.29 billion) long before this check does, but a lattice
        // genuinely that large would silently wrap those logged indices
        // rather than fail loudly. Reject it here instead, at construction
        // time, rather than let corruption surface downstream in a
        // trajectory log no one's watching this check for.
        if len > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "lattice of {width}x{height} ({len} sites) exceeds u32::MAX sites \
                     ({}); site indices would silently wrap in the trajectory log",
                    u32::MAX
                ),
            ));
        }

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

        Ok(Self {
            mmap,
            width,
            height,
        })
    }

    /// Flat row-major byte view: `width * height` bytes, one per site,
    /// index `y * width + x`. No 2D indirection is exposed on purpose --
    /// callers compute the linear index themselves so the compiler keeps
    /// scans over it a single contiguous walk.
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap[..]
    }

    /// Mutable counterpart to [`as_slice`](Self::as_slice).
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.mmap[..]
    }

    /// Read one site's occupancy byte by flat index.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> u8 {
        self.mmap[idx]
    }

    /// Write one site's occupancy byte by flat index.
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

/// One 320-byte, five-cache-line-resident block of `LANES` reaction
/// records laid out Array-of-Structures-of-Arrays: each *field* is
/// contiguous across the block's lanes, but blocks themselves are
/// addressed together as single units.
///
/// `#[repr(C, align(64))]`: `align(64)` pins every block to a 64-byte
/// hardware cache-line boundary on all mainstream x86_64/AArch64 parts, so
/// a worker reading or updating one block's propensities never straddles
/// a line at the block's own boundary, and never false-shares with a
/// neighboring block a different worker is writing. `repr(C)` is required
/// alongside it: without it the compiler is free to reorder these fields,
/// which would silently desync this struct's in-memory layout from the
/// byte-for-byte format `reactions.lut` is written in on disk. The five
/// field arrays are sized so the struct is *exactly* 320 bytes (5 cache
/// lines) with no trailing padding: `4*32 + 1*32 + 2*32 + 2*32 + 1*32 =
/// 320`.
///
/// **This block used to be exactly one 64-byte cache line with `LANES =
/// 8`.** Widening `transition_a`/`transition_b` from `u8` to `u16` (to
/// support more than 4 species -- see `SPECIES_BITS`'s doc comment) makes
/// each lane 10 bytes instead of 8; `10 * LANES` only lands back on a
/// clean multiple of 64 (needed so every block in the array stays
/// 64-byte-aligned, and so no implicit tail padding sneaks in -- see the
/// safety note on `write_lut`) at `LANES = 32`, giving a 320-byte, 5-line
/// block instead of a 1-line one. A smaller `LANES` would need explicit
/// reserved padding bytes to reach a 64-byte multiple; `LANES = 32` is
/// the smallest lane count with zero waste, so that's what this uses
/// instead of introducing an unused field.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct ReactionLutBlock {
    /// Fixed-point Q16.16 propensity-contributing rate constant, one per
    /// lane. See `gillespie::FixedPoint` for the encoding this feeds.
    pub rate_q16: [u32; Self::LANES],
    /// Composition-rejection bin this lane's reaction currently belongs to.
    pub bin_id: [u8; Self::LANES],
    /// Packed `(reactant_mask << 8) | product_mask` transition for the
    /// reaction's primary site -- the only site touched for a
    /// monomolecular (adsorption/desorption) reaction. Each mask is a
    /// one-hot byte (see `SPECIES_BITS`), not a nibble.
    pub transition_a: [u16; Self::LANES],
    /// Packed `(reactant_mask << 8) | product_mask` transition for a
    /// *second*, spatially adjacent site. Meaningful only when
    /// `is_bimolecular` is set (e.g. a Langmuir-Hinshelwood surface
    /// reaction like `O* + CO* -> CO2 + 2*`, where site A's `O*` and site
    /// B's `CO*` both clear to `VACANT` in the same event); `0` and
    /// ignored otherwise.
    pub transition_b: [u16; Self::LANES],
    /// `1` if this lane is a two-site (bimolecular) reaction that touches
    /// both `transition_a`'s and `transition_b`'s sites atomically; `0`
    /// for an ordinary single-site (monomolecular) reaction that only
    /// touches `transition_a`'s site. Kept as a full byte per lane (rather
    /// than stealing a bit from another field) so this block's layout
    /// stays simple fixed-width arrays with no bit-packing to unpack on
    /// the hot path.
    pub is_bimolecular: [u8; Self::LANES],
}

impl ReactionLutBlock {
    /// Reactions packed per 320-byte block. See the struct-level layout
    /// comment for how this lane count was chosen to divide evenly into
    /// whole 64-byte cache lines with zero padding, given `u16`-wide
    /// transition fields.
    pub const LANES: usize = 32;
}

// Compile-time invariants the `unsafe` reinterpretation in `ReactionLut`
// below depends on: if either ever drifts (e.g. a field is added), this
// fails the build instead of silently corrupting reads.
const _: () = assert!(std::mem::size_of::<ReactionLutBlock>() == 320);
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

/// Which selection semantics a `reactions.lut` file was built for. Both
/// kinds share the exact same on-disk `ReactionLutBlock` layout -- what
/// differs is how the *engine* interprets `bin_id` and picks a site to fire
/// on, not the bytes themselves. Recorded in the file's magic header so
/// `main.rs` can dispatch to the right engine path automatically instead of
/// relying on a CLI flag staying in sync with how the file was built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutKind {
    /// `bin_id` is a composition-rejection magnitude class
    /// (`31 - rate_q16.leading_zeros()`); every reaction is an
    /// independent, always-available channel with a fixed propensity.
    /// Written by `main.rs`'s `--generate-lut` synthetic demo generator.
    /// Selected via `gillespie::CompositionTable`/`GillespieDomain`.
    Static,
    /// `bin_id` is a quantile-bucket index (0..`occupancy::
    /// BUCKETS_PER_SPECIES`) for monomolecular reactions, unused for
    /// bimolecular ones; propensity is `rate_q16` scaled by how many
    /// lattice sites currently match the reaction's reactant state.
    /// Written by `oc20_ingest`. Selected via `occupancy::OccupancyCounters`.
    OccupancyGated,
}

impl LutKind {
    const fn magic(self) -> &'static [u8; 8] {
        match self {
            LutKind::Static => b"KMCSTAT1",
            LutKind::OccupancyGated => b"KMCOCC01",
        }
    }

    fn from_magic(magic: &[u8]) -> io::Result<Self> {
        match magic {
            m if m == LutKind::Static.magic() => Ok(LutKind::Static),
            m if m == LutKind::OccupancyGated.magic() => Ok(LutKind::OccupancyGated),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "reactions.lut has an unrecognized magic header",
            )),
        }
    }
}

/// Every `reactions.lut` file starts with one 64-byte (cache-line-sized)
/// header: an 8-byte magic (see `LutKind`) followed by 56 reserved/zeroed
/// bytes. 64 rather than a bare 8 bytes specifically so the block array
/// that follows stays 64-byte aligned relative to the mmap's page-aligned
/// base -- `ReactionLutBlock` is `repr(align(64))`, and offsetting by
/// anything not a multiple of 64 would break that invariant for every
/// block after the first.
const LUT_HEADER_SIZE: usize = 64;

/// How many of `LUT_HEADER_SIZE`'s bytes are available for `SpeciesTable`
/// to encode into, after the 8-byte magic.
const SPECIES_HEADER_CAPACITY: usize = LUT_HEADER_SIZE - 8;

/// Runtime species identity self-described inside a `reactions.lut`'s own
/// header -- the LUT's own record of "what does bit 0x01 mean" so any
/// binary that opens it (not just the tool that built it) can label
/// pressures, CSV columns, and log output without needing the config file
/// that built the LUT in the first place. See `ReactionLut::species`.
///
/// Deliberately holds only a one-hot bit and a display name per species --
/// everything else a config file might carry for a species (gas source,
/// stoichiometry, adsorption role...) is `oc20_ingest`-internal build
/// information with no reason to survive into the built artifact.
///
/// The empty table (`SpeciesTable::default()`) is what every LUT built
/// before this type existed decodes to (a zeroed header reads as `count =
/// 0`), and what a `Static` demo LUT carries deliberately, since it has no
/// real species identity to self-describe.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpeciesTable {
    entries: Vec<(u8, String)>,
}

impl SpeciesTable {
    /// Build a table from `(one-hot bit, display name)` pairs, in the
    /// order they correspond to reaction-record species indices. Errors
    /// on anything `ReactionLut::open` would later refuse to decode from
    /// a file, so a caller finds out at build time rather than being
    /// handed a `reactions.lut` that fails to reopen: more than
    /// `MAX_SPECIES` entries, a non-one-hot bit, a duplicate bit, or a
    /// total encoding that wouldn't fit the header's reserved space.
    pub fn new(entries: Vec<(u8, String)>) -> Result<Self, String> {
        if entries.len() > MAX_SPECIES {
            return Err(format!(
                "{} species exceeds the architectural ceiling of {MAX_SPECIES} \
                 (see layout::SPECIES_BITS's doc comment for why)",
                entries.len()
            ));
        }
        let mut encoded_size = 1usize; // the leading species-count byte
        for (i, (bit, name)) in entries.iter().enumerate() {
            if bit.count_ones() != 1 {
                return Err(format!("species `{name}`'s bit {bit:#04x} is not one-hot"));
            }
            if entries[..i].iter().any(|(b, _)| b == bit) {
                return Err(format!("species bit {bit:#04x} is named more than once"));
            }
            encoded_size += 2 + name.len();
        }
        if encoded_size > SPECIES_HEADER_CAPACITY {
            return Err(format!(
                "species table needs {encoded_size} bytes to encode, only \
                 {SPECIES_HEADER_CAPACITY} are available in the LUT header"
            ));
        }
        Ok(SpeciesTable { entries })
    }

    /// How many species this table names.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether this table names zero species.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// This table's species index for `bit`, if any -- the runtime
    /// counterpart to `occupancy`'s compile-time `species_index` against
    /// the fixed `SPECIES_BITS` list.
    pub fn index_of(&self, bit: u8) -> Option<usize> {
        self.entries.iter().position(|&(b, _)| b == bit)
    }

    /// This table's species index for the species displayed as `name`, if
    /// any -- the name-based counterpart to `index_of` (bit-based). Used
    /// by `main.rs`'s `--pressure <name> <value>` flag to resolve a
    /// user-typed species name against a LUT's own self-described
    /// identity, without the caller needing to already know that species'
    /// one-hot bit.
    pub fn index_of_name(&self, name: &str) -> Option<usize> {
        self.entries.iter().position(|(_, n)| n == name)
    }

    /// The display name for the species at `index`, if this table names
    /// one there.
    pub fn name(&self, index: usize) -> Option<&str> {
        self.entries.get(index).map(|(_, name)| name.as_str())
    }

    /// The one-hot bit for the species at `index`, if this table names
    /// one there.
    pub fn bit(&self, index: usize) -> Option<u8> {
        self.entries.get(index).map(|(bit, _)| *bit)
    }

    /// Serialize into `header` (must be exactly `SPECIES_HEADER_CAPACITY`
    /// bytes -- the LUT header's reserved region past the 8-byte magic):
    /// a leading count byte, then each entry as `[bit, name_len,
    /// name_bytes...]`. Infallible: `new`'s validation already guarantees
    /// this table's encoding fits, and both other constructors
    /// (`Default`, `decode_from`) only ever produce already-valid tables.
    fn encode_into(&self, header: &mut [u8]) {
        debug_assert_eq!(header.len(), SPECIES_HEADER_CAPACITY);
        header[0] = self.entries.len() as u8;
        let mut offset = 1usize;
        for (bit, name) in &self.entries {
            let name_bytes = name.as_bytes();
            header[offset] = *bit;
            header[offset + 1] = name_bytes.len() as u8;
            header[offset + 2..offset + 2 + name_bytes.len()].copy_from_slice(name_bytes);
            offset += 2 + name_bytes.len();
        }
    }

    /// Decode a table from `header` (must be exactly
    /// `SPECIES_HEADER_CAPACITY` bytes). Never panics on malformed input
    /// -- these are bytes read from a file that might be corrupted or
    /// adversarial (see `fuzz/fuzz_targets/reactions_lut_parse.rs`), so
    /// any inconsistency (a count exceeding `MAX_SPECIES`, a length prefix
    /// running past the header's end, a non-one-hot bit, a duplicate bit,
    /// or non-UTF-8 name bytes) is a decode error, not a panic.
    fn decode_from(header: &[u8]) -> io::Result<Self> {
        debug_assert_eq!(header.len(), SPECIES_HEADER_CAPACITY);
        let bad_data = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, msg.to_string());

        let count = header[0] as usize;
        if count > MAX_SPECIES {
            return Err(bad_data(&format!(
                "reactions.lut header claims {count} species, exceeding the {MAX_SPECIES} ceiling"
            )));
        }

        let mut entries: Vec<(u8, String)> = Vec::with_capacity(count);
        let mut offset = 1usize;
        for _ in 0..count {
            if offset + 2 > header.len() {
                return Err(bad_data("reactions.lut species header is truncated"));
            }
            let bit = header[offset];
            let name_len = header[offset + 1] as usize;
            offset += 2;

            if bit.count_ones() != 1 {
                return Err(bad_data(&format!(
                    "reactions.lut species bit {bit:#04x} is not one-hot"
                )));
            }
            if entries.iter().any(|&(b, _)| b == bit) {
                return Err(bad_data(&format!(
                    "reactions.lut names species bit {bit:#04x} more than once"
                )));
            }
            if offset + name_len > header.len() {
                return Err(bad_data("reactions.lut species name overruns the header"));
            }
            let name = std::str::from_utf8(&header[offset..offset + name_len])
                .map_err(|_| bad_data("reactions.lut species name is not valid UTF-8"))?
                .to_string();
            offset += name_len;

            entries.push((bit, name));
        }
        Ok(SpeciesTable { entries })
    }
}

/// Read-only, memory-mapped view over a prebuilt `reactions.lut` file: an
/// `LUT_HEADER_SIZE`-byte magic header followed by a flat run of
/// `ReactionLutBlock`s, so mapping it (past the header) is just a pointer
/// cast over the file's bytes.
pub struct ReactionLut {
    // Kept only to hold the mapping alive for the lifetime of `blocks`;
    // never read directly after `open` validates and casts it.
    _mmap: Mmap,
    blocks: *const ReactionLutBlock,
    len: usize,
    kind: LutKind,
    species: SpeciesTable,
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
    /// Map `path`, validate its magic header, and validate the remaining
    /// bytes as a whole number of correctly-aligned `ReactionLutBlock`s.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;

        // SAFETY: read-only mapping of a file opened read-only above; this
        // process does not concurrently write to it, and `reactions.lut` is
        // treated as a static build artifact for the lifetime of the run.
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < LUT_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "reactions.lut is too short to contain a header",
            ));
        }
        let kind = LutKind::from_magic(&mmap[0..8])?;
        let species = SpeciesTable::decode_from(&mmap[8..LUT_HEADER_SIZE])?;

        let body_len = mmap.len() - LUT_HEADER_SIZE;
        let block_size = std::mem::size_of::<ReactionLutBlock>();
        if !body_len.is_multiple_of(block_size) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "reactions.lut body length is not a whole number of 64-byte AoSOA blocks",
            ));
        }

        // SAFETY-relevant: `LUT_HEADER_SIZE` (64) is itself a multiple of
        // `align_of::<ReactionLutBlock>()` (64), so offsetting a
        // page-aligned base by it preserves 64-byte alignment -- the
        // `align_offset` check below is defensive, not expected to fire.
        let ptr = unsafe { mmap.as_ptr().add(LUT_HEADER_SIZE) };
        if ptr.align_offset(std::mem::align_of::<ReactionLutBlock>()) != 0 {
            // mmap bases are page-aligned (>= 4096 bytes) on every platform
            // this project targets, which is always a multiple of 64; this
            // branch exists as a defensive check, not an expected path.
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "reactions.lut block array is not 64-byte aligned",
            ));
        }

        let len = body_len / block_size;
        let blocks = ptr as *const ReactionLutBlock;

        Ok(Self {
            _mmap: mmap,
            blocks,
            len,
            kind,
            species,
        })
    }

    /// Total number of reaction records across every block.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this LUT holds zero reaction records.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Which engine (`Static` or `OccupancyGated`) this LUT's magic
    /// header dispatches to.
    #[inline(always)]
    pub fn kind(&self) -> LutKind {
        self.kind
    }

    /// This LUT's self-described species identity (see `SpeciesTable`) --
    /// empty for a `Static` demo LUT or any LUT built before this table
    /// existed.
    #[inline(always)]
    pub fn species(&self) -> &SpeciesTable {
        &self.species
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
    /// id, without materializing an intermediate `ReactionLutBlock` for
    /// the caller to index into themselves.
    #[inline(always)]
    pub fn rate_of(&self, reaction_id: usize) -> ReactionRecord {
        let (block_idx, lane) = block_and_lane(reaction_id);
        let block = &self.as_slice()[block_idx];
        ReactionRecord {
            rate_q16: block.rate_q16[lane],
            bin_id: block.bin_id[lane],
            transition_a: block.transition_a[lane],
            transition_b: block.transition_b[lane],
            is_bimolecular: block.is_bimolecular[lane] != 0,
        }
    }
}

/// One reaction record prior to AoSOA packing (or after unpacking a single
/// lane back out via `ReactionLut::rate_of`) -- everything
/// `pack_records_into_blocks` needs to place into one `ReactionLutBlock`
/// lane. See `ReactionLutBlock`'s field docs for what each member means.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReactionRecord {
    /// See `ReactionLutBlock::rate_q16`.
    pub rate_q16: u32,
    /// See `ReactionLutBlock::bin_id`.
    pub bin_id: u8,
    /// See `ReactionLutBlock::transition_a`.
    pub transition_a: u16,
    /// See `ReactionLutBlock::transition_b`.
    pub transition_b: u16,
    /// See `ReactionLutBlock::is_bimolecular`.
    pub is_bimolecular: bool,
}

/// Pack `ReactionRecord`s into `ReactionLutBlock`s, sorting by `bin_id`
/// first (the invariant `CompositionTable::build` relies on to collapse
/// bin membership to a `[start, count)` range). Used by both the synthetic
/// demo generator (`main.rs`) and the real data-ingestion tool
/// (`bin/oc20_ingest.rs`) so the on-disk packing logic exists in exactly
/// one place.
pub fn pack_records_into_blocks(mut records: Vec<ReactionRecord>) -> Vec<ReactionLutBlock> {
    records.sort_by_key(|r| r.bin_id);

    let block_count = records.len().div_ceil(ReactionLutBlock::LANES).max(1);
    let mut blocks = Vec::with_capacity(block_count);

    for chunk in records.chunks(ReactionLutBlock::LANES) {
        let mut rate_q16 = [0u32; ReactionLutBlock::LANES];
        let mut bin_id = [0u8; ReactionLutBlock::LANES];
        let mut transition_a = [0u16; ReactionLutBlock::LANES];
        let mut transition_b = [0u16; ReactionLutBlock::LANES];
        let mut is_bimolecular = [0u8; ReactionLutBlock::LANES];

        for (lane, r) in chunk.iter().enumerate() {
            rate_q16[lane] = r.rate_q16;
            bin_id[lane] = r.bin_id;
            transition_a[lane] = r.transition_a;
            transition_b[lane] = r.transition_b;
            is_bimolecular[lane] = r.is_bimolecular as u8;
        }

        blocks.push(ReactionLutBlock {
            rate_q16,
            bin_id,
            transition_a,
            transition_b,
            is_bimolecular,
        });
    }

    blocks
}

/// Write `kind`'s magic header followed by `blocks` verbatim to `path`, as
/// the raw bytes `ReactionLut::open` expects to map back in. The header's
/// species-identity region is left empty (`SpeciesTable::default()`) --
/// see `write_lut_with_species` for a build that stamps one in.
pub fn write_lut(
    path: impl AsRef<Path>,
    kind: LutKind,
    blocks: &[ReactionLutBlock],
) -> io::Result<()> {
    write_lut_with_species(path, kind, blocks, &SpeciesTable::default())
}

/// Like `write_lut`, but also stamps `species`'s runtime identity table
/// into the header's reserved bytes, so a later `ReactionLut::open` can
/// recover species names/bits (see `ReactionLut::species`) without
/// needing the config file that built this LUT.
pub fn write_lut_with_species(
    path: impl AsRef<Path>,
    kind: LutKind,
    blocks: &[ReactionLutBlock],
    species: &SpeciesTable,
) -> io::Result<()> {
    let mut header = [0u8; LUT_HEADER_SIZE];
    header[0..8].copy_from_slice(kind.magic());
    species.encode_into(&mut header[8..LUT_HEADER_SIZE]);

    // SAFETY: `ReactionLutBlock` is `repr(C, align(64))`, `Copy`, and every
    // field is a plain fixed-width integer array with no padding bytes
    // (enforced by the `size_of::<ReactionLutBlock>() == 320` assertion
    // above), so reinterpreting `&[ReactionLutBlock]` as `&[u8]` for the
    // duration of this write is a sound, lossless byte-for-byte view --
    // there is no uninitialized padding to expose, and no lifetime hazard
    // since the byte slice does not outlive `blocks`.
    let bytes = unsafe {
        slice::from_raw_parts(blocks.as_ptr() as *const u8, std::mem::size_of_val(blocks))
    };

    let mut file = std::fs::File::create(path)?;
    file.write_all(&header)?;
    file.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_path;

    #[test]
    fn block_and_lane_computes_correct_coordinates() {
        assert_eq!(block_and_lane(0), (0, 0));
        assert_eq!(block_and_lane(31), (0, 31));
        assert_eq!(block_and_lane(32), (1, 0));
        assert_eq!(block_and_lane(95), (2, 31));
    }

    fn rec(rate_q16: u32, bin_id: u8, transition_a: u16) -> ReactionRecord {
        ReactionRecord {
            rate_q16,
            bin_id,
            transition_a,
            transition_b: 0,
            is_bimolecular: false,
        }
    }

    #[test]
    fn pack_records_into_blocks_sorts_by_bin_id_and_pads_last_block() {
        let records = vec![rec(100, 3, 0x12), rec(50, 1, 0x21), rec(10, 0, 0x00)];
        let blocks = pack_records_into_blocks(records);
        assert_eq!(blocks.len(), 1);

        let b = &blocks[0];
        assert_eq!((b.rate_q16[0], b.bin_id[0]), (10, 0));
        assert_eq!((b.rate_q16[1], b.bin_id[1]), (50, 1));
        assert_eq!((b.rate_q16[2], b.bin_id[2]), (100, 3));
        assert_eq!(b.transition_a[2], 0x12);

        for lane in 3..ReactionLutBlock::LANES {
            assert_eq!(b.rate_q16[lane], 0);
            assert_eq!(b.bin_id[lane], 0);
            assert_eq!(b.transition_a[lane], 0);
            assert_eq!(b.is_bimolecular[lane], 0);
        }
    }

    #[test]
    fn pack_records_into_blocks_empty_input_yields_no_blocks() {
        assert!(pack_records_into_blocks(Vec::new()).is_empty());
    }

    #[test]
    fn write_and_reopen_lut_round_trips_reaction_rates() {
        let records = vec![
            rec(10, 0, 0x01),
            rec(20, 1, 0x02),
            rec(30, 2, 0x04),
            rec(40, 31, 0x00),
        ];
        let blocks = pack_records_into_blocks(records.clone());
        let path = temp_path("lut_roundtrip");
        write_lut(&path, LutKind::Static, &blocks).unwrap();

        let lut = ReactionLut::open(&path).unwrap();
        assert_eq!(lut.kind(), LutKind::Static);
        assert_eq!(lut.len(), blocks.len());
        for (i, &expected) in records.iter().enumerate() {
            assert_eq!(lut.rate_of(i), expected);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_and_reopen_lut_round_trips_bimolecular_reaction() {
        let records = vec![ReactionRecord {
            rate_q16: 500,
            bin_id: 4,
            transition_a: (ADS_O as u16) << 8,  // O* -> vacant
            transition_b: (ADS_CO as u16) << 8, // CO* -> vacant
            is_bimolecular: true,
        }];
        let blocks = pack_records_into_blocks(records.clone());
        let path = temp_path("lut_bimolecular_roundtrip");
        write_lut(&path, LutKind::OccupancyGated, &blocks).unwrap();

        let lut = ReactionLut::open(&path).unwrap();
        assert_eq!(lut.kind(), LutKind::OccupancyGated);
        assert_eq!(lut.rate_of(0), records[0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn species_table_round_trips_through_write_lut_with_species() {
        let species = SpeciesTable::new(vec![
            (ADS_O, "O".to_string()),
            (ADS_H, "H".to_string()),
            (ADS_CO, "CO".to_string()),
        ])
        .unwrap();
        let path = temp_path("lut_species_roundtrip");
        write_lut_with_species(&path, LutKind::OccupancyGated, &[], &species).unwrap();

        let lut = ReactionLut::open(&path).unwrap();
        let decoded = lut.species();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded.bit(0), Some(ADS_O));
        assert_eq!(decoded.name(0), Some("O"));
        assert_eq!(decoded.bit(1), Some(ADS_H));
        assert_eq!(decoded.name(1), Some("H"));
        assert_eq!(decoded.bit(2), Some(ADS_CO));
        assert_eq!(decoded.name(2), Some("CO"));
        assert_eq!(decoded.index_of(ADS_CO), Some(2));
        assert_eq!(decoded.index_of(ADS_OH), None);
        assert_eq!(decoded.index_of_name("CO"), Some(2));
        assert_eq!(decoded.index_of_name("OH"), None);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_lut_leaves_species_table_empty() {
        let path = temp_path("lut_species_default_empty");
        write_lut(&path, LutKind::Static, &[]).unwrap();

        let lut = ReactionLut::open(&path).unwrap();
        assert!(lut.species().is_empty());
        assert_eq!(lut.species(), &SpeciesTable::default());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn species_table_new_rejects_more_than_max_species_entries() {
        let entries: Vec<(u8, String)> = (0..(MAX_SPECIES + 1) as u32)
            .map(|i| (1u8 << (i % 8), format!("S{i}")))
            .collect();
        assert!(SpeciesTable::new(entries).is_err());
    }

    #[test]
    fn species_table_new_rejects_non_one_hot_bit() {
        assert!(SpeciesTable::new(vec![(0x03, "bad".to_string())]).is_err());
        assert!(SpeciesTable::new(vec![(0x00, "vacant".to_string())]).is_err());
    }

    #[test]
    fn species_table_new_rejects_duplicate_bit() {
        assert!(SpeciesTable::new(vec![
            (ADS_O, "O".to_string()),
            (ADS_O, "O-again".to_string()),
        ])
        .is_err());
    }

    #[test]
    fn species_table_new_rejects_encoding_that_overflows_the_header_budget() {
        // 8 species x a name long enough that the total can't fit in the
        // header's 56 reserved bytes (1 count byte + 8 x (2 + name_len)).
        let entries: Vec<(u8, String)> = (0..8)
            .map(|i| (1u8 << i, "a_very_long_species_name".to_string()))
            .collect();
        assert!(SpeciesTable::new(entries).is_err());
    }

    #[test]
    fn open_rejects_species_header_claiming_more_than_max_species() {
        let path = temp_path("lut_species_header_too_many");
        let mut bytes = vec![0u8; LUT_HEADER_SIZE];
        bytes[0..8].copy_from_slice(LutKind::Static.magic());
        bytes[8] = (MAX_SPECIES + 1) as u8; // claims more species than the ceiling allows
        std::fs::write(&path, &bytes).unwrap();
        assert!(ReactionLut::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_species_header_with_non_one_hot_bit() {
        let path = temp_path("lut_species_header_bad_bit");
        let mut bytes = vec![0u8; LUT_HEADER_SIZE];
        bytes[0..8].copy_from_slice(LutKind::Static.magic());
        bytes[8] = 1; // count = 1
        bytes[9] = 0x03; // not one-hot
        bytes[10] = 1; // name_len = 1
        bytes[11] = b'X';
        std::fs::write(&path, &bytes).unwrap();
        assert!(ReactionLut::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_species_header_with_duplicate_bit() {
        let path = temp_path("lut_species_header_dup_bit");
        let mut bytes = vec![0u8; LUT_HEADER_SIZE];
        bytes[0..8].copy_from_slice(LutKind::Static.magic());
        bytes[8] = 2; // count = 2
        bytes[9] = ADS_O; // first entry: bit=ADS_O, name_len=1, "A"
        bytes[10] = 1;
        bytes[11] = b'A';
        bytes[12] = ADS_O; // second entry: same bit again
        bytes[13] = 1;
        bytes[14] = b'B';
        std::fs::write(&path, &bytes).unwrap();
        assert!(ReactionLut::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_species_header_with_name_overrunning_the_header() {
        let path = temp_path("lut_species_header_name_overrun");
        let mut bytes = vec![0u8; LUT_HEADER_SIZE];
        bytes[0..8].copy_from_slice(LutKind::Static.magic());
        bytes[8] = 1; // count = 1
        bytes[9] = ADS_O;
        bytes[10] = 255; // name_len claims far more bytes than the header has left
        std::fs::write(&path, &bytes).unwrap();
        assert!(ReactionLut::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_species_header_with_invalid_utf8_name() {
        let path = temp_path("lut_species_header_bad_utf8");
        let mut bytes = vec![0u8; LUT_HEADER_SIZE];
        bytes[0..8].copy_from_slice(LutKind::Static.magic());
        bytes[8] = 1; // count = 1
        bytes[9] = ADS_O;
        bytes[10] = 1; // name_len = 1
        bytes[11] = 0xFF; // not valid UTF-8 on its own
        std::fs::write(&path, &bytes).unwrap();
        assert!(ReactionLut::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_file_with_length_not_multiple_of_block_size() {
        let path = temp_path("lut_bad_len");
        let mut bytes = vec![0u8; LUT_HEADER_SIZE];
        bytes[0..8].copy_from_slice(LutKind::Static.magic());
        bytes.extend_from_slice(&[0u8; 50]); // body not a multiple of 64
        std::fs::write(&path, &bytes).unwrap();
        assert!(ReactionLut::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_file_shorter_than_header() {
        let path = temp_path("lut_too_short");
        std::fs::write(&path, [0u8; 10]).unwrap();
        assert!(ReactionLut::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_unrecognized_magic() {
        let path = temp_path("lut_bad_magic");
        let mut bytes = vec![0u8; LUT_HEADER_SIZE];
        bytes[0..8].copy_from_slice(b"BOGUSMAG");
        std::fs::write(&path, &bytes).unwrap();
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

    /// A code audit flagged `u32` site-index overflow at very large
    /// lattice sizes as untested: `engine.rs`'s `TrajectoryRecord.site_idx`
    /// narrows a site's flat index to `u32`, so a lattice with more than
    /// `u32::MAX` sites would silently wrap it. `open` now rejects that
    /// case outright rather than letting it surface downstream. Chosen
    /// dimensions (100,000 x 100,000 = 10 billion sites) exceed `u32::MAX`
    /// (~4.29 billion) without needing to actually allocate anything --
    /// this must fail before `set_len`/`mmap`, not after.
    #[test]
    fn open_rejects_lattice_dimensions_exceeding_u32_max_sites() {
        let path = temp_path("lattice_too_large");
        let result = SiteLattice::open(&path, 100_000, 100_000);
        assert!(
            result.is_err(),
            "10 billion sites must be rejected, not silently wrapped"
        );
        assert!(
            !path.exists(),
            "must fail before creating/sizing the backing file"
        );
    }

    /// An error-handling audit found `width.checked_mul(height)` was
    /// followed by `.expect(...)`, so dimensions large enough to overflow
    /// `usize` itself (reachable via `--lattice-width`/`--lattice-height`,
    /// both plain CLI-parsed integers) panicked instead of returning the
    /// same clean `io::Error` the `u32::MAX` check above already uses for a
    /// smaller version of the same problem. Chosen dimensions each exceed
    /// 2^32, so their product exceeds `usize::MAX` on a 64-bit target
    /// without needing to actually allocate anything.
    #[test]
    fn open_rejects_lattice_dimensions_overflowing_usize() {
        let path = temp_path("lattice_usize_overflow");
        let huge = 1usize << 40;
        let result = SiteLattice::open(&path, huge, huge);
        assert!(
            result.is_err(),
            "a usize-overflowing product must be rejected, not panic"
        );
        assert!(
            !path.exists(),
            "must fail before creating/sizing the backing file"
        );
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

    proptest::proptest! {
        /// `apply_transition`'s doc comment states its contract in prose
        /// ("clear the reactant's bits and OR in the product's") but no
        /// existing test checks that contract independently of the
        /// one-line formula that implements it -- every example test that
        /// exercises this function does so indirectly, through a handful
        /// of hand-picked reaction records elsewhere. Checked here as
        /// three properties any bit of `current`/`reactant_mask`/
        /// `product_mask` must satisfy for *every* representable byte
        /// triple, independent of the implementation:
        /// - every product bit ends up set (a reaction's product always
        ///   lands, regardless of prior occupancy);
        /// - every reactant bit *not* also a product bit ends up cleared
        ///   (the reactant is consumed, unless the same species is also
        ///   produced);
        /// - every bit outside both masks is left exactly as it was
        ///   (a transition never touches a site's other species bits --
        ///   the property `engine.rs`'s "no site holds >1 species bit"
        ///   corruption check ultimately depends on).
        #[test]
        fn apply_transition_only_touches_reactant_and_product_bits(
            current: u8, reactant_mask: u8, product_mask: u8,
        ) {
            let transition = ((reactant_mask as u16) << 8) | (product_mask as u16);
            let result = apply_transition(current, transition);

            proptest::prop_assert_eq!(
                result & product_mask, product_mask,
                "product bit(s) not set: current={:#010b} reactant={:#010b} \
                 product={:#010b} result={:#010b}",
                current, reactant_mask, product_mask, result
            );

            let consumed_only = reactant_mask & !product_mask;
            proptest::prop_assert_eq!(
                result & consumed_only, 0,
                "reactant-only bit(s) not cleared: current={:#010b} \
                 reactant={:#010b} product={:#010b} result={:#010b}",
                current, reactant_mask, product_mask, result
            );

            let untouched_mask = !(reactant_mask | product_mask);
            proptest::prop_assert_eq!(
                result & untouched_mask, current & untouched_mask,
                "bit(s) outside both masks changed: current={:#010b} \
                 reactant={:#010b} product={:#010b} result={:#010b}",
                current, reactant_mask, product_mask, result
            );
        }
    }
}
