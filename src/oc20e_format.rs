//! Reader for the flat binary formats `scripts/oc20e_format.py` writes:
//! `OC20E003` (single-species monomolecular adsorption/reaction-energy
//! records) and `OC20BI03` (two-species bimolecular records). See that
//! script for the authoritative byte layout both this module and its
//! Python counterpart must stay in lockstep with.
//!
//! Lives in the library (not in `src/bin/oc20_ingest.rs`, the only
//! production caller) for the same reason `layout::ReactionLut::open`
//! does: both parse untrusted on-disk bytes (a `--input`/`--bimolecular-
//! input` file can be truncated by a killed extraction script, corrupted,
//! or from a mismatched format version), and putting the parser in the
//! library is what lets `fuzz/fuzz_targets/` exercise the *real* parsing
//! code path instead of a parallel copy that could drift from it -- the
//! same reasoning `fuzz_targets/reactions_lut_parse.rs`'s own doc comment
//! gives for `ReactionLut::open`.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use crate::layout::SPECIES_BITS;

/// `OC20E003`'s 8-byte magic header, checked at the start of every file
/// [`read_energy_records`] parses.
pub const MAGIC: &[u8; 8] = b"OC20E003";
/// species(1) + energy_mev(4) + sid(4) + has_real_ea(1) + real_ea_mev(4)
/// + metal(1) + facet(2).
const RECORD_SIZE: usize = 17;

/// `OC20BI03`: the parallel bimolecular format `extract_catalysis_hub.py`'s
/// `write_bimolecular_records` writes -- see `scripts/oc20e_format.py` for
/// the authoritative byte layout this must match.
pub const MAGIC_BI: &[u8; 8] = b"OC20BI03";
/// species_a(1) + species_b(1) + energy_mev(4) + sid(4) + ea_mev(4)
/// + metal(1) + facet(2) + is_dissociative(1).
const RECORD_SIZE_BI: usize = 18;

/// One parsed input record: which adsorbate, its relaxed adsorption/
/// reaction energy in eV, the source system/reaction id (kept only for
/// diagnostics), and -- rarely -- a genuine DFT-computed activation energy
/// in eV, when the source publishes one instead of just the reaction
/// energy.
#[derive(Clone, Copy)]
pub struct EnergyRecord {
    /// Index into `SPECIES_BITS`/`oc20_ingest`'s `SPECIES_NAMES`.
    pub species: u8,
    /// Relaxed adsorption/reaction energy, eV.
    pub energy_ev: f64,
    /// Source system/reaction id -- diagnostics only, never read for
    /// anything the resulting `reactions.lut` depends on.
    pub sid: u32,
    /// A genuine DFT-computed activation energy, eV, when the source
    /// publishes one instead of just the reaction energy.
    pub real_ea_ev: Option<f64>,
    /// Index into `oc20_ingest`'s `METALS`; 0 = unknown/not tracked.
    pub metal: u8,
    /// Decimal-digit Miller-index encoding (e.g. 111); 0 = unknown.
    pub facet: u16,
}

/// One parsed bimolecular record: two adsorbed species consumed/produced by
/// the same event (indices into `SPECIES_BITS`, same convention as
/// `EnergyRecord::species`), a real DFT-computed forward activation energy
/// (this format never carries a BEP-derived one, since there is no
/// bimolecular BEP relation here), and the forward reaction energy --
/// meaningful when `is_dissociative` is set, to derive a real
/// thermodynamic-consistency reverse rate the same way monomolecular
/// adsorption/desorption pairs already do.
///
/// `is_dissociative` distinguishes which *direction* this real barrier
/// was measured in, since the two site transitions this drives are
/// direction-dependent, not just a magnitude:
/// - `false` (recombination, e.g. CO-oxidation, H2-recombination): both
///   sites start occupied and clear to vacant, releasing a gas product.
///   Forward-only -- there's no thermodynamically meaningful reverse for a
///   gas product that doesn't dissociatively re-adsorb the same way it
///   left.
/// - `true` (dissociative adsorption, e.g. water splitting): both sites
///   start vacant and fill from a gas reactant. Built *both* directions --
///   forward from the real `ea_ev` directly, reverse (associative
///   desorption) via `Ea_rev = ea_ev - energy_ev`, since this direction's
///   reverse genuinely is the same elementary step run backward.
#[derive(Clone, Copy)]
pub struct BiEnergyRecord {
    /// Index into `SPECIES_BITS` for the first site.
    pub species_a: u8,
    /// Index into `SPECIES_BITS` for the second, spatially adjacent site.
    pub species_b: u8,
    /// Source system/reaction id -- diagnostics only.
    pub sid: u32,
    /// The forward reaction energy, eV -- only meaningful (used to derive
    /// a reverse rate) when `is_dissociative` is set.
    pub energy_ev: f64,
    /// The real, DFT-computed forward activation energy, eV.
    pub ea_ev: f64,
    /// Index into `oc20_ingest`'s `METALS`; 0 = unknown/not tracked.
    pub metal: u8,
    /// Decimal-digit Miller-index encoding (e.g. 111); 0 = unknown.
    pub facet: u16,
    /// See this struct's own doc comment for what `true`/`false` mean.
    pub is_dissociative: bool,
}

/// Parse an `OC20E003` file into monomolecular energy records, skipping any
/// record whose species index this build doesn't know (defensive
/// forward-compatibility, not an error).
///
/// Validates the magic header and, before indexing into the byte buffer at
/// all, that the file actually holds as many bytes as its own `count`
/// field claims -- a truncated or corrupted file (e.g. from a killed
/// extraction script) returns `Err` here rather than panicking on an
/// out-of-bounds slice partway through the parse loop.
pub fn read_energy_records(path: &Path) -> io::Result<Vec<EnergyRecord>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;

    if bytes.len() < 12 || &bytes[0..8] != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not an OC20E003 energy file (bad magic/too short)",
        ));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

    let required_len = 12usize.saturating_add(count.saturating_mul(RECORD_SIZE));
    if bytes.len() < required_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "OC20E003 file claims {count} records ({required_len} bytes incl. header) \
                 but is only {} bytes -- truncated or corrupted",
                bytes.len()
            ),
        ));
    }

    let mut records = Vec::with_capacity(count);
    let mut offset = 12usize;
    for _ in 0..count {
        let species = bytes[offset];
        let energy_mev = i32::from_le_bytes(bytes[offset + 1..offset + 5].try_into().unwrap());
        let sid = u32::from_le_bytes(bytes[offset + 5..offset + 9].try_into().unwrap());
        let has_real_ea = bytes[offset + 9] != 0;
        let real_ea_mev = i32::from_le_bytes(bytes[offset + 10..offset + 14].try_into().unwrap());
        let metal = bytes[offset + 14];
        let facet = u16::from_le_bytes(bytes[offset + 15..offset + 17].try_into().unwrap());
        offset += RECORD_SIZE;

        if (species as usize) >= SPECIES_BITS.len() {
            continue; // defensive: ignore any species index this build doesn't know
        }
        records.push(EnergyRecord {
            species,
            energy_ev: energy_mev as f64 / 1000.0,
            sid,
            real_ea_ev: has_real_ea.then_some(real_ea_mev as f64 / 1000.0),
            metal,
            facet,
        });
    }

    Ok(records)
}

/// Parse an `OC20BI03` file into bimolecular energy records. Same
/// truncation/corruption handling as [`read_energy_records`] -- see its
/// doc comment.
pub fn read_bimolecular_records(path: &Path) -> io::Result<Vec<BiEnergyRecord>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;

    if bytes.len() < 12 || &bytes[0..8] != MAGIC_BI {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not an OC20BI03 bimolecular-energy file (bad magic/too short)",
        ));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

    let required_len = 12usize.saturating_add(count.saturating_mul(RECORD_SIZE_BI));
    if bytes.len() < required_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "OC20BI03 file claims {count} records ({required_len} bytes incl. header) \
                 but is only {} bytes -- truncated or corrupted",
                bytes.len()
            ),
        ));
    }

    let mut records = Vec::with_capacity(count);
    let mut offset = 12usize;
    for _ in 0..count {
        let species_a = bytes[offset];
        let species_b = bytes[offset + 1];
        let energy_mev = i32::from_le_bytes(bytes[offset + 2..offset + 6].try_into().unwrap());
        let sid = u32::from_le_bytes(bytes[offset + 6..offset + 10].try_into().unwrap());
        let ea_mev = i32::from_le_bytes(bytes[offset + 10..offset + 14].try_into().unwrap());
        let metal = bytes[offset + 14];
        let facet = u16::from_le_bytes(bytes[offset + 15..offset + 17].try_into().unwrap());
        let is_dissociative = bytes[offset + 17] != 0;
        offset += RECORD_SIZE_BI;

        if (species_a as usize) >= SPECIES_BITS.len() || (species_b as usize) >= SPECIES_BITS.len()
        {
            continue; // defensive: ignore any species index this build doesn't know
        }
        records.push(BiEnergyRecord {
            species_a,
            species_b,
            sid,
            energy_ev: energy_mev as f64 / 1000.0,
            ea_ev: ea_mev as f64 / 1000.0,
            metal,
            facet,
            is_dissociative,
        });
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(tag: &str) -> PathBuf {
        crate::test_support::temp_path(&format!("oc20e_format_{tag}"))
    }

    #[test]
    fn read_energy_records_rejects_bad_magic() {
        let path = temp_path("bad_magic");
        std::fs::write(&path, b"NOTMAGIC\x00\x00\x00\x00").unwrap();
        assert!(read_energy_records(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    /// A truncated/corrupted file (a valid magic header, but a `count`
    /// claiming more records than actually follow -- e.g. from a killed
    /// extraction script) must return `Err`, not panic on an out-of-bounds
    /// slice partway through the parse loop.
    #[test]
    fn read_energy_records_rejects_truncated_file_with_inflated_count() {
        let path = temp_path("energy_truncated");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&500u32.to_le_bytes()); // claims 500 records
        bytes.extend_from_slice(&[0u8; 5]); // far fewer bytes actually follow

        std::fs::write(&path, &bytes).unwrap();
        let result = read_energy_records(&path);
        let _ = std::fs::remove_file(&path);

        assert!(
            result.is_err(),
            "a record count exceeding the file's actual length must be rejected, not panic"
        );
    }

    #[test]
    fn read_bimolecular_records_rejects_bad_magic() {
        let path = temp_path("bi_bad_magic");
        std::fs::write(&path, b"NOTMAGIC\x00\x00\x00\x00").unwrap();
        assert!(read_bimolecular_records(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    /// Same finding as `read_energy_records_rejects_truncated_file_with_
    /// inflated_count`, for the bimolecular reader.
    #[test]
    fn read_bimolecular_records_rejects_truncated_file_with_inflated_count() {
        let path = temp_path("bi_truncated");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC_BI);
        bytes.extend_from_slice(&500u32.to_le_bytes()); // claims 500 records
        bytes.extend_from_slice(&[0u8; 5]); // far fewer bytes actually follow

        std::fs::write(&path, &bytes).unwrap();
        let result = read_bimolecular_records(&path);
        let _ = std::fs::remove_file(&path);

        assert!(
            result.is_err(),
            "a record count exceeding the file's actual length must be rejected, not panic"
        );
    }
}
