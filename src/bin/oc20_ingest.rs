//! Builds a real `reactions.lut` from real adsorption-energy data (OC20 or
//! Catalysis-Hub.org), instead of `kinetica --generate-lut`'s synthetic
//! demo records.
//!
//! Neither source generally publishes transition-state barriers or rate
//! constants -- both give relaxed adsorption/reaction energies (an initial
//! guess -> DFT-relaxed structure and energy), not the height of the
//! energy barrier between them. For the large majority of input records,
//! this tool bridges that gap with two standard, well-known approximations
//! from computational catalysis, rather than pretending the energy alone
//! is a rate constant:
//!
//! 1. A **Bronsted-Evans-Polanyi (BEP) relation** estimates an activation
//!    energy from a reaction energy: `E_a = max(0, alpha * dE_rxn + beta)`.
//!    This is the standard proxy used when explicit NEB-computed barriers
//!    aren't available.
//! 2. **Harmonic transition-state theory / Arrhenius** converts that
//!    barrier into a rate constant: `k = nu * exp(-E_a / (kB * T))`.
//!
//! `alpha`, `beta`, and `nu` are exposed as CLI flags rather than hardcoded
//! because they are illustrative literature-typical defaults, not
//! coefficients fitted to any specific real catalyst system -- see
//! `--help` for the exact defaults.
//!
//! A small minority of input records *do* carry a genuine, non-BEP
//! activation energy (Catalysis-Hub.org has real NEB/dimer-method barriers
//! for a handful of elementary O2/H2/CO dissociative-adsorption steps --
//! see `scripts/extract_catalysis_hub.py`'s `fetch_real_barrier_records`).
//! For those, this tool uses the real barrier directly for the forward
//! (adsorption) direction and only applies the thermodynamic-consistency
//! relation `Ea_rev = Ea_fwd - dE_rxn` to derive the reverse (desorption)
//! barrier -- BEP never enters the picture for these reactions.
//!
//! `--bimolecular-input` adds a third kind of record: real two-site
//! recombination barriers (CO oxidation, `O* + CO* -> CO2 + 2*`; H2
//! recombination, `2 H* -> H2 + 2*`), which the engine represents natively
//! as `is_bimolecular` reactions. These never use BEP and never get a
//! derived reverse reaction. For a *homoatomic* one (both sites the same
//! species, e.g. the H2 case), this tool goes one step further and skips
//! building that species' monomolecular desorption reaction entirely --
//! see `run`'s `replaces_desorption` for why building both would just be
//! the same physical event modeled twice at two levels of approximation,
//! not two distinct reactions.
//!
//! Upstream of this tool, small Python scripts (`scripts/extract_energies.py`
//! for OC20, `scripts/extract_catalysis_hub.py` for Catalysis-Hub.org) each
//! emit the flat binary format this tool consumes as `--input` -- see
//! `scripts/oc20e_format.py` for the exact byte layout.

use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;

use kinetica::layout::{self, ReactionLutBlock};

/// The three adsorbates kinetica's lattice bitflags (layout.rs) model,
/// matching OC20's global adsorbate-index table
/// (`mapping_adsorbates_2020may12.txt`: 0 = *O, 1 = *H, 5 = *CO) as
/// remapped to a dense 0..3 range by the extraction scripts.
const SPECIES_BITS: [u8; 3] = [layout::ADS_O, layout::ADS_H, layout::ADS_CO];
const SPECIES_NAMES: [&str; 3] = ["O", "H", "CO"];

const MAGIC: &[u8; 8] = b"OC20E002";
/// species(1) + energy_mev(4) + sid(4) + has_real_ea(1) + real_ea_mev(4).
const RECORD_SIZE: usize = 14;

/// `OC20BI01`: the parallel bimolecular format `extract_catalysis_hub.py`'s
/// `write_bimolecular_records` writes -- see `scripts/oc20e_format.py` for
/// the authoritative byte layout this must match.
const MAGIC_BI: &[u8; 8] = b"OC20BI01";
/// species_a(1) + species_b(1) + energy_mev(4) + sid(4) + ea_mev(4).
const RECORD_SIZE_BI: usize = 14;

const KB_EV_PER_K: f64 = 8.617_333_262e-5;

#[derive(Debug)]
struct Config {
    input: PathBuf,
    bimolecular_input: Option<PathBuf>,
    out: PathBuf,
    alpha: f64,
    beta_ev: f64,
    nu: f64,
    temperature_k: f64,
}

impl Config {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut args = args;
        let _bin = args.next();

        let mut input = None;
        let mut bimolecular_input = None;
        let mut out = PathBuf::from("reactions.lut");
        let mut alpha = 0.87; // typical BEP slope for atomic adsorption/dissociation steps
        let mut beta_ev = 0.0; // typical BEP intercept, eV
        let mut nu = 1.0e13; // typical harmonic TST attempt frequency, s^-1
        let mut temperature_k = 298.15;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => input = Some(PathBuf::from(next_value(&mut args, "--input")?)),
                "--bimolecular-input" => {
                    bimolecular_input =
                        Some(PathBuf::from(next_value(&mut args, "--bimolecular-input")?))
                }
                "--out" => out = PathBuf::from(next_value(&mut args, "--out")?),
                "--alpha" => alpha = parse_value(&mut args, "--alpha")?,
                "--beta" => beta_ev = parse_value(&mut args, "--beta")?,
                "--nu" => nu = parse_value(&mut args, "--nu")?,
                "--temperature" => temperature_k = parse_value(&mut args, "--temperature")?,
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
            }
        }

        Ok(Self {
            input: input.ok_or_else(|| format!("`--input` is required\n\n{}", usage()))?,
            bimolecular_input,
            out,
            alpha,
            beta_ev,
            nu,
            temperature_k,
        })
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("`{flag}` requires a value"))
}

fn parse_value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    let raw = next_value(args, flag)?;
    raw.parse()
        .map_err(|_| format!("`{flag}` expects a number, got `{raw}`"))
}

fn usage() -> String {
    "oc20_ingest: build reactions.lut from real adsorption-energy data\n\n\
     USAGE:\n    \
       oc20_ingest --input <PATH> [OPTIONS]\n\n\
     OPTIONS:\n    \
       --input <PATH>              Flat binary from an extract_*.py script (required)\n    \
       --bimolecular-input <PATH>  Optional OC20BI01 binary (from\n                                    \
                                    extract_catalysis_hub.py's --bimolecular-out)\n                                    \
                                    carrying real two-site reaction barriers,\n                                    \
                                    e.g. CO oxidation\n    \
       --out <PATH>                Output reactions.lut [default: reactions.lut]\n    \
       --alpha <F>                 BEP relation slope [default: 0.87]\n    \
       --beta <F>                  BEP relation intercept, eV [default: 0.0]\n    \
       --nu <F>                    Arrhenius prefactor, s^-1 [default: 1e13]\n    \
       --temperature <F>           Temperature, K [default: 298.15]\n    \
       -h, --help                  Print this message"
        .to_string()
}

/// One parsed input record: which adsorbate, its relaxed adsorption/
/// reaction energy in eV, the source system/reaction id (kept only for
/// diagnostics), and -- rarely -- a genuine DFT-computed activation energy
/// in eV, when the source publishes one instead of just the reaction
/// energy.
struct EnergyRecord {
    species: u8,
    energy_ev: f64,
    #[allow(dead_code)]
    sid: u32,
    real_ea_ev: Option<f64>,
}

fn read_energy_records(path: &std::path::Path) -> io::Result<Vec<EnergyRecord>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;

    if bytes.len() < 12 || &bytes[0..8] != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not an OC20E002 energy file (bad magic/too short)",
        ));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

    let mut records = Vec::with_capacity(count);
    let mut offset = 12usize;
    for _ in 0..count {
        let species = bytes[offset];
        let energy_mev = i32::from_le_bytes(bytes[offset + 1..offset + 5].try_into().unwrap());
        let sid = u32::from_le_bytes(bytes[offset + 5..offset + 9].try_into().unwrap());
        let has_real_ea = bytes[offset + 9] != 0;
        let real_ea_mev = i32::from_le_bytes(bytes[offset + 10..offset + 14].try_into().unwrap());
        offset += RECORD_SIZE;

        if (species as usize) >= SPECIES_BITS.len() {
            continue; // defensive: ignore any species index this build doesn't know
        }
        records.push(EnergyRecord {
            species,
            energy_ev: energy_mev as f64 / 1000.0,
            sid,
            real_ea_ev: has_real_ea.then_some(real_ea_mev as f64 / 1000.0),
        });
    }

    Ok(records)
}

/// One parsed bimolecular record: two adsorbed species consumed by the
/// same event (indices into `SPECIES_BITS`, same convention as
/// `EnergyRecord::species`), plus a real DFT-computed forward activation
/// energy -- this format never carries a BEP-derived one, since there is
/// no bimolecular BEP relation here (see `oc20e_format.py`).
struct BiEnergyRecord {
    species_a: u8,
    species_b: u8,
    #[allow(dead_code)]
    sid: u32,
    ea_ev: f64,
}

fn read_bimolecular_records(path: &std::path::Path) -> io::Result<Vec<BiEnergyRecord>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;

    if bytes.len() < 12 || &bytes[0..8] != MAGIC_BI {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not an OC20BI01 bimolecular-energy file (bad magic/too short)",
        ));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

    let mut records = Vec::with_capacity(count);
    let mut offset = 12usize;
    for _ in 0..count {
        let species_a = bytes[offset];
        let species_b = bytes[offset + 1];
        // energy_mev (reaction energy) at offset+2..offset+6 is read from
        // the file but not currently used -- kept for future thermodynamic
        // bookkeeping (see oc20e_format.py's field docs).
        let sid = u32::from_le_bytes(bytes[offset + 6..offset + 10].try_into().unwrap());
        let ea_mev = i32::from_le_bytes(bytes[offset + 10..offset + 14].try_into().unwrap());
        offset += RECORD_SIZE_BI;

        if (species_a as usize) >= SPECIES_BITS.len() || (species_b as usize) >= SPECIES_BITS.len()
        {
            continue; // defensive: ignore any species index this build doesn't know
        }
        records.push(BiEnergyRecord {
            species_a,
            species_b,
            sid,
            ea_ev: ea_mev as f64 / 1000.0,
        });
    }

    Ok(records)
}

/// The activation energy for a reaction: a genuine DFT-computed barrier
/// when the source provided one, otherwise the BEP estimate from its
/// reaction energy.
#[inline]
fn activation_energy_ev(delta_e_rxn_ev: f64, real_ea_ev: Option<f64>, config: &Config) -> f64 {
    real_ea_ev.unwrap_or_else(|| (config.alpha * delta_e_rxn_ev + config.beta_ev).max(0.0))
}

/// Arrhenius: turn an activation energy into a rate constant.
#[inline]
fn rate_from_activation(activation_ev: f64, config: &Config) -> f64 {
    config.nu * (-activation_ev / (KB_EV_PER_K * config.temperature_k)).exp()
}

fn main() -> std::process::ExitCode {
    let config = match Config::parse(std::env::args()) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("{msg}");
            return std::process::ExitCode::FAILURE;
        }
    };

    match run(&config) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("oc20_ingest: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(config: &Config) -> io::Result<()> {
    let energy_records = read_energy_records(&config.input)?;
    println!(
        "oc20_ingest: loaded {} adsorption-energy records from {}",
        energy_records.len(),
        config.input.display()
    );
    if energy_records.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no energy records to build reactions.lut from",
        ));
    }

    // Log per-species coverage explicitly rather than let a gap pass
    // silently: OC20's `train`/`val` splits do not cover every adsorbate
    // uniformly. `*CO` in particular is held out of `train`/`val` entirely
    // by the benchmark's own design (it's one of the "unseen adsorbate"
    // out-of-domain test classes), and OC20's `test_*` splits ship with
    // `y_relaxed`/`y_init` withheld (both `None`) to prevent leaderboard
    // cheating -- so there is currently no real-energy source for CO
    // anywhere in this dataset bundle. A `--input` built from `train` will
    // therefore always show 0 CO records; that is expected, not a bug.
    let mut species_counts = [0usize; SPECIES_BITS.len()];
    let mut species_real_ea_counts = [0usize; SPECIES_BITS.len()];
    for rec in &energy_records {
        species_counts[rec.species as usize] += 1;
        if rec.real_ea_ev.is_some() {
            species_real_ea_counts[rec.species as usize] += 1;
        }
    }
    for ((name, count), real_ea_count) in SPECIES_NAMES
        .iter()
        .zip(species_counts.iter())
        .zip(species_real_ea_counts.iter())
    {
        let note = if *count == 0 {
            "  (absent from --input; reactions.lut will have no reactions for this species)"
                .to_string()
        } else if *real_ea_count > 0 {
            format!("  ({real_ea_count} with a real DFT-computed activation energy, not BEP)")
        } else {
            String::new()
        };
        println!("oc20_ingest: species {name}: {count} adsorption-energy records{note}");
    }

    let bimolecular_records = match &config.bimolecular_input {
        Some(path) => read_bimolecular_records(path)?,
        None => Vec::new(),
    };
    if let Some(path) = &config.bimolecular_input {
        println!(
            "oc20_ingest: loaded {} real bimolecular reaction records from {}",
            bimolecular_records.len(),
            path.display()
        );
    }

    // A *homoatomic* bimolecular record (species_a == species_b, e.g.
    // 2 H* -> H2 + 2*) is a genuine two-site measurement of the same
    // physical process the monomolecular desorption reaction below
    // already approximates as a single-site event (using half the
    // dissociative-adsorption energy per atom -- see SPECIES_PATTERNS in
    // extract_catalysis_hub.py). Building both would give that species
    // two independent rate channels for the same real-world event, which
    // isn't "more detail," just double-counted propensity split across
    // two models of one thing. So wherever a homoatomic bimolecular
    // record exists for a species, it *replaces* that species'
    // monomolecular desorption reaction below rather than supplementing
    // it; adsorption is untouched, since these records say nothing about
    // the adsorption direction.
    let mut replaces_desorption = [false; SPECIES_BITS.len()];
    for rec in &bimolecular_records {
        if rec.species_a == rec.species_b {
            replaces_desorption[rec.species_a as usize] = true;
        }
    }

    // Each input sample normally yields TWO reactions on the lattice:
    // adsorption (VACANT -> ADS_X, forward reaction energy = the relaxed
    // adsorption energy itself) and desorption (ADS_X -> VACANT, the
    // reverse). Using Ea_rev = Ea_fwd - dE_rxn keeps the pair
    // thermodynamically consistent (same forward/reverse ratio a real
    // free-energy landscape would give) regardless of whether Ea_fwd came
    // from a real barrier or BEP. The desorption half is skipped for a
    // species covered by `replaces_desorption` -- see above.
    //
    // `(rate, transition_a, transition_b, is_bimolecular)` -- transition_b
    // is only meaningful (and non-zero) for the bimolecular records
    // appended below.
    let mut raw_rates: Vec<(f64, u8, u8, bool)> =
        Vec::with_capacity(energy_records.len() * 2 + bimolecular_records.len());
    let mut desorption_count = 0usize;
    for rec in &energy_records {
        let species_bit = SPECIES_BITS[rec.species as usize];

        let ea_fwd = activation_energy_ev(rec.energy_ev, rec.real_ea_ev, config);
        let k_ads = rate_from_activation(ea_fwd, config);
        raw_rates.push((k_ads, species_bit, 0, false)); // transition = 0x0_species (adsorption)

        if !replaces_desorption[rec.species as usize] {
            let ea_rev = (ea_fwd - rec.energy_ev).max(0.0);
            let k_des = rate_from_activation(ea_rev, config);
            raw_rates.push((k_des, species_bit << 4, 0, false)); // transition = species_0x0 (desorption)
            desorption_count += 1;
        }
    }
    for (name, replaced) in SPECIES_NAMES.iter().zip(replaces_desorption.iter()) {
        if *replaced {
            println!(
                "oc20_ingest: species {name}: monomolecular desorption replaced by real \
                 bimolecular recombination records (see --bimolecular-input)"
            );
        }
    }

    // Bimolecular records (e.g. CO oxidation, O* + CO* -> CO2 + 2*; H2
    // recombination, 2 H* -> H2 + 2*) carry a real DFT-computed forward
    // barrier only -- no BEP fallback exists for a two-species step, and
    // no reverse reaction is built: the gas product leaving the surface
    // isn't a single elementary step back onto two sites, so there's no
    // thermodynamically meaningful Ea_rev to derive here (unlike the
    // monomolecular adsorption/desorption pair above).
    for rec in &bimolecular_records {
        let bit_a = SPECIES_BITS[rec.species_a as usize];
        let bit_b = SPECIES_BITS[rec.species_b as usize];
        let k = rate_from_activation(rec.ea_ev, config);
        // transition_a/b = species_0x0 (each site: occupied -> vacant).
        raw_rates.push((k, bit_a << 4, bit_b << 4, true));
    }

    // Rescale into the Q16.16 fixed-point domain `ReactionLutBlock` uses:
    // real Arrhenius rate constants span far more dynamic range (many
    // orders of magnitude) than a 32-bit fixed-point field can represent
    // directly. Since kMC event selection only depends on *ratios* between
    // propensities, uniformly rescaling every rate by the same factor
    // changes nothing about which reaction is likeliest to fire -- it only
    // changes the absolute wall-clock/tau units, which this synthetic
    // engine doesn't otherwise calibrate against real time anyway. The
    // scale is chosen so the single fastest reaction (mono- or bimolecular
    // alike -- both compete for the same propensity budget) lands just
    // under 2^31, leaving headroom in the u32 field and keeping bin_id
    // (log2 of this value) comfortably inside CompositionTable's 32 bins.
    let max_k = raw_rates
        .iter()
        .map(|&(k, _, _, _)| k)
        .fold(0.0_f64, f64::max);
    let scale = if max_k > 0.0 {
        (1u64 << 31) as f64 / max_k
    } else {
        1.0
    };

    let records: Vec<layout::ReactionRecord> = raw_rates
        .into_iter()
        .map(|(k, transition_a, transition_b, is_bimolecular)| {
            let rate_q16 = ((k * scale).round() as u64).clamp(1, u32::MAX as u64) as u32;
            let bin_id = (31 - rate_q16.leading_zeros()) as u8;
            layout::ReactionRecord {
                rate_q16,
                bin_id,
                transition_a,
                transition_b,
                is_bimolecular,
            }
        })
        .collect();

    println!(
        "oc20_ingest: built {} reactions ({} adsorption + {} desorption + {} bimolecular), \
         rate scale factor {:.3e}",
        records.len(),
        energy_records.len(),
        desorption_count,
        bimolecular_records.len(),
        scale
    );

    let blocks: Vec<ReactionLutBlock> = layout::pack_records_into_blocks(records);
    layout::write_lut(&config.out, &blocks)?;

    println!(
        "oc20_ingest: wrote {} blocks ({} reactions) to {}",
        blocks.len(),
        blocks.len() * ReactionLutBlock::LANES,
        config.out.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "kinetica_test_oc20_ingest_{tag}_{}",
            std::process::id()
        ))
    }

    fn cfg(alpha: f64, beta_ev: f64, nu: f64, temperature_k: f64) -> Config {
        Config {
            input: PathBuf::new(),
            bimolecular_input: None,
            out: PathBuf::new(),
            alpha,
            beta_ev,
            nu,
            temperature_k,
        }
    }

    #[test]
    fn rate_constant_applies_bep_and_arrhenius() {
        // alpha=1, beta=0, and T chosen so kB*T = 1 eV -- k should reduce
        // to exp(-dE_rxn) exactly.
        let c = cfg(1.0, 0.0, 1.0, 1.0 / KB_EV_PER_K);
        let ea = activation_energy_ev(0.5, None, &c);
        let k = rate_from_activation(ea, &c);
        assert!((k - (-0.5f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn rate_constant_clamps_negative_activation_to_zero() {
        let c = cfg(1.0, 0.0, 2.0, 1.0 / KB_EV_PER_K);
        // A strongly negative reaction energy would otherwise drive the
        // BEP-estimated activation energy negative; it must clamp to 0,
        // leaving k == nu.
        let ea = activation_energy_ev(-10.0, None, &c);
        let k = rate_from_activation(ea, &c);
        assert!((k - 2.0).abs() < 1e-9);
    }

    #[test]
    fn activation_energy_uses_real_ea_when_present_bypassing_bep() {
        let c = cfg(1.0, 0.0, 1.0, 1.0 / KB_EV_PER_K);
        // BEP would give alpha*dE + beta = 5.0 here; a real Ea must
        // override that entirely, not blend with it.
        let ea = activation_energy_ev(5.0, Some(0.2), &c);
        assert!((ea - 0.2).abs() < 1e-12);
    }

    #[test]
    fn config_parse_requires_input() {
        let args = vec!["oc20_ingest".to_string()];
        let err = Config::parse(args.into_iter()).unwrap_err();
        assert!(err.contains("--input"));
    }

    #[test]
    fn config_parse_applies_defaults_and_overrides() {
        let args = ["oc20_ingest", "--input", "energies.bin", "--alpha", "0.5", "--out", "custom.lut"]
            .iter()
            .map(|s| s.to_string());
        let c = Config::parse(args).unwrap();
        assert_eq!(c.input, PathBuf::from("energies.bin"));
        assert_eq!(c.out, PathBuf::from("custom.lut"));
        assert_eq!(c.alpha, 0.5);
        assert_eq!(c.beta_ev, 0.0);
        assert_eq!(c.nu, 1.0e13);
        assert_eq!(c.temperature_k, 298.15);
    }

    #[test]
    fn config_parse_rejects_unknown_flag() {
        let args = ["oc20_ingest", "--input", "e.bin", "--bogus"]
            .iter()
            .map(|s| s.to_string());
        let err = Config::parse(args).unwrap_err();
        assert!(err.contains("--bogus"));
    }

    #[test]
    fn config_parse_accepts_bimolecular_input() {
        let args = [
            "oc20_ingest",
            "--input",
            "e.bin",
            "--bimolecular-input",
            "bi.bin",
        ]
        .iter()
        .map(|s| s.to_string());
        let c = Config::parse(args).unwrap();
        assert_eq!(c.bimolecular_input, Some(PathBuf::from("bi.bin")));
    }

    fn push_record(bytes: &mut Vec<u8>, species: u8, energy_mev: i32, sid: u32, real_ea_mev: Option<i32>) {
        bytes.push(species);
        bytes.extend_from_slice(&energy_mev.to_le_bytes());
        bytes.extend_from_slice(&sid.to_le_bytes());
        bytes.push(real_ea_mev.is_some() as u8);
        bytes.extend_from_slice(&real_ea_mev.unwrap_or(0).to_le_bytes());
    }

    #[test]
    fn read_energy_records_round_trips_and_skips_unknown_species() {
        let path = temp_path("roundtrip");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&3u32.to_le_bytes()); // record_count

        push_record(&mut bytes, 0, -123, 7, None); // species 0 = O
        push_record(&mut bytes, 9, 0, 0, None); // unknown species index -- must be skipped
        push_record(&mut bytes, 2, 456, 8, None); // species 2 = CO

        std::fs::write(&path, &bytes).unwrap();
        let records = read_energy_records(&path).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].species, 0);
        assert!((records[0].energy_ev - (-0.123)).abs() < 1e-9);
        assert!(records[0].real_ea_ev.is_none());
        assert_eq!(records[1].species, 2);
        assert!((records[1].energy_ev - 0.456).abs() < 1e-9);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_energy_records_parses_real_activation_energy_when_present() {
        let path = temp_path("real_ea");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        push_record(&mut bytes, 1, -700, 42, Some(190)); // real Ea = 0.190 eV

        std::fs::write(&path, &bytes).unwrap();
        let records = read_energy_records(&path).unwrap();

        assert_eq!(records.len(), 1);
        assert!((records[0].real_ea_ev.unwrap() - 0.190).abs() < 1e-9);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_energy_records_rejects_bad_magic() {
        let path = temp_path("bad_magic");
        std::fs::write(&path, b"NOTMAGIC\x00\x00\x00\x00").unwrap();
        assert!(read_energy_records(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    fn push_bimolecular_record(
        bytes: &mut Vec<u8>,
        species_a: u8,
        species_b: u8,
        energy_mev: i32,
        sid: u32,
        ea_mev: i32,
    ) {
        bytes.push(species_a);
        bytes.push(species_b);
        bytes.extend_from_slice(&energy_mev.to_le_bytes());
        bytes.extend_from_slice(&sid.to_le_bytes());
        bytes.extend_from_slice(&ea_mev.to_le_bytes());
    }

    #[test]
    fn read_bimolecular_records_round_trips_and_skips_unknown_species() {
        let path = temp_path("bi_roundtrip");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC_BI);
        bytes.extend_from_slice(&3u32.to_le_bytes()); // record_count

        push_bimolecular_record(&mut bytes, 0, 2, -980, 7, 980); // O + CO -> real
        push_bimolecular_record(&mut bytes, 9, 2, 0, 0, 0); // unknown species_a -- must be skipped
        push_bimolecular_record(&mut bytes, 0, 9, 0, 0, 0); // unknown species_b -- must be skipped

        std::fs::write(&path, &bytes).unwrap();
        let records = read_bimolecular_records(&path).unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].species_a, 0);
        assert_eq!(records[0].species_b, 2);
        assert_eq!(records[0].sid, 7);
        assert!((records[0].ea_ev - 0.980).abs() < 1e-9);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_bimolecular_records_rejects_bad_magic() {
        let path = temp_path("bi_bad_magic");
        std::fs::write(&path, b"NOTMAGIC\x00\x00\x00\x00").unwrap();
        assert!(read_bimolecular_records(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn run_builds_a_single_bimolecular_reaction_with_no_reverse() {
        let input_path = temp_path("run_bi_mono_input");
        let bi_path = temp_path("run_bi_bimolecular_input");
        let out_path = temp_path("run_bi_out.lut");

        // One ordinary O adsorption/desorption pair, so `energy_records`
        // isn't empty (run() rejects that up front).
        let mut mono_bytes = Vec::new();
        mono_bytes.extend_from_slice(MAGIC);
        mono_bytes.extend_from_slice(&1u32.to_le_bytes());
        push_record(&mut mono_bytes, 0, -500, 1, None);
        std::fs::write(&input_path, &mono_bytes).unwrap();

        // One real CO-oxidation barrier: O* + CO* -> CO2 + 2*, Ea = 1.0 eV.
        let mut bi_bytes = Vec::new();
        bi_bytes.extend_from_slice(MAGIC_BI);
        bi_bytes.extend_from_slice(&1u32.to_le_bytes());
        push_bimolecular_record(&mut bi_bytes, 0, 2, -980, 42, 1000);
        std::fs::write(&bi_path, &bi_bytes).unwrap();

        let config = Config {
            input: input_path.clone(),
            bimolecular_input: Some(bi_path.clone()),
            out: out_path.clone(),
            alpha: 0.87,
            beta_ev: 0.0,
            nu: 1.0e13,
            temperature_k: 298.15,
        };
        run(&config).unwrap();

        let lut = layout::ReactionLut::open(&out_path).unwrap();
        let reaction_count = lut.len() * ReactionLutBlock::LANES;
        let all_records: Vec<_> = (0..reaction_count).map(|id| lut.rate_of(id)).collect();
        // 2 monomolecular (adsorption + desorption) + 1 bimolecular; the
        // rest of the last (8-lane) block is zero-padding.
        let real_records = all_records.iter().filter(|r| r.rate_q16 > 0).count();
        assert_eq!(real_records, 3);

        let bimolecular: Vec<_> = all_records
            .into_iter()
            .filter(|r| r.is_bimolecular)
            .collect();
        assert_eq!(bimolecular.len(), 1, "exactly one bimolecular reaction, no reverse built");
        let r = bimolecular[0];
        assert_eq!(r.transition_a, layout::ADS_O << 4); // O* -> vacant
        assert_eq!(r.transition_b, layout::ADS_CO << 4); // CO* -> vacant

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&bi_path);
        let _ = std::fs::remove_file(&out_path);
    }

    #[test]
    fn run_replaces_monomolecular_desorption_only_for_the_homoatomic_species() {
        let input_path = temp_path("run_bi_homo_input");
        let bi_path = temp_path("run_bi_homo_bimolecular_input");
        let out_path = temp_path("run_bi_homo_out.lut");

        // One O record and one H record -- O should keep both its
        // adsorption and desorption reactions; H should keep only
        // adsorption once a homoatomic bimolecular record exists for it.
        let mut mono_bytes = Vec::new();
        mono_bytes.extend_from_slice(MAGIC);
        mono_bytes.extend_from_slice(&2u32.to_le_bytes());
        push_record(&mut mono_bytes, 0, -500, 1, None); // O
        push_record(&mut mono_bytes, 1, -400, 2, None); // H
        std::fs::write(&input_path, &mono_bytes).unwrap();

        // One real H2 recombination barrier: 2 H* -> H2 + 2*, Ea = 0.35 eV.
        let mut bi_bytes = Vec::new();
        bi_bytes.extend_from_slice(MAGIC_BI);
        bi_bytes.extend_from_slice(&1u32.to_le_bytes());
        push_bimolecular_record(&mut bi_bytes, 1, 1, -33, 99, 350);
        std::fs::write(&bi_path, &bi_bytes).unwrap();

        let config = Config {
            input: input_path.clone(),
            bimolecular_input: Some(bi_path.clone()),
            out: out_path.clone(),
            alpha: 0.87,
            beta_ev: 0.0,
            nu: 1.0e13,
            temperature_k: 298.15,
        };
        run(&config).unwrap();

        let lut = layout::ReactionLut::open(&out_path).unwrap();
        let reaction_count = lut.len() * ReactionLutBlock::LANES;
        let real_records: Vec<_> = (0..reaction_count)
            .map(|id| lut.rate_of(id))
            .filter(|r| r.rate_q16 > 0)
            .collect();
        // O adsorption + O desorption + H adsorption + 1 bimolecular
        // (H desorption is replaced, not built).
        assert_eq!(real_records.len(), 4);

        let o_desorption = real_records
            .iter()
            .filter(|r| !r.is_bimolecular && r.transition_a == layout::ADS_O << 4)
            .count();
        assert_eq!(o_desorption, 1, "O's monomolecular desorption must be untouched");

        let h_desorption = real_records
            .iter()
            .filter(|r| !r.is_bimolecular && r.transition_a == layout::ADS_H << 4)
            .count();
        assert_eq!(
            h_desorption, 0,
            "H's monomolecular desorption must be replaced by the bimolecular record"
        );

        let h_adsorption = real_records
            .iter()
            .filter(|r| !r.is_bimolecular && r.transition_a == layout::ADS_H)
            .count();
        assert_eq!(h_adsorption, 1, "H's adsorption reaction is unaffected");

        let bimolecular: Vec<_> = real_records.into_iter().filter(|r| r.is_bimolecular).collect();
        assert_eq!(bimolecular.len(), 1);
        assert_eq!(bimolecular[0].transition_a, layout::ADS_H << 4);
        assert_eq!(bimolecular[0].transition_b, layout::ADS_H << 4);

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&bi_path);
        let _ = std::fs::remove_file(&out_path);
    }
}
