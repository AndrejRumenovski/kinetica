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
//!
//! **Output is a small, quantile-bucketed catalogue, not one record per
//! DFT sample.** A `--input` file typically carries hundreds of adsorption-
//! energy records per species (one per real DFT calculation this species
//! happened to appear in, across many different surfaces). Earlier
//! versions of this tool turned every single one into its own independent
//! `ReactionRecord` -- which meant propensity had no way to reflect how
//! much of a species the lattice actually had on it, since a static list
//! of "reactions" has no notion of the live surface state at all (see
//! `kinetica::occupancy` for the fix on the engine side). This tool's half
//! of that fix is `bucket_by_quantile`: each species' samples are sorted
//! by reaction energy and split into `BUCKETS_PER_SPECIES` (4) roughly
//! equal groups, and each group collapses to *one* representative
//! adsorption (and, usually, desorption) `ReactionRecord` built from the
//! group's mean energy (mean real Ea too, when any group member has one).
//! This keeps real, meaningful heterogeneity -- a genuinely fast-reacting
//! quartile of surfaces versus a genuinely slow-reacting one -- without
//! either collapsing to a single averaged number per species or keeping
//! hundreds of individually-untracked channels the engine has no way to
//! gate on actual occupancy.

use std::io;
use std::path::PathBuf;

use kinetica::config::{SimConfig, SpeciesEntry, SpeciesRole};
use kinetica::layout::{self, ReactionLutBlock, MAX_SPECIES};
use kinetica::oc20e_format::{
    read_bimolecular_records, read_energy_records, BiEnergyRecord, EnergyRecord,
};
use kinetica::occupancy::BUCKETS_PER_SPECIES;

/// Metal index table, in lockstep with `scripts/oc20e_format.py`'s own
/// `METALS` list -- the numeric index is what's on disk in both
/// `OC20E003`/`OC20BI02` records, not the string, so the two lists must
/// stay identical. Index 0 ("unknown") means "not one of the metals this
/// pipeline tracks," not "absent" -- every real record still carries some
/// index, even if it's 0.
const METALS: [&str; 13] = [
    "unknown", "Pd", "Pt", "Cu", "Ni", "Rh", "Ru", "Ag", "Au", "Fe", "Co", "Ir", "Os",
];

fn metal_index(symbol: &str) -> Option<u8> {
    METALS.iter().position(|&m| m == symbol).map(|i| i as u8)
}

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
    /// Restrict ingestion to this metal (index into `METALS`), with a
    /// per-species fallback to "this metal, any facet" when the
    /// `--facet`-filtered pool is too sparse to bucket meaningfully --
    /// see `run`'s `filter_with_fallback`.
    metal: Option<u8>,
    facet: Option<u16>,
    /// The active species set (identity, gas role) for this run, in
    /// `--config`'s `[species]` declaration order -- that order is each
    /// species' index into `EnergyRecord`/`BiEnergyRecord`'s on-disk
    /// `species`/`species_a`/`species_b` bytes, replacing what used to be
    /// the compile-time `SPECIES_NAMES`/`SPECIES_BITS`/
    /// `DISSOCIATIVE_SPECIES` constants.
    species: Vec<SpeciesEntry>,
}

impl Config {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut args = args;
        let _bin = args.next();

        let mut input = None;
        let mut bimolecular_input = None;
        let mut out = PathBuf::from("reactions.lut");
        let mut config_path = None;
        let mut alpha_override = None;
        let mut beta_override = None;
        let mut nu_override = None;
        let mut temperature_override = None;
        let mut metal_override = None;
        let mut facet_override = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => input = Some(PathBuf::from(next_value(&mut args, "--input")?)),
                "--bimolecular-input" => {
                    bimolecular_input =
                        Some(PathBuf::from(next_value(&mut args, "--bimolecular-input")?))
                }
                "--out" => out = PathBuf::from(next_value(&mut args, "--out")?),
                "--config" => config_path = Some(PathBuf::from(next_value(&mut args, "--config")?)),
                "--alpha" => alpha_override = Some(parse_value(&mut args, "--alpha")?),
                "--beta" => beta_override = Some(parse_value(&mut args, "--beta")?),
                "--nu" => nu_override = Some(parse_value(&mut args, "--nu")?),
                "--temperature" => {
                    temperature_override = Some(parse_value(&mut args, "--temperature")?)
                }
                "--metal" => metal_override = Some(next_value(&mut args, "--metal")?),
                "--facet" => facet_override = Some(parse_value(&mut args, "--facet")?),
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
            }
        }

        let config_path =
            config_path.ok_or_else(|| format!("`--config` is required\n\n{}", usage()))?;
        let config_text = std::fs::read_to_string(&config_path).map_err(|e| {
            format!(
                "failed to read --config file {}: {e}",
                config_path.display()
            )
        })?;
        let sim_config = SimConfig::parse(&config_text)
            .map_err(|e| format!("error in --config file {}: {e}", config_path.display()))?;

        let resolve_metal = |symbol: &str| -> Result<u8, String> {
            metal_index(symbol).ok_or_else(|| {
                format!(
                    "metal `{symbol}` isn't tracked; known metals: {}",
                    METALS[1..].join(", ")
                )
            })
        };
        let metal = match metal_override {
            Some(symbol) => Some(resolve_metal(&symbol)?),
            None => sim_config.metal.as_deref().map(resolve_metal).transpose()?,
        };
        let facet = facet_override.or(sim_config.facet);

        Ok(Self {
            input: input.ok_or_else(|| format!("`--input` is required\n\n{}", usage()))?,
            bimolecular_input,
            out,
            alpha: alpha_override.unwrap_or(sim_config.alpha),
            beta_ev: beta_override.unwrap_or(sim_config.beta_ev),
            nu: nu_override.unwrap_or(sim_config.nu),
            temperature_k: temperature_override.unwrap_or(sim_config.temperature_k),
            metal,
            facet,
            species: sim_config.species,
        })
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("`{flag}` requires a value"))
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
       oc20_ingest --input <PATH> --config <PATH> [OPTIONS]\n\n\
     OPTIONS:\n    \
       --input <PATH>              Flat binary from an extract_*.py script (required)\n    \
       --config <PATH>             Sectioned text config declaring the metal/facet/\n                                    \
                                    BEP defaults/species set to build from (required;\n                                    \
                                    see configs/pd111.conf for an example, and\n                                    \
                                    kinetica::config's own doc comment for the format)\n    \
       --bimolecular-input <PATH>  Optional OC20BI01 binary (from\n                                    \
                                    extract_catalysis_hub.py's --bimolecular-out)\n                                    \
                                    carrying real two-site reaction barriers,\n                                    \
                                    e.g. CO oxidation\n    \
       --out <PATH>                Output reactions.lut [default: reactions.lut]\n    \
       --alpha <F>                 BEP relation slope [default: --config's [bep] alpha]\n    \
       --beta <F>                  BEP relation intercept, eV [default: --config's [bep] beta]\n    \
       --nu <F>                    Arrhenius prefactor, s^-1 [default: --config's [bep] nu]\n    \
       --temperature <F>           Temperature, K [default: --config's [bep] temperature]\n    \
       --metal <SYMBOL>            Restrict to this metal (e.g. Pd), overriding\n                                    \
                                    --config's [system] metal; per-species fallback\n                                    \
                                    to metal/any-facet if --facet leaves too few\n                                    \
                                    samples to bucket [default: --config's [system] metal]\n    \
       --facet <N>                 Restrict to this Miller-index facet (e.g. 111),\n                                    \
                                    overriding --config's [system] facet\n                                    \
                                    [default: --config's [system] facet]\n    \
       -h, --help                  Print this message"
        .to_string()
}

/// One quantile bucket's summary: how many real DFT samples fell into it,
/// their mean reaction energy (what BEP is applied to when nothing better
/// is available), and the mean of just the members that carried a real
/// activation energy, if any did.
#[derive(Debug, Clone, Copy, PartialEq)]
struct BucketSummary {
    mean_energy_ev: f64,
    real_ea_ev: Option<f64>,
    sample_count: usize,
}

/// Split `records` into up to `num_buckets` roughly-equal-sized groups by
/// sorted reaction energy, and summarize each into one `BucketSummary`.
/// Bucketing by reaction energy (the direct DFT observable) rather than by
/// derived activation energy keeps the split meaningful even for the
/// majority of records that don't have a real Ea to sort by. Never
/// creates more buckets than there are records -- with `N < num_buckets`
/// samples this gracefully degrades to one bucket per sample, i.e. the
/// same per-record behavior this function replaces.
fn bucket_by_quantile(records: &[EnergyRecord], num_buckets: usize) -> Vec<BucketSummary> {
    if records.is_empty() {
        return Vec::new();
    }

    let mut sorted = records.to_vec();
    sorted.sort_by(|a, b| a.energy_ev.total_cmp(&b.energy_ev));

    let n = sorted.len();
    let buckets = num_buckets.clamp(1, n);
    let mut summaries = Vec::with_capacity(buckets);

    for b in 0..buckets {
        let start = b * n / buckets;
        let end = (b + 1) * n / buckets;
        let group = &sorted[start..end];

        let mean_energy_ev = group.iter().map(|r| r.energy_ev).sum::<f64>() / group.len() as f64;
        let real_eas: Vec<f64> = group.iter().filter_map(|r| r.real_ea_ev).collect();
        let real_ea_ev = if real_eas.is_empty() {
            None
        } else {
            Some(real_eas.iter().sum::<f64>() / real_eas.len() as f64)
        };

        summaries.push(BucketSummary {
            mean_energy_ev,
            real_ea_ev,
            sample_count: group.len(),
        });
    }

    summaries
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

/// Apply `config`'s `--metal`/`--facet` filters to `records`, grouped per
/// species (`config.species`'s declaration order). When both are set and a given
/// species' `(metal, facet)`-filtered pool has fewer than
/// `BUCKETS_PER_SPECIES` samples, that species alone falls back to
/// `metal`-only (any facet) -- logged explicitly so a reader can tell a
/// deliberate, honest broadening from a silent pooling regression. A
/// species left with zero samples even after falling back stays empty,
/// same as today's "absent from --input" case.
fn filter_with_fallback(
    records: &[EnergyRecord],
    config: &Config,
) -> [Vec<EnergyRecord>; MAX_SPECIES] {
    let mut by_species: [Vec<EnergyRecord>; MAX_SPECIES] = Default::default();

    let Some(metal) = config.metal else {
        for &rec in records {
            by_species[rec.species as usize].push(rec);
        }
        return by_species;
    };

    let metal_only: Vec<EnergyRecord> = records
        .iter()
        .copied()
        .filter(|r| r.metal == metal)
        .collect();
    let metal_name = METALS[metal as usize];

    for (species, slot) in by_species.iter_mut().enumerate().take(config.species.len()) {
        let metal_only_species: Vec<EnergyRecord> = metal_only
            .iter()
            .copied()
            .filter(|r| r.species as usize == species)
            .collect();

        let Some(facet) = config.facet else {
            *slot = metal_only_species;
            continue;
        };

        let filtered: Vec<EnergyRecord> = metal_only_species
            .iter()
            .copied()
            .filter(|r| r.facet == facet)
            .collect();

        if filtered.len() < BUCKETS_PER_SPECIES && metal_only_species.len() > filtered.len() {
            println!(
                "oc20_ingest: species {}: only {} record(s) match --metal {metal_name} \
                 --facet {facet}; broadening to --metal {metal_name} (any facet) -> \
                 {} record(s)",
                config.species[species].name,
                filtered.len(),
                metal_only_species.len()
            );
            *slot = metal_only_species;
        } else {
            *slot = filtered;
        }
    }

    by_species
}

/// Same `--metal`/`--facet` filtering for the (much smaller) bimolecular
/// record set. Bimolecular real barriers are rare enough that there's no
/// meaningful quantile-bucket threshold to fall back against -- the
/// fallback here triggers whenever the facet-filtered pool is empty but
/// the metal-only one isn't.
fn filter_bimolecular_with_fallback(
    records: &[BiEnergyRecord],
    config: &Config,
) -> Vec<BiEnergyRecord> {
    let Some(metal) = config.metal else {
        return records.to_vec();
    };
    let metal_only: Vec<BiEnergyRecord> = records
        .iter()
        .copied()
        .filter(|r| r.metal == metal)
        .collect();
    let metal_name = METALS[metal as usize];

    let Some(facet) = config.facet else {
        return metal_only;
    };

    let filtered: Vec<BiEnergyRecord> = metal_only
        .iter()
        .copied()
        .filter(|r| r.facet == facet)
        .collect();
    if filtered.is_empty() && !metal_only.is_empty() {
        println!(
            "oc20_ingest: bimolecular: no record matches --metal {metal_name} --facet {facet}; \
             broadening to --metal {metal_name} (any facet) -> {} record(s)",
            metal_only.len()
        );
        metal_only
    } else {
        filtered
    }
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

    if let Some(metal) = config.metal {
        let facet_desc = config
            .facet
            .map(|f| f.to_string())
            .unwrap_or_else(|| "any".to_string());
        println!(
            "oc20_ingest: filtering to metal={} facet={facet_desc} (per-species fallback to \
             metal-only if --facet leaves too few samples to bucket)",
            METALS[metal as usize]
        );
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
    let by_species = filter_with_fallback(&energy_records, config);

    let mut bucketed_by_species: [Vec<BucketSummary>; MAX_SPECIES] = Default::default();
    for species in 0..config.species.len() {
        let count = by_species[species].len();
        let real_ea_count = by_species[species]
            .iter()
            .filter(|r| r.real_ea_ev.is_some())
            .count();
        let note = if count == 0 {
            "  (absent from --input; reactions.lut will have no reactions for this species)"
                .to_string()
        } else if real_ea_count > 0 {
            format!("  ({real_ea_count} with a real DFT-computed activation energy, not BEP)")
        } else {
            String::new()
        };
        println!(
            "oc20_ingest: species {}: {count} adsorption-energy records{note}",
            config.species[species].name
        );

        let buckets = bucket_by_quantile(&by_species[species], BUCKETS_PER_SPECIES);
        if !buckets.is_empty() {
            let sizes: Vec<String> = buckets.iter().map(|b| b.sample_count.to_string()).collect();
            println!(
                "oc20_ingest: species {}: collapsed into {} quantile bucket(s) (sizes: {})",
                config.species[species].name,
                buckets.len(),
                sizes.join(", ")
            );
        }
        bucketed_by_species[species] = buckets;
    }

    let bimolecular_records = match &config.bimolecular_input {
        Some(path) => filter_bimolecular_with_fallback(&read_bimolecular_records(path)?, config),
        None => Vec::new(),
    };
    if let Some(path) = &config.bimolecular_input {
        println!(
            "oc20_ingest: loaded {} real bimolecular reaction records from {} (after --metal/--facet filtering)",
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
    let mut replaces_desorption = [false; MAX_SPECIES];
    for rec in &bimolecular_records {
        if !rec.is_dissociative && rec.species_a == rec.species_b {
            replaces_desorption[rec.species_a as usize] = true;
        }
    }

    // Each quantile bucket normally yields TWO reactions on the lattice:
    // adsorption and desorption (the reverse), thermodynamically
    // consistent via Ea_rev = Ea_fwd - dE_rxn regardless of whether Ea_fwd
    // came from a real barrier or BEP. The desorption half is skipped for
    // a species covered by `replaces_desorption` -- see above.
    //
    // For O and H (`DISSOCIATIVE_SPECIES`), the adsorption half is built
    // as a genuine two-site *dissociative* event (`2* + O2(g)/H2(g) ->
    // 2 species*`, `is_bimolecular = true`, both sites VACANT -> species)
    // instead of the pseudo-monomolecular approximation (one site,
    // `VACANT -> species`) used previously -- the underlying energy is
    // exactly the same per-atom DFT value this pipeline already extracts
    // (see `SPECIES_PATTERNS`'s 0.5 stoichiometry), the fix is applying it
    // to a correctly-gated *pair* of sites (drawing from
    // `occupancy::OccupancyCounters::vacant_pairs`) rather than two
    // independent single sites with no coverage-blocking relationship to
    // each other. CO adsorbs molecularly (one site) and keeps the
    // original monomolecular form unchanged. Desorption is untouched
    // either way -- this only corrects the adsorption direction.
    //
    // `(rate, transition_a, transition_b, is_bimolecular, bucket_id)` --
    // transition_b is only meaningful (and non-zero) for a bimolecular
    // record; bucket_id becomes each `ReactionRecord`'s `bin_id` (unused/0
    // for bimolecular templates -- see `occupancy.rs`).
    let mut raw_rates: Vec<(f64, u16, u16, bool, u8)> = Vec::new();
    let mut ads_count = 0usize;
    let mut dissociative_ads_count = 0usize;
    let mut des_count = 0usize;
    for species in 0..config.species.len() {
        // A `product_only` species (e.g. OH, which only ever forms via
        // water splitting) has no single-gas source of its own -- it must
        // never get a monomolecular adsorption/desorption template built
        // for it, regardless of what (unexpectedly) shows up in
        // `--input`. Today this is a no-op against real data (no
        // extraction script ever emits an OH monomolecular record), but
        // it makes the "no direct gas source" contract explicit rather
        // than relying on data happening to be absent.
        if config.species[species].role == SpeciesRole::ProductOnly {
            continue;
        }
        let species_bit = config.species[species].bit;
        let dissociative = config.species[species].role == SpeciesRole::Dissociative;
        for (bucket_idx, bucket) in bucketed_by_species[species].iter().enumerate() {
            let ea_fwd = activation_energy_ev(bucket.mean_energy_ev, bucket.real_ea_ev, config);
            let k_ads = rate_from_activation(ea_fwd, config);
            if dissociative {
                // transition_a = transition_b = 0x00_species (both sites:
                // vacant -> species), un-bucketed like other bimolecular
                // records -- see `occupancy::OccupancyCounters::live_count`.
                raw_rates.push((k_ads, species_bit as u16, species_bit as u16, true, 0));
                dissociative_ads_count += 1;
            } else {
                raw_rates.push((k_ads, species_bit as u16, 0, false, bucket_idx as u8)); // 0x00_species (adsorption)
                ads_count += 1;
            }

            if !replaces_desorption[species] {
                let ea_rev = (ea_fwd - bucket.mean_energy_ev).max(0.0);
                let k_des = rate_from_activation(ea_rev, config);
                raw_rates.push((k_des, (species_bit as u16) << 8, 0, false, bucket_idx as u8)); // species_0x00 (desorption)
                des_count += 1;
            }
        }
    }
    for (entry, replaced) in config.species.iter().zip(replaces_desorption.iter()) {
        if *replaced {
            println!(
                "oc20_ingest: species {}: monomolecular desorption replaced by real \
                 bimolecular recombination records (see --bimolecular-input)",
                entry.name
            );
        }
    }

    // Bimolecular records carry a real DFT-computed forward barrier only --
    // no BEP fallback exists for a two-species step. What gets built from
    // each one depends on `is_dissociative` (see `BiEnergyRecord`'s doc
    // comment):
    //
    // - Recombination (e.g. CO oxidation, O* + CO* -> CO2 + 2*; H2
    //   recombination, 2 H* -> H2 + 2*): forward-only, no reverse -- the
    //   gas product leaving the surface isn't a single elementary step
    //   back onto two sites, so there's no thermodynamically meaningful
    //   Ea_rev to derive here (unlike the monomolecular adsorption/
    //   desorption pair above).
    // - Dissociative adsorption (currently just water splitting,
    //   2* + H2O(g) -> H* + OH*): built *both* directions, same
    //   Ea_rev = Ea_fwd - dE_rxn thermodynamic-consistency relation the
    //   monomolecular pair above uses -- this direction's reverse
    //   (associative desorption) genuinely is the same elementary step
    //   run backward, unlike recombination's gas product.
    //
    // Kept un-bucketed (bucket_id = 0, unused by the engine for
    // bimolecular templates -- see `occupancy::OccupancyCounters::
    // live_count`): there are only a handful of these real barriers, not
    // enough to meaningfully quantile-split.
    // Both counted directly in the loop below, not derived by subtracting
    // one from `bimolecular_records.len()` -- a record whose species index
    // this `--config` doesn't declare hits the `continue` a few lines down
    // and contributes to neither count, so `bimolecular_records.len() -
    // dissociative_bimolecular_count` over-counts "recombination" by
    // exactly the number of skipped records (a real bug this project's
    // Phase 7 generality config, `configs/pd111_ohco_subset.conf`, caught:
    // the frozen Pd(111) fixture's 2 real bimolecular records both name OH,
    // which that config doesn't declare, and the old subtraction-based
    // count silently mislabeled both as "recombination" instead of
    // reflecting that they built zero reactions).
    let mut dissociative_bimolecular_count = 0usize;
    let mut recombination_bimolecular_count = 0usize;
    for rec in &bimolecular_records {
        // `oc20e_format::read_bimolecular_records` only bounds-checks
        // against the architectural `MAX_SPECIES` ceiling, not this run's
        // *active* species count (it has no way to know that) -- a record
        // naming a species index this `--config` didn't declare is
        // ignored here the same way an unknown species already is
        // upstream, rather than indexing `config.species` out of bounds.
        let (Some(entry_a), Some(entry_b)) = (
            config.species.get(rec.species_a as usize),
            config.species.get(rec.species_b as usize),
        ) else {
            continue;
        };
        let bit_a = entry_a.bit;
        let bit_b = entry_b.bit;
        let k_fwd = rate_from_activation(rec.ea_ev, config);
        if rec.is_dissociative {
            // Forward: 2* -> species_a* + species_b* (both sites: vacant
            // -> species).
            raw_rates.push((k_fwd, bit_a as u16, bit_b as u16, true, 0));
            // Reverse: species_a* + species_b* -> 2* (both sites: species
            // -> vacant), associative desorption.
            let ea_rev = (rec.ea_ev - rec.energy_ev).max(0.0);
            let k_rev = rate_from_activation(ea_rev, config);
            raw_rates.push((k_rev, (bit_a as u16) << 8, (bit_b as u16) << 8, true, 0));
            dissociative_bimolecular_count += 1;
        } else {
            // transition_a/b = species_0x00 (each site: occupied -> vacant).
            raw_rates.push((k_fwd, (bit_a as u16) << 8, (bit_b as u16) << 8, true, 0));
            recombination_bimolecular_count += 1;
        }
    }

    // Rescale into the Q16.16 fixed-point domain `ReactionLutBlock` uses:
    // real Arrhenius rate constants span far more dynamic range (many
    // orders of magnitude) than a 32-bit fixed-point field can represent
    // directly. Since kMC event selection only depends on *ratios* between
    // propensities, uniformly rescaling every rate by the same factor
    // changes nothing about which reaction is likeliest to fire -- it only
    // changes the absolute wall-clock/tau units, which this engine doesn't
    // otherwise calibrate against real time anyway. The scale is chosen so
    // the single fastest reaction (mono- or bimolecular alike -- both
    // compete for the same propensity budget) lands just under 2^31,
    // leaving headroom in the u32 field.
    let max_k = raw_rates
        .iter()
        .map(|&(k, _, _, _, _)| k)
        .fold(0.0_f64, f64::max);
    let scale = if max_k > 0.0 {
        (1u64 << 31) as f64 / max_k
    } else {
        1.0
    };

    let records: Vec<layout::ReactionRecord> = raw_rates
        .into_iter()
        .map(
            |(k, transition_a, transition_b, is_bimolecular, bucket_id)| {
                let rate_q16 = ((k * scale).round() as u64).clamp(1, u32::MAX as u64) as u32;
                layout::ReactionRecord {
                    rate_q16,
                    // `bin_id`: the quantile bucket index for a monomolecular
                    // template (0 and unused for a bimolecular one) -- *not*
                    // a composition-rejection magnitude class here, unlike
                    // `LutKind::Static` LUTs. See `occupancy.rs`.
                    bin_id: bucket_id,
                    transition_a,
                    transition_b,
                    is_bimolecular,
                }
            },
        )
        .collect();

    println!(
        "oc20_ingest: built {} reactions ({} monomolecular adsorption + {} desorption + \
         {} homoatomic dissociative-adsorption bimolecular + {} heteroatomic dissociative \
         bimolecular pair(s) (forward+reverse) + {} recombination bimolecular), \
         rate scale factor {:.3e}",
        records.len(),
        ads_count,
        des_count,
        dissociative_ads_count,
        dissociative_bimolecular_count,
        recombination_bimolecular_count,
        scale
    );

    let blocks: Vec<ReactionLutBlock> = layout::pack_records_into_blocks(records);
    let species_table = layout::SpeciesTable::new(
        config
            .species
            .iter()
            .map(|entry| (entry.bit, entry.name.clone()))
            .collect(),
    )
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    layout::write_lut_with_species(
        &config.out,
        layout::LutKind::OccupancyGated,
        &blocks,
        &species_table,
    )?;

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
    use kinetica::oc20e_format::{MAGIC, MAGIC_BI};

    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "kinetica_test_oc20_ingest_{tag}_{}",
            std::process::id()
        ))
    }

    /// The pd111.conf-equivalent 5-species set (O, H, CO, OH, H2O), same
    /// bits/roles the compile-time `SPECIES_NAMES`/`SPECIES_BITS`/
    /// `DISSOCIATIVE_SPECIES` constants used to hardcode -- shared by
    /// every test below so each doesn't have to spell it out.
    fn default_species() -> Vec<SpeciesEntry> {
        vec![
            SpeciesEntry {
                name: "O".to_string(),
                bit: layout::ADS_O,
                gas: None,
                stoich: None,
                product: None,
                role: SpeciesRole::Dissociative,
                oc20_ads_id: None,
            },
            SpeciesEntry {
                name: "H".to_string(),
                bit: layout::ADS_H,
                gas: None,
                stoich: None,
                product: None,
                role: SpeciesRole::Dissociative,
                oc20_ads_id: None,
            },
            SpeciesEntry {
                name: "CO".to_string(),
                bit: layout::ADS_CO,
                gas: None,
                stoich: None,
                product: None,
                role: SpeciesRole::Molecular,
                oc20_ads_id: None,
            },
            SpeciesEntry {
                name: "OH".to_string(),
                bit: layout::ADS_OH,
                gas: None,
                stoich: None,
                product: None,
                role: SpeciesRole::ProductOnly,
                oc20_ads_id: None,
            },
            SpeciesEntry {
                name: "H2O".to_string(),
                bit: layout::ADS_H2O,
                gas: None,
                stoich: None,
                product: None,
                role: SpeciesRole::Molecular,
                oc20_ads_id: None,
            },
        ]
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
            metal: None,
            facet: None,
            species: default_species(),
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

    /// A minimal but valid `--config` file for CLI-parsing tests that
    /// don't care about species specifics -- just enough for
    /// `SimConfig::parse` to succeed so `Config::parse` reaches the
    /// CLI-flag logic these tests actually exercise. `tag` must be unique
    /// per call site: `temp_path` (this file's own, not the library's
    /// counter-suffixed one) derives its path from the tag and this
    /// process's pid alone, so two tests sharing a tag can race on the
    /// same file when `cargo test` runs them in parallel threads.
    fn write_minimal_config(tag: &str) -> PathBuf {
        let path = temp_path(tag);
        std::fs::write(
            &path,
            "[species]\nO = 0x01, O2gas, 0.5, Ostar, dissociative, 0\n",
        )
        .unwrap();
        path
    }

    #[test]
    fn config_parse_applies_defaults_and_overrides() {
        let config_path = write_minimal_config("defaults_and_overrides");
        let args = [
            "oc20_ingest".to_string(),
            "--input".to_string(),
            "energies.bin".to_string(),
            "--config".to_string(),
            config_path.display().to_string(),
            "--alpha".to_string(),
            "0.5".to_string(),
            "--out".to_string(),
            "custom.lut".to_string(),
        ]
        .into_iter();
        let c = Config::parse(args).unwrap();
        assert_eq!(c.input, PathBuf::from("energies.bin"));
        assert_eq!(c.out, PathBuf::from("custom.lut"));
        assert_eq!(c.alpha, 0.5);
        assert_eq!(c.beta_ev, 0.0);
        assert_eq!(c.nu, 1.0e13);
        assert_eq!(c.temperature_k, 298.15);
        let _ = std::fs::remove_file(&config_path);
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
        let config_path = write_minimal_config("accepts_bimolecular_input");
        let args = [
            "oc20_ingest".to_string(),
            "--input".to_string(),
            "e.bin".to_string(),
            "--config".to_string(),
            config_path.display().to_string(),
            "--bimolecular-input".to_string(),
            "bi.bin".to_string(),
        ]
        .into_iter();
        let c = Config::parse(args).unwrap();
        assert_eq!(c.bimolecular_input, Some(PathBuf::from("bi.bin")));
        let _ = std::fs::remove_file(&config_path);
    }

    #[test]
    fn config_parse_accepts_metal_and_facet() {
        let config_path = write_minimal_config("accepts_metal_and_facet");
        let args = [
            "oc20_ingest".to_string(),
            "--input".to_string(),
            "e.bin".to_string(),
            "--config".to_string(),
            config_path.display().to_string(),
            "--metal".to_string(),
            "Pd".to_string(),
            "--facet".to_string(),
            "111".to_string(),
        ]
        .into_iter();
        let c = Config::parse(args).unwrap();
        assert_eq!(c.metal, metal_index("Pd"));
        assert_eq!(c.facet, Some(111));
        let _ = std::fs::remove_file(&config_path);
    }

    #[test]
    fn config_parse_rejects_unknown_metal() {
        let config_path = write_minimal_config("rejects_unknown_metal");
        let args = [
            "oc20_ingest".to_string(),
            "--input".to_string(),
            "e.bin".to_string(),
            "--config".to_string(),
            config_path.display().to_string(),
            "--metal".to_string(),
            "Unobtainium".to_string(),
        ]
        .into_iter();
        let err = Config::parse(args).unwrap_err();
        assert!(err.contains("Unobtainium"));
        let _ = std::fs::remove_file(&config_path);
    }

    #[test]
    fn config_parse_requires_config() {
        let args = ["oc20_ingest", "--input", "e.bin"]
            .iter()
            .map(|s| s.to_string());
        let err = Config::parse(args).unwrap_err();
        assert!(err.contains("--config"));
    }

    #[test]
    fn config_parse_resolves_metal_and_facet_from_config_file() {
        let path = temp_path("config_with_system");
        std::fs::write(
            &path,
            "[system]\nmetal = Pd\nfacet = 111\n\n[species]\nO = 0x01, O2gas, 0.5, Ostar, dissociative, 0\n",
        )
        .unwrap();
        let args = [
            "oc20_ingest".to_string(),
            "--input".to_string(),
            "e.bin".to_string(),
            "--config".to_string(),
            path.display().to_string(),
        ]
        .into_iter();
        let c = Config::parse(args).unwrap();
        assert_eq!(c.metal, metal_index("Pd"));
        assert_eq!(c.facet, Some(111));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn config_parse_cli_metal_overrides_config_file_metal() {
        let path = temp_path("config_with_system_override");
        std::fs::write(
            &path,
            "[system]\nmetal = Pd\nfacet = 111\n\n[species]\nO = 0x01, O2gas, 0.5, Ostar, dissociative, 0\n",
        )
        .unwrap();
        let args = [
            "oc20_ingest".to_string(),
            "--input".to_string(),
            "e.bin".to_string(),
            "--config".to_string(),
            path.display().to_string(),
            "--metal".to_string(),
            "Pt".to_string(),
        ]
        .into_iter();
        let c = Config::parse(args).unwrap();
        assert_eq!(
            c.metal,
            metal_index("Pt"),
            "--metal must override the config file's [system] metal"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn config_parse_loads_species_from_config_file() {
        let path = temp_path("config_with_species");
        std::fs::write(
            &path,
            "[species]\n\
             O = 0x01, O2gas, 0.5, Ostar, dissociative, 0\n\
             CO = 0x02, COgas, 1.0, COstar, molecular, 5\n",
        )
        .unwrap();
        let args = [
            "oc20_ingest".to_string(),
            "--input".to_string(),
            "e.bin".to_string(),
            "--config".to_string(),
            path.display().to_string(),
        ]
        .into_iter();
        let c = Config::parse(args).unwrap();
        assert_eq!(c.species.len(), 2);
        assert_eq!(c.species[0].name, "O");
        assert_eq!(c.species[1].name, "CO");
        let _ = std::fs::remove_file(&path);
    }

    fn energy_record_with_metal(species: u8, metal: u8, facet: u16) -> EnergyRecord {
        EnergyRecord {
            species,
            energy_ev: -0.5,
            sid: 0,
            real_ea_ev: None,
            metal,
            facet,
        }
    }

    #[test]
    fn filter_with_fallback_passes_through_everything_when_no_metal_set() {
        let records = vec![
            energy_record_with_metal(0, 0, 0),
            energy_record_with_metal(0, 1, 111),
        ];
        let c = cfg(0.87, 0.0, 1e13, 298.15);
        let by_species = filter_with_fallback(&records, &c);
        assert_eq!(by_species[0].len(), 2);
    }

    #[test]
    fn filter_with_fallback_restricts_to_metal_and_facet() {
        let pd = metal_index("Pd").unwrap();
        let pt = metal_index("Pt").unwrap();
        let records = vec![
            energy_record_with_metal(0, pd, 111), // matches
            energy_record_with_metal(0, pd, 100), // wrong facet
            energy_record_with_metal(0, pt, 111), // wrong metal
        ];
        let mut c = cfg(0.87, 0.0, 1e13, 298.15);
        c.metal = Some(pd);
        c.facet = Some(111);
        // Only 1 record matches metal+facet, below BUCKETS_PER_SPECIES --
        // this exercises the fallback path (see the next test for a case
        // where the fallback shouldn't trigger).
        let by_species = filter_with_fallback(&records, &c);
        // Falls back to metal-only (any facet): both Pd records.
        assert_eq!(by_species[0].len(), 2);
        assert!(by_species[0].iter().all(|r| r.metal == pd));
    }

    #[test]
    fn filter_with_fallback_does_not_broaden_when_facet_pool_is_large_enough() {
        let pd = metal_index("Pd").unwrap();
        let pt = metal_index("Pt").unwrap();
        let mut records: Vec<EnergyRecord> = (0..BUCKETS_PER_SPECIES)
            .map(|_| energy_record_with_metal(0, pd, 111))
            .collect();
        records.push(energy_record_with_metal(0, pt, 111)); // wrong metal, must be excluded
        let mut c = cfg(0.87, 0.0, 1e13, 298.15);
        c.metal = Some(pd);
        c.facet = Some(111);
        let by_species = filter_with_fallback(&records, &c);
        assert_eq!(by_species[0].len(), BUCKETS_PER_SPECIES);
        assert!(by_species[0]
            .iter()
            .all(|r| r.metal == pd && r.facet == 111));
    }

    fn bimolecular_record_with_metal(
        species_a: u8,
        species_b: u8,
        metal: u8,
        facet: u16,
    ) -> BiEnergyRecord {
        BiEnergyRecord {
            species_a,
            species_b,
            sid: 0,
            energy_ev: -0.5,
            ea_ev: 1.0,
            metal,
            facet,
            is_dissociative: false,
        }
    }

    #[test]
    fn filter_bimolecular_with_fallback_broadens_when_facet_pool_is_empty() {
        let pd = metal_index("Pd").unwrap();
        let records = vec![bimolecular_record_with_metal(0, 2, pd, 211)]; // facet 211, not 111
        let mut c = cfg(0.87, 0.0, 1e13, 298.15);
        c.metal = Some(pd);
        c.facet = Some(111);
        let filtered = filter_bimolecular_with_fallback(&records, &c);
        assert_eq!(
            filtered.len(),
            1,
            "should broaden to metal-only rather than drop the only record"
        );
    }

    fn energy_record(energy_ev: f64, real_ea_ev: Option<f64>) -> EnergyRecord {
        EnergyRecord {
            species: 0,
            energy_ev,
            sid: 0,
            real_ea_ev,
            metal: 0,
            facet: 0,
        }
    }

    #[test]
    fn bucket_by_quantile_empty_input_yields_no_buckets() {
        assert!(bucket_by_quantile(&[], 4).is_empty());
    }

    #[test]
    fn bucket_by_quantile_degenerates_gracefully_with_fewer_samples_than_buckets() {
        let records = vec![energy_record(-1.0, None), energy_record(-0.5, None)];
        let buckets = bucket_by_quantile(&records, 4);
        // Never more buckets than samples -- each of the 2 records gets
        // its own bucket, matching the old one-record-per-channel
        // behavior this replaces.
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0].sample_count, 1);
        assert_eq!(buckets[1].sample_count, 1);
        assert!((buckets[0].mean_energy_ev - (-1.0)).abs() < 1e-9);
        assert!((buckets[1].mean_energy_ev - (-0.5)).abs() < 1e-9);
    }

    #[test]
    fn bucket_by_quantile_splits_into_roughly_equal_sorted_groups() {
        // 8 records, energies 0..8 in a shuffled order -- must sort first,
        // then split into 4 buckets of 2 each: {0,1},{2,3},{4,5},{6,7}.
        let energies = [5.0, 1.0, 7.0, 0.0, 3.0, 6.0, 2.0, 4.0];
        let records: Vec<EnergyRecord> = energies.iter().map(|&e| energy_record(e, None)).collect();
        let buckets = bucket_by_quantile(&records, 4);

        assert_eq!(buckets.len(), 4);
        for b in &buckets {
            assert_eq!(b.sample_count, 2);
        }
        assert!((buckets[0].mean_energy_ev - 0.5).abs() < 1e-9); // mean(0,1)
        assert!((buckets[1].mean_energy_ev - 2.5).abs() < 1e-9); // mean(2,3)
        assert!((buckets[2].mean_energy_ev - 4.5).abs() < 1e-9); // mean(4,5)
        assert!((buckets[3].mean_energy_ev - 6.5).abs() < 1e-9); // mean(6,7)
    }

    #[test]
    fn bucket_by_quantile_means_only_the_real_ea_members_within_a_bucket() {
        let records = vec![
            energy_record(0.0, Some(0.10)),
            energy_record(0.1, None),
            energy_record(0.2, Some(0.30)),
        ];
        // All 3 sort into one bucket (num_buckets=1) -- the bucket's
        // real_ea_ev must average only the 2 members that had one, not
        // treat the missing one as 0.
        let buckets = bucket_by_quantile(&records, 1);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].sample_count, 3);
        assert!((buckets[0].real_ea_ev.unwrap() - 0.20).abs() < 1e-9); // mean(0.10, 0.30)
    }

    #[test]
    fn bucket_by_quantile_real_ea_is_none_when_no_member_has_one() {
        let records = vec![energy_record(0.0, None), energy_record(0.1, None)];
        let buckets = bucket_by_quantile(&records, 1);
        assert_eq!(buckets[0].real_ea_ev, None);
    }

    proptest::proptest! {
        /// `bucket_by_quantile`'s doc comment makes three claims the
        /// example tests above only check on small hand-picked inputs
        /// (2, 3, or 8 records): every record ends up in exactly one
        /// bucket (none lost or double-counted), the bucket count is
        /// never more than requested and never more than there are
        /// records, and the split is "roughly equal-sized" -- meaning no
        /// two buckets can differ in size by more than one record, not
        /// just "each bucket gets some records." Checked here for
        /// arbitrary record counts and bucket counts, including sizes
        /// that don't divide evenly (the case most likely to expose an
        /// off-by-one in the `b * n / buckets` boundary arithmetic).
        #[test]
        fn bucket_by_quantile_splits_every_record_into_a_roughly_equal_bucket(
            energies in proptest::collection::vec(-1000.0f64..1000.0, 0..50),
            num_buckets in 1usize..8,
        ) {
            let n = energies.len();
            let records: Vec<EnergyRecord> =
                energies.iter().map(|&e| energy_record(e, None)).collect();
            let buckets = bucket_by_quantile(&records, num_buckets);

            if n == 0 {
                proptest::prop_assert!(buckets.is_empty());
                return Ok(());
            }

            proptest::prop_assert_eq!(
                buckets.len(), num_buckets.min(n),
                "bucket count must be min(requested, record count)"
            );

            let total: usize = buckets.iter().map(|b| b.sample_count).sum();
            proptest::prop_assert_eq!(total, n, "every record must land in exactly one bucket");

            let min_size = buckets.iter().map(|b| b.sample_count).min().unwrap();
            let max_size = buckets.iter().map(|b| b.sample_count).max().unwrap();
            proptest::prop_assert!(
                max_size - min_size <= 1,
                "bucket sizes {:?} are not roughly equal (min={}, max={})",
                buckets.iter().map(|b| b.sample_count).collect::<Vec<_>>(), min_size, max_size
            );
        }
    }

    fn push_record(
        bytes: &mut Vec<u8>,
        species: u8,
        energy_mev: i32,
        sid: u32,
        real_ea_mev: Option<i32>,
    ) {
        push_record_with_metal(bytes, species, energy_mev, sid, real_ea_mev, 0, 0);
    }

    #[allow(clippy::too_many_arguments)]
    fn push_record_with_metal(
        bytes: &mut Vec<u8>,
        species: u8,
        energy_mev: i32,
        sid: u32,
        real_ea_mev: Option<i32>,
        metal: u8,
        facet: u16,
    ) {
        bytes.push(species);
        bytes.extend_from_slice(&energy_mev.to_le_bytes());
        bytes.extend_from_slice(&sid.to_le_bytes());
        bytes.push(real_ea_mev.is_some() as u8);
        bytes.extend_from_slice(&real_ea_mev.unwrap_or(0).to_le_bytes());
        bytes.push(metal);
        bytes.extend_from_slice(&facet.to_le_bytes());
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

    // `read_energy_records`'s bad-magic and truncated-file rejection are
    // now tested directly in `oc20e_format`'s own test module, where the
    // function lives -- see `read_energy_records_rejects_bad_magic` and
    // `read_energy_records_rejects_truncated_file_with_inflated_count`
    // there. The round-trip test above stays here since it also exercises
    // this binary's own species-filtering usage of the reader.

    fn push_bimolecular_record(
        bytes: &mut Vec<u8>,
        species_a: u8,
        species_b: u8,
        energy_mev: i32,
        sid: u32,
        ea_mev: i32,
    ) {
        push_bimolecular_record_with_metal(
            bytes, species_a, species_b, energy_mev, sid, ea_mev, 0, 0, false,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn push_bimolecular_record_with_metal(
        bytes: &mut Vec<u8>,
        species_a: u8,
        species_b: u8,
        energy_mev: i32,
        sid: u32,
        ea_mev: i32,
        metal: u8,
        facet: u16,
        is_dissociative: bool,
    ) {
        bytes.push(species_a);
        bytes.push(species_b);
        bytes.extend_from_slice(&energy_mev.to_le_bytes());
        bytes.extend_from_slice(&sid.to_le_bytes());
        bytes.extend_from_slice(&ea_mev.to_le_bytes());
        bytes.push(metal);
        bytes.extend_from_slice(&facet.to_le_bytes());
        bytes.push(is_dissociative as u8);
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

    // `read_bimolecular_records`'s bad-magic and truncated-file rejection
    // are now tested directly in `oc20e_format`'s own test module -- see
    // `read_bimolecular_records_rejects_bad_magic` and
    // `read_bimolecular_records_rejects_truncated_file_with_inflated_count`
    // there. The round-trip test above stays here since it also exercises
    // this binary's own species-filtering usage of the reader.

    #[test]
    fn run_builds_a_single_bimolecular_reaction_with_no_reverse() {
        let input_path = temp_path("run_bi_mono_input");
        let bi_path = temp_path("run_bi_bimolecular_input");
        let out_path = temp_path("run_bi_out.lut");

        // One ordinary CO adsorption/desorption pair (CO adsorbs
        // molecularly, not dissociatively, so this stays a plain
        // monomolecular pair -- keeps this test focused on the
        // recombination bimolecular record below, not on O/H's separate
        // dissociative-adsorption behavior), so `energy_records` isn't
        // empty (run() rejects that up front).
        let mut mono_bytes = Vec::new();
        mono_bytes.extend_from_slice(MAGIC);
        mono_bytes.extend_from_slice(&1u32.to_le_bytes());
        push_record(&mut mono_bytes, 2, -500, 1, None);
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
            metal: None,
            facet: None,
            species: default_species(),
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
        assert_eq!(
            bimolecular.len(),
            1,
            "exactly one bimolecular reaction, no reverse built"
        );
        let r = bimolecular[0];
        assert_eq!(r.transition_a, (layout::ADS_O as u16) << 8); // O* -> vacant
        assert_eq!(r.transition_b, (layout::ADS_CO as u16) << 8); // CO* -> vacant

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
            metal: None,
            facet: None,
            species: default_species(),
        };
        run(&config).unwrap();

        let lut = layout::ReactionLut::open(&out_path).unwrap();
        let reaction_count = lut.len() * ReactionLutBlock::LANES;
        let real_records: Vec<_> = (0..reaction_count)
            .map(|id| lut.rate_of(id))
            .filter(|r| r.rate_q16 > 0)
            .collect();
        // O desorption (monomolecular) + O dissociative adsorption
        // (bimolecular, both species are dissociative -- see
        // DISSOCIATIVE_SPECIES) + H dissociative adsorption (bimolecular)
        // + 1 recombination bimolecular (H desorption is replaced, not
        // built).
        assert_eq!(real_records.len(), 4);

        let o_desorption = real_records
            .iter()
            .filter(|r| !r.is_bimolecular && r.transition_a == (layout::ADS_O as u16) << 8)
            .count();
        assert_eq!(
            o_desorption, 1,
            "O's monomolecular desorption must be untouched"
        );

        let h_desorption = real_records
            .iter()
            .filter(|r| !r.is_bimolecular && r.transition_a == (layout::ADS_H as u16) << 8)
            .count();
        assert_eq!(
            h_desorption, 0,
            "H's monomolecular desorption must be replaced by the bimolecular record"
        );

        let o_dissociative_adsorption = real_records
            .iter()
            .filter(|r| {
                r.is_bimolecular
                    && r.transition_a == layout::ADS_O as u16
                    && r.transition_b == layout::ADS_O as u16
            })
            .count();
        assert_eq!(
            o_dissociative_adsorption, 1,
            "O's dissociative adsorption is built regardless of desorption replacement"
        );

        let h_dissociative_adsorption = real_records
            .iter()
            .filter(|r| {
                r.is_bimolecular
                    && r.transition_a == layout::ADS_H as u16
                    && r.transition_b == layout::ADS_H as u16
            })
            .count();
        assert_eq!(
            h_dissociative_adsorption, 1,
            "H's dissociative adsorption is unaffected by desorption replacement -- only desorption is replaced"
        );

        let recombination_bimolecular: Vec<_> = real_records
            .iter()
            .filter(|r| r.is_bimolecular && r.transition_a == (layout::ADS_H as u16) << 8)
            .collect();
        assert_eq!(recombination_bimolecular.len(), 1);
        assert_eq!(
            recombination_bimolecular[0].transition_a,
            (layout::ADS_H as u16) << 8
        );
        assert_eq!(
            recombination_bimolecular[0].transition_b,
            (layout::ADS_H as u16) << 8
        );

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&bi_path);
        let _ = std::fs::remove_file(&out_path);
    }

    #[test]
    fn run_collapses_many_samples_into_bucketed_templates_not_one_per_sample() {
        let input_path = temp_path("run_bucketing_input");
        let out_path = temp_path("run_bucketing_out.lut");

        // 12 real CO adsorption-energy records -- more than
        // BUCKETS_PER_SPECIES (4), so this must collapse to 4 adsorption +
        // 4 desorption templates, not 12 + 12. CO specifically (not O/H):
        // this test is about quantile-bucketing behavior in general, which
        // is orthogonal to O/H's separate dissociative-adsorption handling
        // (dissociative-adsorption records are deliberately un-bucketed,
        // bin_id always 0 -- see DISSOCIATIVE_SPECIES).
        let mut mono_bytes = Vec::new();
        mono_bytes.extend_from_slice(MAGIC);
        mono_bytes.extend_from_slice(&12u32.to_le_bytes());
        for i in 0..12 {
            push_record(&mut mono_bytes, 2, -100 * i, i as u32, None);
        }
        std::fs::write(&input_path, &mono_bytes).unwrap();

        let config = Config {
            input: input_path.clone(),
            bimolecular_input: None,
            out: out_path.clone(),
            alpha: 0.87,
            beta_ev: 0.0,
            nu: 1.0e13,
            temperature_k: 298.15,
            metal: None,
            facet: None,
            species: default_species(),
        };
        run(&config).unwrap();

        let lut = layout::ReactionLut::open(&out_path).unwrap();
        assert_eq!(lut.kind(), layout::LutKind::OccupancyGated);
        let reaction_count = lut.len() * ReactionLutBlock::LANES;
        let real_records: Vec<_> = (0..reaction_count)
            .map(|id| lut.rate_of(id))
            .filter(|r| r.rate_q16 > 0)
            .collect();

        // 4 buckets x (adsorption + desorption) = 8, not 12 + 12 = 24.
        assert_eq!(real_records.len(), 8);

        let ads_bin_ids: std::collections::BTreeSet<u8> = real_records
            .iter()
            .filter(|r| r.transition_a == layout::ADS_CO as u16)
            .map(|r| r.bin_id)
            .collect();
        assert_eq!(ads_bin_ids, [0u8, 1, 2, 3].into_iter().collect());

        let des_bin_ids: std::collections::BTreeSet<u8> = real_records
            .iter()
            .filter(|r| r.transition_a == (layout::ADS_CO as u16) << 8)
            .map(|r| r.bin_id)
            .collect();
        assert_eq!(des_bin_ids, [0u8, 1, 2, 3].into_iter().collect());

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&out_path);
    }

    /// The falsifying test Phase 3 (genuine two-site dissociative
    /// adsorption) exists to pass at the ingest level: O and H (both
    /// dissociate from a diatomic gas) must build a bimolecular
    /// `VACANT -> species` record on both sites for adsorption, while CO
    /// (adsorbs molecularly) keeps the original monomolecular form.
    #[test]
    fn run_builds_dissociative_adsorption_for_o_and_h_but_not_co() {
        let input_path = temp_path("run_dissociative_input");
        let out_path = temp_path("run_dissociative_out.lut");

        let mut mono_bytes = Vec::new();
        mono_bytes.extend_from_slice(MAGIC);
        mono_bytes.extend_from_slice(&3u32.to_le_bytes());
        push_record(&mut mono_bytes, 0, -500, 1, None); // O
        push_record(&mut mono_bytes, 1, -400, 2, None); // H
        push_record(&mut mono_bytes, 2, -300, 3, None); // CO
        std::fs::write(&input_path, &mono_bytes).unwrap();

        let config = Config {
            input: input_path.clone(),
            bimolecular_input: None,
            out: out_path.clone(),
            alpha: 0.87,
            beta_ev: 0.0,
            nu: 1.0e13,
            temperature_k: 298.15,
            metal: None,
            facet: None,
            species: default_species(),
        };
        run(&config).unwrap();

        let lut = layout::ReactionLut::open(&out_path).unwrap();
        let reaction_count = lut.len() * ReactionLutBlock::LANES;
        let real_records: Vec<_> = (0..reaction_count)
            .map(|id| lut.rate_of(id))
            .filter(|r| r.rate_q16 > 0)
            .collect();

        for (name, bit) in [("O", layout::ADS_O), ("H", layout::ADS_H)] {
            let bit = bit as u16;
            let dissociative = real_records
                .iter()
                .filter(|r| r.is_bimolecular && r.transition_a == bit && r.transition_b == bit)
                .count();
            assert_eq!(
                dissociative, 1,
                "{name} adsorption must be a two-site dissociative record"
            );
            let monomolecular_ads = real_records
                .iter()
                .filter(|r| !r.is_bimolecular && r.transition_a == bit)
                .count();
            assert_eq!(
                monomolecular_ads, 0,
                "{name} must not also have a monomolecular adsorption record"
            );
        }

        let co_monomolecular_ads = real_records
            .iter()
            .filter(|r| !r.is_bimolecular && r.transition_a == layout::ADS_CO as u16)
            .count();
        assert_eq!(
            co_monomolecular_ads, 1,
            "CO adsorbs molecularly -- must stay monomolecular"
        );
        let co_dissociative = real_records
            .iter()
            .filter(|r| r.is_bimolecular && r.transition_a == layout::ADS_CO as u16)
            .count();
        assert_eq!(
            co_dissociative, 0,
            "CO must not be built as a dissociative-adsorption record"
        );

        // Desorption is untouched for all three species regardless of the
        // adsorption-direction change.
        for bit in [layout::ADS_O, layout::ADS_H, layout::ADS_CO] {
            let desorption = real_records
                .iter()
                .filter(|r| !r.is_bimolecular && r.transition_a == (bit as u16) << 8)
                .count();
            assert_eq!(desorption, 1);
        }

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&out_path);
    }

    /// The falsifying test Phase 4 (broader reaction coverage: OH* via
    /// water splitting) exists to pass: a real `is_dissociative` barrier
    /// must build *both* a forward (dissociative-adsorption) record using
    /// the real Ea directly and a reverse (associative-desorption) record
    /// via thermodynamic consistency (`Ea_rev = Ea_fwd - dE_rxn`) --
    /// unlike a recombination-direction record (`is_dissociative = 0`,
    /// covered by `run_builds_a_single_bimolecular_reaction_with_no_reverse`
    /// above), which stays forward-only.
    #[test]
    fn run_builds_both_directions_for_a_dissociative_bimolecular_record() {
        let input_path = temp_path("run_dissociative_bi_input");
        let bi_path = temp_path("run_dissociative_bi_bimolecular_input");
        let out_path = temp_path("run_dissociative_bi_out.lut");

        // One CO record, energy_mev = 0, so `energy_records` isn't empty
        // (CO isn't one of DISSOCIATIVE_SPECIES, so it doesn't interact
        // with the water-splitting record below).
        let mut mono_bytes = Vec::new();
        mono_bytes.extend_from_slice(MAGIC);
        mono_bytes.extend_from_slice(&1u32.to_le_bytes());
        push_record(&mut mono_bytes, 2, 0, 1, None);
        std::fs::write(&input_path, &mono_bytes).unwrap();

        // One real water-splitting barrier: 2* + H2O(g) -> H* + OH*,
        // dE_rxn = 0.220 eV, Ea_fwd = 1.011 eV (species 1 = H, species 3 = OH).
        let mut bi_bytes = Vec::new();
        bi_bytes.extend_from_slice(MAGIC_BI);
        bi_bytes.extend_from_slice(&1u32.to_le_bytes());
        push_bimolecular_record_with_metal(&mut bi_bytes, 1, 3, 220, 42, 1011, 0, 0, true);
        std::fs::write(&bi_path, &bi_bytes).unwrap();

        // beta_ev = 0.5 (not the usual default 0.0): with energy_mev = 0
        // above, BEP's `Ea = max(0, alpha*dE + beta)` gives CO the *same*
        // non-zero barrier (0.5 eV) in both directions -- avoiding the
        // BEP relation's structural property that any nonzero dE clamps
        // *one* direction to a barrierless Ea = 0 (since alpha != 1), a
        // near-barrierless competing channel would dominate the Q16.16
        // rescaling and round *both* water-splitting directions down to
        // the same rate_q16 floor, defeating this test's point (checking
        // their *relative* ordering survives rescaling).
        let config = Config {
            input: input_path.clone(),
            bimolecular_input: Some(bi_path.clone()),
            out: out_path.clone(),
            alpha: 0.87,
            beta_ev: 0.5,
            nu: 1.0e13,
            temperature_k: 298.15,
            metal: None,
            facet: None,
            species: default_species(),
        };
        run(&config).unwrap();

        let lut = layout::ReactionLut::open(&out_path).unwrap();
        let reaction_count = lut.len() * ReactionLutBlock::LANES;
        let real_records: Vec<_> = (0..reaction_count)
            .map(|id| lut.rate_of(id))
            .filter(|r| r.rate_q16 > 0)
            .collect();

        let forward: Vec<_> = real_records
            .iter()
            .filter(|r| {
                r.is_bimolecular
                    && r.transition_a == layout::ADS_H as u16
                    && r.transition_b == layout::ADS_OH as u16
            })
            .collect();
        assert_eq!(
            forward.len(),
            1,
            "forward dissociative-adsorption record must be built"
        );

        let reverse: Vec<_> = real_records
            .iter()
            .filter(|r| {
                r.is_bimolecular
                    && r.transition_a == (layout::ADS_H as u16) << 8
                    && r.transition_b == (layout::ADS_OH as u16) << 8
            })
            .collect();
        assert_eq!(
            reverse.len(),
            1,
            "reverse associative-desorption record must be built too"
        );

        // Forward uses the real Ea directly; reverse uses Ea_rev = Ea_fwd
        // - dE_rxn = 1.011 - 0.220 = 0.791, a smaller barrier, so its rate
        // constant (before Q16.16 rescaling) is larger -- confirm the
        // *ordering* survives rescaling: reverse's rate_q16 must be
        // greater than forward's.
        assert!(
            reverse[0].rate_q16 > forward[0].rate_q16,
            "reverse (smaller barrier) should have a larger rate_q16 than forward: \
             forward={} reverse={}",
            forward[0].rate_q16,
            reverse[0].rate_q16
        );

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&bi_path);
        let _ = std::fs::remove_file(&out_path);
    }
}
