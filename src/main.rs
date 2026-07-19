//! CLI entrypoint for `kinetica`, an asynchronous, out-of-core lattice kMC
//! engine. This binary wires together the three architectural pieces the
//! rest of the crate provides:
//!
//! * [`layout::SiteLattice`] -- the memory-mapped, bit-packed catalyst
//!   surface.
//! * [`layout::ReactionLut`] -- the cache-line-aligned OC20 rate-constant
//!   table.
//! * [`engine::run_simulation`] -- the `rayon` work-stealing + `crossbeam`
//!   boundary-channel + `io_uring` trajectory-logging execution pipeline.
//!
//! Argument parsing is hand-rolled rather than pulled in from a crate: the
//! architecture spec pins this project's dependency list to exactly
//! `memmap2`, `rio`, `rayon`, and `crossbeam-channel`, so a CLI-parsing
//! dependency would be scope creep.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use kinetica::{engine, gillespie, layout};
use layout::{ReactionLut, ReactionLutBlock, SiteLattice};

struct Config {
    lattice_path: PathBuf,
    lattice_width: usize,
    lattice_height: usize,
    lut_path: PathBuf,
    trajectory_path: PathBuf,
    patches: usize,
    steps_per_patch: u64,
    /// `Some(n)` => synthesize `n` demo reaction records into `lut_path`
    /// before running, for exercising the pipeline without a real OC20
    /// rate-constant export on hand.
    generate_lut: Option<usize>,
    /// Relative partial pressures for the O2/H2/CO/H2O feed gas, applied
    /// only to the occupancy-gated engine's adsorption channels (see
    /// `occupancy::Pressures`). Named after the gas-phase molecule fed in
    /// (O2, H2, CO, H2O), not the surface species index it couples to.
    pressure_o2: f64,
    pressure_h2: f64,
    pressure_co: f64,
    pressure_h2o: f64,
}

impl Config {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut args = args;
        let _bin = args.next();

        let mut lattice_path = PathBuf::from("surface.lattice");
        let mut lattice_width = 4096usize;
        let mut lattice_height = 4096usize;
        let mut lut_path = PathBuf::from("reactions.lut");
        let mut trajectory_path = PathBuf::from("trajectory.bin");
        let mut patches = rayon::current_num_threads();
        let mut steps_per_patch = 1_000_000u64;
        let mut generate_lut = None;
        let mut pressure_o2 = 1.0f64;
        let mut pressure_h2 = 1.0f64;
        let mut pressure_co = 1.0f64;
        let mut pressure_h2o = 1.0f64;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--lattice-path" => {
                    lattice_path = PathBuf::from(next_value(&mut args, "--lattice-path")?)
                }
                "--lattice-width" => lattice_width = parse_value(&mut args, "--lattice-width")?,
                "--lattice-height" => lattice_height = parse_value(&mut args, "--lattice-height")?,
                "--lut-path" => lut_path = PathBuf::from(next_value(&mut args, "--lut-path")?),
                "--trajectory-path" => {
                    trajectory_path = PathBuf::from(next_value(&mut args, "--trajectory-path")?)
                }
                "--patches" => patches = parse_value(&mut args, "--patches")?,
                "--steps" => steps_per_patch = parse_value(&mut args, "--steps")?,
                "--generate-lut" => generate_lut = Some(parse_value(&mut args, "--generate-lut")?),
                "--pressure-o2" => pressure_o2 = parse_value(&mut args, "--pressure-o2")?,
                "--pressure-h2" => pressure_h2 = parse_value(&mut args, "--pressure-h2")?,
                "--pressure-co" => pressure_co = parse_value(&mut args, "--pressure-co")?,
                "--pressure-h2o" => pressure_h2o = parse_value(&mut args, "--pressure-h2o")?,
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
            }
        }

        Ok(Self {
            lattice_path,
            lattice_width,
            lattice_height,
            lut_path,
            trajectory_path,
            patches,
            steps_per_patch,
            generate_lut,
            pressure_o2,
            pressure_h2,
            pressure_co,
            pressure_h2o,
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
    "kinetica: async out-of-core lattice kMC engine\n\n\
     USAGE:\n    \
       kinetica [OPTIONS]\n\n\
     OPTIONS:\n    \
       --lattice-path <PATH>      Backing mmap file for the surface [default: surface.lattice]\n    \
       --lattice-width <N>        Lattice width in sites [default: 4096]\n    \
       --lattice-height <N>       Lattice height in sites [default: 4096]\n    \
       --lut-path <PATH>          reactions.lut rate-constant table [default: reactions.lut]\n    \
       --trajectory-path <PATH>   Output trajectory log [default: trajectory.bin]\n    \
       --patches <N>              Spatial domains / rayon tasks [default: available CPUs]\n    \
       --steps <N>                Gillespie steps per patch [default: 1000000]\n    \
       --generate-lut <N>         Synthesize N demo reactions into --lut-path first\n    \
       --pressure-o2 <F>          Relative O2 partial pressure, gates O* adsorption\n                                    \
                                    (occupancy-gated engine only) [default: 1.0]\n    \
       --pressure-h2 <F>          Relative H2 partial pressure, gates H* adsorption\n                                    \
                                    [default: 1.0]\n    \
       --pressure-co <F>          Relative CO partial pressure, gates CO* adsorption\n                                    \
                                    [default: 1.0]\n    \
       --pressure-h2o <F>         Relative H2O partial pressure, gates H2O* adsorption\n                                    \
                                    (does not affect water-splitting -- see README)\n                                    \
                                    [default: 1.0]\n    \
       -h, --help                 Print this message"
        .to_string()
}

fn main() -> ExitCode {
    let config = match Config::parse(std::env::args()) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };

    match run(&config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("kinetica: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(config: &Config) -> std::io::Result<()> {
    if let Some(count) = config.generate_lut {
        generate_demo_lut(&config.lut_path, count)?;
        println!(
            "kinetica: wrote {count} synthetic reactions to {}",
            config.lut_path.display()
        );
    }

    println!(
        "kinetica: opening lattice {}x{} at {}",
        config.lattice_width,
        config.lattice_height,
        config.lattice_path.display()
    );
    let mut lattice = SiteLattice::open(
        &config.lattice_path,
        config.lattice_width,
        config.lattice_height,
    )?;

    println!(
        "kinetica: mapping reaction LUT {}",
        config.lut_path.display()
    );
    let lut = ReactionLut::open(&config.lut_path)?;
    let engine_name = match lut.kind() {
        layout::LutKind::Static => "static composition-rejection (gillespie.rs)",
        layout::LutKind::OccupancyGated => "occupancy-gated (occupancy.rs)",
    };
    println!(
        "kinetica: {} blocks ({} reactions) mapped from {} -- {engine_name} engine",
        lut.len(),
        lut.len() * ReactionLutBlock::LANES,
        config.lut_path.display()
    );

    println!(
        "kinetica: fanning out across {} patch(es), {} steps/patch -> {}",
        config.patches,
        config.steps_per_patch,
        config.trajectory_path.display()
    );
    if lut.kind() == layout::LutKind::OccupancyGated {
        println!(
            "kinetica: relative partial pressures: O2={} H2={} CO={} H2O={}",
            config.pressure_o2, config.pressure_h2, config.pressure_co, config.pressure_h2o
        );
    }
    // Index 3 (OH) has no independent gas-phase pressure knob -- OH only
    // ever forms via the heteroatomic water-splitting reaction, which
    // `occupancy::pressure_factor` always treats as pressure-neutral (see
    // its doc comment), so this slot is never actually read. Kept at 1.0
    // rather than omitted so the array stays indexed exactly like
    // `layout::SPECIES_BITS`. Indices 5..MAX_SPECIES (beyond today's 5
    // active species) are unused padding, same reasoning as
    // `Pressures.values`'s own doc comment.
    let pressures = kinetica::occupancy::Pressures {
        values: [
            config.pressure_o2,
            config.pressure_h2,
            config.pressure_co,
            1.0,
            config.pressure_h2o,
            1.0,
            1.0,
            1.0,
        ],
    };

    let start = Instant::now();
    engine::run_simulation(
        &mut lattice,
        &lut,
        config.patches,
        config.steps_per_patch,
        &config.trajectory_path,
        &pressures,
    )?;
    let elapsed = start.elapsed();

    lattice.flush()?;

    let total_steps = config.patches as u64 * config.steps_per_patch;
    let rate = total_steps as f64 / elapsed.as_secs_f64().max(1e-9);
    println!(
        "kinetica: done in {:.3}s ({:.0} reactions/sec)",
        elapsed.as_secs_f64(),
        rate
    );

    Ok(())
}

/// Synthesize a `reactions.lut` file of `count` demo reaction records so
/// the pipeline can be exercised end to end without a real OC20
/// transition-state export on hand. Not part of the core architecture --
/// a development convenience only.
fn generate_demo_lut(path: &std::path::Path, count: usize) -> std::io::Result<()> {
    let block_count = count.div_ceil(ReactionLutBlock::LANES).max(1);
    let mut rng = gillespie::Rng::seeded(0xC0FF_EE00_C0FF_EE00);

    // `layout::pack_records_into_blocks` sorts these by bin_id and packs
    // them into cache-line blocks; see its doc comment for why that
    // ordering matters to `CompositionTable::build`. About 1 in 8 records
    // are synthesized as bimolecular (two-site) reactions purely so
    // `--generate-lut` exercises engine.rs's bimolecular path even without
    // real Langmuir-Hinshelwood data on hand -- see `oc20_ingest.rs` for
    // where a *real* bimolecular reaction (CO oxidation) is wired in.
    let records: Vec<layout::ReactionRecord> = (0..block_count * ReactionLutBlock::LANES)
        .map(|_| {
            let raw = rng.next_u64();
            let rate_q16 = ((raw & 0x00FF_FFFF) as u32).max(1);
            let bin_id = (31 - rate_q16.leading_zeros()) as u8;
            // Packed (reactant_mask << 8) | product_mask demo transitions,
            // restricted to the bitflags defined in layout.rs.
            let reactant_a = ((raw >> 24) & 0x7) as u16;
            let product_a = ((raw >> 27) & 0x7) as u16;
            let transition_a = (reactant_a << 8) | product_a;

            let is_bimolecular = (raw >> 30) & 0x7 == 0;
            let reactant_b = ((raw >> 33) & 0x7) as u16;
            let product_b = ((raw >> 36) & 0x7) as u16;
            let transition_b = if is_bimolecular {
                (reactant_b << 8) | product_b
            } else {
                0
            };

            layout::ReactionRecord {
                rate_q16,
                bin_id,
                transition_a,
                transition_b,
                is_bimolecular,
            }
        })
        .collect();

    let blocks = layout::pack_records_into_blocks(records);
    layout::write_lut(path, layout::LutKind::Static, &blocks)
}
