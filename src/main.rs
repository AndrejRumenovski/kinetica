//! CLI entrypoint for `cattrace`, an asynchronous, out-of-core lattice kMC
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

mod engine;
mod gillespie;
mod layout;

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

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

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--lattice-path" => {
                    lattice_path = PathBuf::from(next_value(&mut args, "--lattice-path")?)
                }
                "--lattice-width" => {
                    lattice_width = parse_value(&mut args, "--lattice-width")?
                }
                "--lattice-height" => {
                    lattice_height = parse_value(&mut args, "--lattice-height")?
                }
                "--lut-path" => lut_path = PathBuf::from(next_value(&mut args, "--lut-path")?),
                "--trajectory-path" => {
                    trajectory_path = PathBuf::from(next_value(&mut args, "--trajectory-path")?)
                }
                "--patches" => patches = parse_value(&mut args, "--patches")?,
                "--steps" => steps_per_patch = parse_value(&mut args, "--steps")?,
                "--generate-lut" => generate_lut = Some(parse_value(&mut args, "--generate-lut")?),
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
    "cattrace: async out-of-core lattice kMC engine\n\n\
     USAGE:\n    \
       cattrace [OPTIONS]\n\n\
     OPTIONS:\n    \
       --lattice-path <PATH>      Backing mmap file for the surface [default: surface.lattice]\n    \
       --lattice-width <N>        Lattice width in sites [default: 4096]\n    \
       --lattice-height <N>       Lattice height in sites [default: 4096]\n    \
       --lut-path <PATH>          reactions.lut rate-constant table [default: reactions.lut]\n    \
       --trajectory-path <PATH>   Output trajectory log [default: trajectory.bin]\n    \
       --patches <N>              Spatial domains / rayon tasks [default: available CPUs]\n    \
       --steps <N>                Gillespie steps per patch [default: 1000000]\n    \
       --generate-lut <N>         Synthesize N demo reactions into --lut-path first\n    \
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
            eprintln!("cattrace: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(config: &Config) -> std::io::Result<()> {
    if let Some(count) = config.generate_lut {
        generate_demo_lut(&config.lut_path, count)?;
        println!(
            "cattrace: wrote {count} synthetic reactions to {}",
            config.lut_path.display()
        );
    }

    println!(
        "cattrace: opening lattice {}x{} at {}",
        config.lattice_width,
        config.lattice_height,
        config.lattice_path.display()
    );
    let mut lattice = SiteLattice::open(&config.lattice_path, config.lattice_width, config.lattice_height)?;

    println!("cattrace: mapping reaction LUT {}", config.lut_path.display());
    let lut = ReactionLut::open(&config.lut_path)?;
    println!(
        "cattrace: {} blocks ({} reactions) mapped from {}",
        lut.len(),
        lut.len() * ReactionLutBlock::LANES,
        config.lut_path.display()
    );

    println!(
        "cattrace: fanning out across {} patch(es), {} steps/patch -> {}",
        config.patches,
        config.steps_per_patch,
        config.trajectory_path.display()
    );

    let start = Instant::now();
    engine::run_simulation(
        &mut lattice,
        &lut,
        config.patches,
        config.steps_per_patch,
        &config.trajectory_path,
    )?;
    let elapsed = start.elapsed();

    lattice.flush()?;

    let total_steps = config.patches as u64 * config.steps_per_patch;
    let rate = total_steps as f64 / elapsed.as_secs_f64().max(1e-9);
    println!(
        "cattrace: done in {:.3}s ({:.0} reactions/sec)",
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

    // (rate_q16, bin_id, transition) triples, sorted by bin_id ascending
    // afterward -- `CompositionTable::build` (gillespie.rs) requires
    // `reactions.lut` to already be grouped by magnitude class so bin
    // membership collapses to a contiguous `[start, count)` range instead
    // of needing a separately allocated index.
    let mut records: Vec<(u32, u8, u8)> = (0..block_count * ReactionLutBlock::LANES)
        .map(|_| {
            let raw = rng.next_u64();
            let rate_q16 = ((raw & 0x00FF_FFFF) as u32).max(1);
            let bin_id = (31 - rate_q16.leading_zeros()) as u8;
            // Packed (reactant_mask << 4) | product_mask demo transition,
            // restricted to the bitflags defined in layout.rs.
            let reactant = ((raw >> 24) & 0x7) as u8;
            let product = ((raw >> 27) & 0x7) as u8;
            let transition = (reactant << 4) | product;
            (rate_q16, bin_id, transition)
        })
        .collect();
    records.sort_by_key(|&(_, bin_id, _)| bin_id);

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

    // SAFETY: `ReactionLutBlock` is `repr(C, align(64))`, `Copy`, and every
    // field is a plain fixed-width integer array with no padding bytes
    // (enforced by the `size_of::<ReactionLutBlock>() == 64` assertion in
    // layout.rs), so reinterpreting a `&[ReactionLutBlock]` as `&[u8]` for
    // the duration of this write is a sound, lossless byte-for-byte view --
    // there is no uninitialized padding to expose and no lifetime hazard
    // since the byte slice does not outlive `blocks`.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            blocks.len() * std::mem::size_of::<ReactionLutBlock>(),
        )
    };

    let mut file = std::fs::File::create(path)?;
    file.write_all(bytes)?;
    Ok(())
}
