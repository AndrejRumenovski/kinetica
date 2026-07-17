//! Replays a `trajectory.bin` fired-reaction log against `reactions.lut`
//! to reconstruct per-species surface coverage over simulated time, and
//! prints it as CSV -- input for `scripts/plot_coverage.py`.
//!
//! `trajectory.bin` only logs `(sim_time, site_idx, reaction_id)` per
//! fired event, not the resulting occupancy state at that point -- the
//! state has to be replayed. This starts from an all-`VACANT` scratch
//! lattice (the same state `SiteLattice::open` creates for a fresh path)
//! and applies each event's `layout::apply_transition` in **simulated-time
//! order**, not file order: `engine::run_simulation` fans out fully
//! independent per-patch loops (see `engine.rs`'s module doc) that each
//! advance their own `sim_time` independently and write to the same
//! trajectory file through two alternating `io_uring` writer threads, so
//! the order records land in the file reflects whichever patch/writer
//! happened to be scheduled first, not global chronological order. Every
//! record carries its own `sim_time`, so sorting by it before replay is
//! what makes "coverage at time T" well-defined across patches at all.
//!
//! Kept as a separate tool rather than folded into `kinetica` itself: the
//! main binary's job is running the simulation as fast as possible, not
//! parsing its own output back -- and a corrupted/truncated trajectory
//! file (from a killed run) should never be able to affect a real
//! simulation, only this offline analysis pass.

use std::io::{self, Read};
use std::path::PathBuf;

use kinetica::layout::{self, ReactionLut, NUM_SPECIES, SPECIES_BITS};

/// One `TrajectoryRecord` as `engine.rs` writes it: `sim_time_bits: u64`,
/// `site_idx: u32`, `reaction_id: u32` -- 16 bytes, `repr(C)`, no padding
/// (every field is already naturally aligned in this order). Decoded here
/// independently rather than importing `engine`'s private type, so this
/// tool has no dependency on anything not part of the on-disk format
/// itself.
const RECORD_SIZE: usize = 16;

struct Record {
    sim_time: f64,
    site_idx: u32,
    reaction_id: u32,
}

/// Parse every record in `bytes`, skipping the trailing zero-padding
/// `engine.rs`'s O_DIRECT page writer leaves in a trajectory file's final
/// page (every write is a fixed 4096-byte page; a page that's only
/// partially filled when the run ends is flushed anyway, with its unused
/// tail left zeroed). A genuine event's `sim_time` is always the strictly
/// positive result of an exponential waiting-time draw applied to a
/// nonzero starting time, so `sim_time_bits == 0` unambiguously means
/// "unwritten padding, not a real event at time zero" -- this tool's one
/// load-bearing assumption about the format beyond the fixed record
/// layout itself.
fn parse_records(bytes: &[u8]) -> Vec<Record> {
    bytes
        .chunks_exact(RECORD_SIZE)
        .filter_map(|chunk| {
            let sim_time_bits = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
            if sim_time_bits == 0 {
                return None;
            }
            let site_idx = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
            let reaction_id = u32::from_le_bytes(chunk[12..16].try_into().unwrap());
            Some(Record {
                sim_time: f64::from_bits(sim_time_bits),
                site_idx,
                reaction_id,
            })
        })
        .collect()
}

struct Config {
    trajectory_path: PathBuf,
    lut_path: PathBuf,
    lattice_width: usize,
    lattice_height: usize,
    sample_every: usize,
}

impl Config {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut args = args;
        let _bin = args.next();

        let mut trajectory_path = PathBuf::from("trajectory.bin");
        let mut lut_path = PathBuf::from("reactions.lut");
        let mut lattice_width = 4096usize;
        let mut lattice_height = 4096usize;
        let mut sample_every = 10_000usize;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--trajectory-path" => trajectory_path = PathBuf::from(next(&mut args, "--trajectory-path")?),
                "--lut-path" => lut_path = PathBuf::from(next(&mut args, "--lut-path")?),
                "--lattice-width" => lattice_width = value(&mut args, "--lattice-width")?,
                "--lattice-height" => lattice_height = value(&mut args, "--lattice-height")?,
                "--sample-every" => sample_every = value(&mut args, "--sample-every")?,
                "--help" | "-h" => {
                    return Err(
                        "coverage_report: replay trajectory.bin into per-species coverage CSV\n\n\
                         OPTIONS:\n    \
                           --trajectory-path <PATH>  [default: trajectory.bin]\n    \
                           --lut-path <PATH>         [default: reactions.lut]\n    \
                           --lattice-width <N>       must match the run that produced the trajectory\n    \
                           --lattice-height <N>      must match the run that produced the trajectory\n    \
                           --sample-every <N>        events between CSV rows [default: 10000]"
                            .to_string(),
                    )
                }
                other => return Err(format!("unrecognized argument `{other}`")),
            }
        }

        Ok(Self {
            trajectory_path,
            lut_path,
            lattice_width,
            lattice_height,
            sample_every,
        })
    }
}

fn next(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("`{flag}` requires a value"))
}

fn value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    let raw = next(args, flag)?;
    raw.parse()
        .map_err(|_| format!("`{flag}` expects a number, got `{raw}`"))
}

fn run(config: &Config) -> io::Result<()> {
    let lut = ReactionLut::open(&config.lut_path)?;

    let mut bytes = Vec::new();
    std::fs::File::open(&config.trajectory_path)?.read_to_end(&mut bytes)?;
    let mut records = parse_records(&bytes);
    records.sort_by(|a, b| a.sim_time.total_cmp(&b.sim_time));
    eprintln!(
        "coverage_report: {} real events parsed from {}",
        records.len(),
        config.trajectory_path.display()
    );

    let site_count = config.lattice_width * config.lattice_height;
    let mut lattice = vec![layout::VACANT; site_count];

    println!(
        "event,sim_time,vacant,{}",
        SPECIES_BITS
            .iter()
            .enumerate()
            .map(|(i, _)| species_name(i))
            .collect::<Vec<_>>()
            .join(",")
    );
    print_snapshot(0, 0.0, &lattice);

    for (i, record) in records.iter().enumerate() {
        let site_idx = record.site_idx as usize;
        if site_idx >= lattice.len() {
            continue; // stale/mismatched --lattice-width/--lattice-height
        }
        let r = lut.rate_of(record.reaction_id as usize);
        lattice[site_idx] = layout::apply_transition(lattice[site_idx], r.transition_a);

        let event_number = i + 1;
        if event_number.is_multiple_of(config.sample_every) || event_number == records.len() {
            print_snapshot(event_number, record.sim_time, &lattice);
        }
    }

    Ok(())
}

fn species_name(index: usize) -> &'static str {
    // Mirrors `oc20_ingest.rs`'s `SPECIES_NAMES` -- kept independent since
    // this tool has no other reason to depend on that binary.
    ["O", "H", "CO", "OH", "H2O"][index]
}

fn print_snapshot(event: usize, sim_time: f64, lattice: &[u8]) {
    let mut counts = [0u64; NUM_SPECIES];
    let mut vacant = 0u64;
    for &state in lattice {
        if state == layout::VACANT {
            vacant += 1;
            continue;
        }
        if let Some(species) = SPECIES_BITS.iter().position(|&b| b == state) {
            counts[species] += 1;
        }
    }
    let counts_str: Vec<String> = counts.iter().map(|c| c.to_string()).collect();
    println!("{event},{sim_time},{vacant},{}", counts_str.join(","));
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
            eprintln!("coverage_report: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_records_skips_trailing_zero_padding() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.5f64.to_bits().to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; RECORD_SIZE]); // padding

        let records = parse_records(&bytes);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].sim_time, 1.5);
        assert_eq!(records[0].site_idx, 7);
        assert_eq!(records[0].reaction_id, 3);
    }

    #[test]
    fn parse_records_handles_empty_input() {
        assert!(parse_records(&[]).is_empty());
    }
}
