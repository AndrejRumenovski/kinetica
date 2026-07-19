//! End-to-end regression test proving runtime species-count generality:
//! the same frozen, real Pd(111) fixtures `tests/real_data_regression.rs`
//! uses build a *correct, smaller* `reactions.lut` when driven by a
//! narrower config (`configs/pd111_ohco_subset.conf`, O/H/CO only,
//! dropping OH/H2O), with no source change and no new committed binary
//! data -- the whole point of the config-driven generalization arc this
//! test closes out (Phase 7 of 7).
//!
//! This is the scenario `tests/real_data_regression.rs`'s own 5-species
//! run structurally cannot exercise: `configs/pd111.conf` always declares
//! every species the fixture data references, so a smaller active species
//! count -- and, in particular, a real bimolecular record naming a
//! species this run's `--config` doesn't declare -- was never actually
//! tested before this file existed. Running this for real (not just
//! reasoning about it) is exactly what caught a genuine bug during this
//! phase's own development: `oc20_ingest.rs`'s printed reaction-count
//! breakdown derived its "recombination bimolecular" figure by
//! subtracting `dissociative_bimolecular_count` from
//! `bimolecular_records.len()`, silently attributing records skipped for
//! an undeclared species (this fixture's own two real water-splitting
//! records, both naming OH) to "recombination" instead of reflecting that
//! they built zero reactions -- fixed by counting both categories
//! directly in the same loop that pushes to the LUT, so the two counts
//! can no longer drift apart. See `oc20_ingest.rs`'s own comment at that
//! loop for the full account.
//!
//! Mirrors `real_data_regression.rs`'s structure closely (temp paths,
//! fixture helper, real-binary invocation via `CARGO_BIN_EXE_*`) rather
//! than sharing code with it -- each file under `tests/` is its own
//! independent integration-test binary, and duplicating a handful of
//! small helpers here keeps this file readable in isolation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use kinetica::layout::{LutKind, ReactionLut, ReactionLutBlock, ADS_CO, ADS_H, ADS_O, VACANT};

fn temp_path(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kinetica_subset_config_regression_{tag}_{}_{n}",
        std::process::id()
    ))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn subset_config() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("configs/pd111_ohco_subset.conf")
}

/// Real per-species byte histogram from the same fixed, deterministic run
/// shape `real_data_regression.rs` uses (16x16 lattice, 4 patches, 2000
/// steps/patch) -- against the *subset* LUT this test builds below, not
/// the 5-species one. Confirmed reproducible by running the pipeline
/// twice from a clean lattice and diffing the raw output bytes.
fn expected_histogram() -> HashMap<u8, usize> {
    HashMap::from([(VACANT, 24), (ADS_O, 67), (ADS_CO, 138), (ADS_H, 27)])
}

#[test]
fn subset_config_builds_a_smaller_reaction_set_and_never_corrupts_the_lattice() {
    let lut_path = temp_path("reactions_lut");
    let lattice_path = temp_path("surface_lattice");
    let trajectory_path = temp_path("trajectory_bin");

    // --- Stage 1: build reactions.lut from the same frozen Pd(111) data
    // real_data_regression.rs uses, but driven by the narrower O/H/CO-only
    // config instead of the full 5-species one. The fixture's own real
    // bimolecular records both name OH (species_b = 3), which this
    // 3-species config doesn't declare -- oc20_ingest.rs is expected to
    // skip them safely rather than build anything from them or panic. ---
    let ingest_status = Command::new(env!("CARGO_BIN_EXE_oc20_ingest"))
        .arg("--input")
        .arg(fixture("pd111_energies.bin"))
        .arg("--bimolecular-input")
        .arg(fixture("pd111_bimolecular.bin"))
        .arg("--config")
        .arg(subset_config())
        .arg("--out")
        .arg(&lut_path)
        .status()
        .expect("failed to run oc20_ingest binary");
    assert!(ingest_status.success(), "oc20_ingest exited non-zero");

    // --- Stage 2: assert the LUT's self-described species table has
    // exactly 3 species (not 5) -- the header round-trips a runtime
    // species count different from every other config this project ships,
    // proving the header encoding itself is genuinely species-count
    // generic, not just tested at N=5. ---
    let lut = ReactionLut::open(&lut_path).expect("failed to open built reactions.lut");
    assert_eq!(lut.kind(), LutKind::OccupancyGated);
    let species = lut.species();
    assert_eq!(
        species.len(),
        3,
        "subset config should stamp exactly 3 species"
    );
    assert_eq!(species.index_of(ADS_O), Some(0));
    assert_eq!(species.name(0), Some("O"));
    assert_eq!(species.index_of(ADS_H), Some(1));
    assert_eq!(species.name(1), Some("H"));
    assert_eq!(species.index_of(ADS_CO), Some(2));
    assert_eq!(species.name(2), Some("CO"));

    // --- Stage 3: assert the exact, smaller reaction breakdown -- proves
    // the two real bimolecular (water-splitting) records were dropped
    // cleanly (0 reactions from them), not silently misattributed or
    // corrupting the count of what *did* build. 22 = 3 monomolecular CO
    // adsorption + 11 desorption (4 O-bucket + 4 H-bucket + 3 CO-bucket)
    // + 8 homoatomic dissociative-adsorption bimolecular (O2, H2) + 0 from
    // the skipped OH-referencing records. ---
    assert_eq!(lut.len(), 1, "unexpected total LUT block count");
    let total_slots = lut.len() * ReactionLutBlock::LANES;
    let real_reaction_count = (0..total_slots)
        .filter(|&i| lut.rate_of(i).rate_q16 != 0)
        .count();
    assert_eq!(
        real_reaction_count, 22,
        "expected exactly 22 real reactions from the frozen Pd(111) fixtures under the \
         O/H/CO-only subset config (3 monomolecular adsorption + 11 desorption + \
         8 homoatomic dissociative-adsorption bimolecular + 0 from the two real \
         bimolecular records this config's species set doesn't cover); if this changed \
         intentionally, update this test's expected count"
    );

    // --- Stage 4: run the real kinetica binary against that smaller LUT,
    // same deterministic parameters as real_data_regression.rs. ---
    let run_status = Command::new(env!("CARGO_BIN_EXE_kinetica"))
        .arg("--lattice-path")
        .arg(&lattice_path)
        .arg("--lattice-width")
        .arg("16")
        .arg("--lattice-height")
        .arg("16")
        .arg("--lut-path")
        .arg(&lut_path)
        .arg("--trajectory-path")
        .arg(&trajectory_path)
        .arg("--patches")
        .arg("4")
        .arg("--steps")
        .arg("2000")
        .status()
        .expect("failed to run kinetica binary");
    assert!(run_status.success(), "kinetica exited non-zero");

    // --- Stage 5: no corrupted site, and an exact final coverage
    // snapshot over only the 3 species this config actually tracks. ---
    let sites = std::fs::read(&lattice_path).expect("failed to read surface.lattice");
    assert_eq!(sites.len(), 16 * 16);

    let known_species = [VACANT, ADS_O, ADS_H, ADS_CO];
    let mut histogram: HashMap<u8, usize> = HashMap::new();
    for &byte in &sites {
        assert!(
            known_species.contains(&byte),
            "site byte {byte:#04x} does not match any known one-hot species value for this \
             subset config -- a corrupted lattice site (or a species this config never \
             declared somehow got written)"
        );
        *histogram.entry(byte).or_insert(0) += 1;
    }

    assert_eq!(
        histogram,
        expected_histogram(),
        "final per-species coverage histogram drifted from the frozen snapshot; if this is \
         an intentional chemistry-pipeline change, update `expected_histogram()` in this test"
    );

    let _ = std::fs::remove_file(&lut_path);
    let _ = std::fs::remove_file(&lattice_path);
    let _ = std::fs::remove_file(&trajectory_path);
}
