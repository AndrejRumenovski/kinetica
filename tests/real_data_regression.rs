//! End-to-end regression test against real, DFT-derived Pd(111) data.
//!
//! Every "verified against the real Pd(111) `reactions.lut`" claim in this
//! project's history (see `handoff.md`'s "What has changed" log) was, until
//! now, a manual one-off check run by whoever was doing the session's work
//! -- never something `cargo test` itself enforced. That gap is exactly
//! what let a real ~65536x performance bug in `gillespie::CompositionTable::
//! bin_ceiling` survive multiple prior "everything's green" audits: it only
//! ever surfaced from a human manually measuring wall-clock throughput.
//!
//! This test closes that gap for the specific claims a fixed, frozen
//! real-data run can check automatically: that `oc20_ingest` builds the
//! expected reaction set from real Pd(111) adsorption/barrier data, and
//! that a deterministic multi-patch `kinetica` run against that data never
//! corrupts a lattice site and always reaches the same final coverage.
//!
//! Deliberately does not hit the live Catalysis-Hub API: this project's own
//! `handoff.md` documents that source as non-deterministic run to run
//! (pagination flakiness, record-count drift), so depending on it from CI
//! would trade one real gap for a flakier one. Instead this test runs
//! against `tests/fixtures/pd111_energies.bin` /
//! `pd111_bimolecular.bin` -- committed copies of the same tiny (386- and
//! 48-byte), already-real, already-DFT-derived data
//! `data/oc20/energies_pd111*.bin` holds locally (gitignored, regenerated
//! from the live API on demand).
//!
//! Invokes the real `oc20_ingest`/`kinetica` binaries (via
//! `CARGO_BIN_EXE_*`, Cargo's standard mechanism for an integration test to
//! run a sibling binary) rather than calling library functions directly,
//! so this test exercises exactly what a user or reviewer running the
//! README's own documented commands would see.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use kinetica::layout::{
    LutKind, ReactionLut, ReactionLutBlock, ADS_CO, ADS_H, ADS_H2O, ADS_O, ADS_OH, VACANT,
};

/// A process- and call-unique path under the OS temp dir, so parallel test
/// binaries/threads never collide on the same backing file. Mirrors
/// `kinetica`'s own internal `test_support::temp_path` (not visible outside
/// the crate to an external integration test).
fn temp_path(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "kinetica_real_data_regression_{tag}_{}_{n}",
        std::process::id()
    ))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Real per-species byte histogram from a fixed, deterministic run (16x16
/// lattice, 4 patches, 2000 steps/patch) against the real Pd(111) LUT this
/// test builds below. `engine.rs`'s per-patch seed
/// (`0x5EED_0000_0000_0000 ^ patch_index`) depends only on the patch index,
/// not on wall-clock time or thread scheduling, so a fixed patch count and
/// step count reproduce this exact histogram every time -- confirmed by
/// running the pipeline twice from a clean lattice and diffing the output.
/// Any *intentional* change to the real chemistry pipeline (new species,
/// different BEP defaults, different bucket count) will need to update
/// this snapshot -- a deliberate cost, not a bug, matching this project's
/// "verify, don't assume" ethos for real-data behavior changes.
///
/// `ADS_OH` never appears in this snapshot (absent as a key, not present
/// with count 0) -- water-splitting's real ~1.0-1.2 eV barrier is orders of
/// magnitude slower than the near-barrierless adsorption channels (see
/// `handoff.md` history entry 14), so it doesn't fire within 2000 steps at
/// this lattice size.
fn expected_histogram() -> HashMap<u8, usize> {
    HashMap::from([
        (VACANT, 2),
        (ADS_O, 83),
        (ADS_H, 79),
        (ADS_CO, 69),
        (ADS_H2O, 23),
    ])
}

#[test]
fn real_pd111_pipeline_builds_expected_reactions_and_never_corrupts_the_lattice() {
    let lut_path = temp_path("reactions_lut");
    let lattice_path = temp_path("surface_lattice");
    let trajectory_path = temp_path("trajectory_bin");

    // --- Stage 1: build reactions.lut from real, frozen Pd(111) data via
    // the real oc20_ingest binary. ---
    let ingest_status = Command::new(env!("CARGO_BIN_EXE_oc20_ingest"))
        .arg("--input")
        .arg(fixture("pd111_energies.bin"))
        .arg("--bimolecular-input")
        .arg(fixture("pd111_bimolecular.bin"))
        .arg("--out")
        .arg(&lut_path)
        .arg("--metal")
        .arg("Pd")
        .arg("--facet")
        .arg("111")
        .status()
        .expect("failed to run oc20_ingest binary");
    assert!(ingest_status.success(), "oc20_ingest exited non-zero");

    // --- Stage 2: assert the exact reaction breakdown, via the library's
    // own real reader -- not by string-matching oc20_ingest's stdout. ---
    let lut = ReactionLut::open(&lut_path).expect("failed to open built reactions.lut");
    assert_eq!(
        lut.kind(),
        LutKind::OccupancyGated,
        "real Pd(111) data should always build an occupancy-gated LUT"
    );
    // `ReactionLut::len()` is a block count, not a record count (see
    // `main.rs`/`gillespie.rs`'s own `lut.len() * ReactionLutBlock::LANES`
    // usage) -- 2 cache-line blocks * 32 lanes/block = 64 total slots;
    // only 34 are real reactions (see below), the rest are zero-filled
    // tail padding from `pack_records_into_blocks`.
    assert_eq!(lut.len(), 2, "unexpected total LUT block count");
    let total_slots = lut.len() * ReactionLutBlock::LANES;
    let real_reaction_count = (0..total_slots)
        .filter(|&i| lut.rate_of(i).rate_q16 != 0)
        .count();
    assert_eq!(
        real_reaction_count, 34,
        "expected exactly 34 real reactions from the frozen Pd(111) fixtures \
         (7 monomolecular adsorption + 15 desorption + 8 homoatomic dissociative \
         bimolecular + 2 heteroatomic dissociative bimolecular + 0 recombination); \
         if this changed intentionally, update this test's expected count"
    );

    // --- Stage 3: run the real kinetica binary against that LUT, with
    // every parameter that affects determinism pinned explicitly. ---
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
        // Explicit, not the CLI's `--patches` default of
        // `rayon::current_num_threads()` (which varies by machine and
        // would break reproducibility). >1 specifically because the one
        // real multi-patch correctness bug this project ever found (the
        // boundary-migration corruption, see handoff.md history entry 16)
        // could never have been exercised by a single-patch run.
        .arg("--patches")
        .arg("4")
        .arg("--steps")
        .arg("2000")
        .status()
        .expect("failed to run kinetica binary");
    assert!(run_status.success(), "kinetica exited non-zero");

    // --- Stage 4: read the resulting lattice's raw bytes directly (same
    // headerless flat format `scripts/plot_lattice.py` reads) and assert
    // both that no site is corrupted, and an exact final coverage
    // snapshot. ---
    let sites = std::fs::read(&lattice_path).expect("failed to read surface.lattice");
    assert_eq!(sites.len(), 16 * 16);

    let known_species = [VACANT, ADS_O, ADS_H, ADS_CO, ADS_OH, ADS_H2O];
    let mut histogram: HashMap<u8, usize> = HashMap::new();
    for &byte in &sites {
        assert!(
            known_species.contains(&byte),
            "site byte {byte:#04x} does not match any known one-hot species value -- \
             a corrupted lattice site"
        );
        *histogram.entry(byte).or_insert(0) += 1;
    }

    assert_eq!(
        histogram,
        expected_histogram(),
        "final per-species coverage histogram drifted from the frozen snapshot; \
         if this is an intentional chemistry-pipeline change, update \
         `expected_histogram()` in this test"
    );

    let _ = std::fs::remove_file(&lut_path);
    let _ = std::fs::remove_file(&lattice_path);
    let _ = std::fs::remove_file(&trajectory_path);
}
