//! Fuzzes `oc20e_format::read_energy_records` against arbitrary bytes.
//!
//! `--input` (an `OC20E003` file) comes from a separate Python extraction
//! script (`scripts/extract_energies.py`/`extract_catalysis_hub.py`), not
//! from this crate -- so it's untrusted the same way `reactions.lut` is
//! (see `reactions_lut_parse.rs`'s own doc comment): a killed extraction
//! run, a stale format-version file, or plain corruption can all produce
//! bytes this reader has to survive without panicking. An error-handling
//! audit found (and fixed) exactly this: a file whose `count` field
//! claimed more records than actually followed panicked with an
//! out-of-bounds slice index instead of returning `Err`. This target's
//! only property to hold is: for *any* input, `read_energy_records`
//! either returns `Err` or a `Vec<EnergyRecord>` -- never a panic.

#![no_main]

use kinetica::oc20e_format::read_energy_records;
use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // `read_energy_records` takes a path, not a byte slice -- write the
    // fuzzer's bytes to a real file each iteration so the exact same
    // read-then-parse code path production callers use is what gets
    // exercised, rather than a parallel in-memory-only version of the
    // parser that could drift from the real one.
    let mut path = std::env::temp_dir();
    path.push(format!("kinetica-fuzz-energy-{}.bin", std::process::id()));

    let mut file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    if file.write_all(data).is_err() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    drop(file);

    let _ = std::hint::black_box(read_energy_records(&path));

    let _ = std::fs::remove_file(&path);
});
