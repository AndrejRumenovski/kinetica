//! Fuzzes `oc20e_format::read_bimolecular_records` against arbitrary
//! bytes. Same rationale as `oc20_energy_records_parse.rs` -- see that
//! target's doc comment -- for the parallel `OC20BI03` bimolecular format
//! (`--bimolecular-input`) instead of `OC20E003`.

#![no_main]

use kinetica::oc20e_format::read_bimolecular_records;
use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "kinetica-fuzz-bimolecular-{}.bin",
        std::process::id()
    ));

    let mut file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    if file.write_all(data).is_err() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    drop(file);

    let _ = std::hint::black_box(read_bimolecular_records(&path));

    let _ = std::fs::remove_file(&path);
});
