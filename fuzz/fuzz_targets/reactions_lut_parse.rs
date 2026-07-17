//! Fuzzes `layout::ReactionLut::open` against arbitrary bytes.
//!
//! `ReactionLut::open` maps a `reactions.lut` file and reinterprets its
//! bytes as `[ReactionLutBlock]` via an `unsafe` pointer cast (see its own
//! safety comment in `src/layout.rs`) -- soundness there depends entirely
//! on the length/alignment checks that run *before* the cast, not on the
//! file actually being a well-formed LUT `oc20_ingest`/`kinetica
//! --generate-lut` produced. This target's only property to hold is: for
//! *any* input, `open` either returns `Err` or a `ReactionLut` that
//! `as_slice()`/`rate_of()` can be read from without panicking or
//! triggering UB (which ASan/UBSan, wired up via cargo-fuzz's default
//! sanitizer, would catch even though safe Rust can't observe it
//! directly) -- never a panic or a crash on malformed/truncated/
//! arbitrary-length/misaligned-looking input.

#![no_main]

use kinetica::layout::ReactionLut;
use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // `ReactionLut::open` takes a path, not a byte slice -- write the
    // fuzzer's bytes to a real (tmpfs-backed) file each iteration so the
    // exact same mmap-then-validate code path production callers use is
    // what gets exercised, rather than a parallel in-memory-only version
    // of the parser that could drift from the real one.
    let mut path = std::env::temp_dir();
    path.push(format!("kinetica-fuzz-lut-{}.bin", std::process::id()));

    let mut file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    if file.write_all(data).is_err() {
        return;
    }
    drop(file);

    if let Ok(lut) = ReactionLut::open(&path) {
        // A successful open must make every block/record actually
        // readable -- walk all of it, the same way every real caller
        // (gillespie.rs, occupancy.rs) does.
        for block in lut.as_slice() {
            std::hint::black_box(block);
        }
        for id in 0..lut.len() * kinetica::layout::ReactionLutBlock::LANES {
            std::hint::black_box(lut.rate_of(id));
        }
    }

    let _ = std::fs::remove_file(&path);
});
