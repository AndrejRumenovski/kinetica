//! Fuzzes `config::SimConfig::parse` against arbitrary bytes.
//!
//! `SimConfig::parse` is a hand-rolled text parser -- a new untrusted-
//! input surface as soon as a future `oc20_ingest --config <path>` flag
//! (not yet wired up) reads a config file someone else authored. This
//! target's only property to hold is: for *any* input, `parse` either
//! returns `Err` or an `Ok(SimConfig)`, never a panic -- no unwrap/index/
//! slice-range assumption anywhere in the parser should be reachable
//! with adversarial bytes.
//!
//! Arbitrary bytes aren't valid UTF-8 in general, but `SimConfig::parse`
//! only accepts `&str` -- `String::from_utf8_lossy` normalizes any byte
//! sequence into one first (replacing invalid sequences rather than
//! rejecting them outright), so the fuzzer spends its time exercising the
//! actual section/key/value parsing logic instead of mostly bouncing off
//! a UTF-8 validity check before reaching it.

#![no_main]

use kinetica::config::SimConfig;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = SimConfig::parse(&text);
});
