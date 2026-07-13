//! `cattrace` library surface, shared by the `cattrace` simulation binary
//! and auxiliary tools (e.g. `oc20_ingest`) that build a `reactions.lut`
//! from real Open Catalyst Project data instead of the synthetic demo
//! generator in `main.rs`.

pub mod engine;
pub mod gillespie;
pub mod layout;
