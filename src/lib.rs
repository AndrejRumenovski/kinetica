//! `kinetica` library surface, shared by the `kinetica` simulation binary
//! and auxiliary tools (e.g. `oc20_ingest`) that build a `reactions.lut`
//! from real Open Catalyst Project data instead of the synthetic demo
//! generator in `main.rs`.

pub mod engine;
pub mod gillespie;
pub mod layout;
pub mod occupancy;

/// Shared by `layout`'s and `gillespie`'s test modules, both of which need
/// a real on-disk file to back a `SiteLattice`/`ReactionLut` mmap (neither
/// type can be exercised over an in-memory buffer). Not visible outside
/// this crate's own test builds.
#[cfg(test)]
pub(crate) mod test_support {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A process- and call-unique path under the OS temp dir, so parallel
    /// `cargo test` threads never collide on the same backing file.
    pub fn temp_path(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("kinetica_test_{tag}_{}_{n}", std::process::id()))
    }
}
