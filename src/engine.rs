//! Spatial domain decomposition, work-stealing concurrency, and the
//! asynchronous `io_uring` trajectory logger.
//!
//! This module wires the memory layout (`layout.rs`) and the O(1) reaction
//! sampler (`gillespie.rs`) into the macro-scale execution pipeline: the
//! lattice is cut into row-band patches, each patch gets its own
//! independent local Gillespie loop running on a `rayon` work-stealing
//! thread, patches exchange boundary-crossing state over lock-free
//! `crossbeam_channel` rings, and every fired reaction is drained into a
//! double-buffered `io_uring` writer so compute threads never block on
//! storage latency.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use crossbeam_channel::{Receiver, Sender};

use crate::gillespie::GillespieDomain;
use crate::layout::{ReactionLut, SiteLattice};

// ------------------------------------------------------------------------
// Cross-patch boundary migration
// ------------------------------------------------------------------------

/// One occupancy update that needs to be mirrored into a neighboring
/// patch's boundary row, because the reaction that produced it touched a
/// site on the edge of this patch's domain.
#[derive(Clone, Copy, Debug)]
pub struct MigrationEvent {
    /// Column within the row (shared coordinate space across all patches,
    /// since every patch spans the lattice's full width).
    pub col: usize,
    /// New occupancy byte to write at that column in the neighbor's
    /// adjacent boundary row.
    pub state: u8,
}

/// The two inbound/outbound `crossbeam_channel` links a single patch uses
/// to exchange boundary state with its vertical neighbors. Endpoints are
/// `None` at the top and bottom edges of the whole lattice, where there is
/// no neighbor to talk to.
struct BoundaryLinks {
    send_up: Option<Sender<MigrationEvent>>,
    send_down: Option<Sender<MigrationEvent>>,
    recv_from_above: Option<Receiver<MigrationEvent>>,
    recv_from_below: Option<Receiver<MigrationEvent>>,
}

fn apply_migration(data: &mut [u8], width: usize, row_local: usize, ev: MigrationEvent) {
    let idx = row_local * width + ev.col;
    if idx < data.len() {
        data[idx] = ev.state;
    }
}

// ------------------------------------------------------------------------
// Per-patch local Gillespie loop
// ------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_patch(
    band: (usize, usize),
    width: usize,
    data: &mut [u8],
    lut: &ReactionLut,
    links: BoundaryLinks,
    steps: u64,
    seed: u64,
    trajectory_tx: &Sender<TrajectoryRecord>,
) {
    let (y0, y1) = band;
    let rows_in_band = y1 - y0;
    let mut domain = GillespieDomain::new(lut, seed);

    for _ in 0..steps {
        // Drain any migrations that arrived from neighbors since the last
        // step before this patch acts on its own boundary rows, so a
        // reaction this step sees the neighbor's latest state rather than
        // a stale one. `try_iter` never blocks -- an empty channel just
        // yields nothing, keeping this patch's thread free-running.
        if let Some(rx) = &links.recv_from_above {
            for ev in rx.try_iter() {
                apply_migration(data, width, 0, ev);
            }
        }
        if let Some(rx) = &links.recv_from_below {
            for ev in rx.try_iter() {
                apply_migration(data, width, rows_in_band - 1, ev);
            }
        }

        let Some((reaction_id, tau)) = domain.step(lut) else {
            break; // domain gone fully quiescent
        };

        // Map the fired reaction onto a concrete lattice site within this
        // patch. A full transition-state model would derive the site from
        // the reaction's neighborhood template; this engine samples
        // uniformly over the patch's own sites, which is sufficient to
        // exercise the occupancy update and boundary-migration paths.
        let site_idx = domain.rng.next_u32_below(data.len() as u32) as usize;
        let (_, _, transition) = lut.rate_of(reaction_id as usize);

        // `transition` packs `(reactant_mask << 4) | product_mask`: clear
        // the reactant's bits and OR in the product's, in place.
        let reactant_mask = transition >> 4;
        let product_mask = transition & 0x0F;
        data[site_idx] = (data[site_idx] & !reactant_mask) | product_mask;

        let _ = trajectory_tx.send(TrajectoryRecord {
            sim_time_bits: domain.sim_time.to_bits(),
            site_idx: (y0 * width + site_idx) as u32,
            reaction_id,
        });
        let _ = tau; // waiting time already folded into domain.sim_time

        // Mirror the update to a neighbor if it landed on this patch's
        // shared edge row.
        let row_local = site_idx / width;
        let col = site_idx % width;
        if row_local == 0 {
            if let Some(tx) = &links.send_up {
                let _ = tx.send(MigrationEvent {
                    col,
                    state: data[site_idx],
                });
            }
        }
        if row_local == rows_in_band - 1 {
            if let Some(tx) = &links.send_down {
                let _ = tx.send(MigrationEvent {
                    col,
                    state: data[site_idx],
                });
            }
        }
    }
}

// ------------------------------------------------------------------------
// Work-stealing fan-out
// ------------------------------------------------------------------------

/// Partition `lattice` into up to `num_patches` grid-aligned row bands and
/// run `steps_per_patch` local Gillespie iterations on each, concurrently,
/// via a `rayon` work-stealing scope. Reaction events stream out to an
/// `io_uring`-backed trajectory writer running on its own thread the whole
/// time, so no compute thread ever waits on disk I/O.
pub fn run_simulation(
    lattice: &mut SiteLattice,
    lut: &ReactionLut,
    num_patches: usize,
    steps_per_patch: u64,
    trajectory_path: &Path,
) -> io::Result<()> {
    let (trajectory_tx, trajectory_rx) = crossbeam_channel::unbounded::<TrajectoryRecord>();

    let writer_path = trajectory_path.to_path_buf();
    let writer_handle =
        std::thread::spawn(move || run_trajectory_writer(trajectory_rx, &writer_path));

    let width = lattice.width;
    let bands = lattice.split_row_bands_mut(num_patches);
    let n = bands.len();

    // Wire up one bounded channel pair per boundary between vertically
    // adjacent patches, up front -- these are the lock-free atomic rings
    // patches use to cross domain boundaries; nothing here is a global
    // lock, and every channel is only ever touched by the exactly two
    // patches on either side of its boundary.
    let mut send_up: Vec<Option<Sender<MigrationEvent>>> = (0..n).map(|_| None).collect();
    let mut send_down: Vec<Option<Sender<MigrationEvent>>> = (0..n).map(|_| None).collect();
    let mut recv_from_above: Vec<Option<Receiver<MigrationEvent>>> = (0..n).map(|_| None).collect();
    let mut recv_from_below: Vec<Option<Receiver<MigrationEvent>>> = (0..n).map(|_| None).collect();

    for i in 0..n.saturating_sub(1) {
        let (tx_down, rx_down) = crossbeam_channel::bounded::<MigrationEvent>(1024);
        send_down[i] = Some(tx_down);
        recv_from_above[i + 1] = Some(rx_down);

        let (tx_up, rx_up) = crossbeam_channel::bounded::<MigrationEvent>(1024);
        send_up[i + 1] = Some(tx_up);
        recv_from_below[i] = Some(rx_up);
    }

    // Each patch's local Gillespie loop is spawned as an independent
    // `rayon` task; the work-stealing scheduler fans these out across
    // however many CPU cores are available and steals idle patches' work
    // if some finish early, with no global execution lock -- the only
    // cross-thread coordination is the bounded channels above and the
    // unbounded trajectory channel.
    rayon::scope(|scope| {
        for (i, (y0, y1, data)) in bands.into_iter().enumerate() {
            let links = BoundaryLinks {
                send_up: send_up[i].take(),
                send_down: send_down[i].take(),
                recv_from_above: recv_from_above[i].take(),
                recv_from_below: recv_from_below[i].take(),
            };
            let tx = trajectory_tx.clone();
            // Distinct, deterministic per-patch seed so re-running the
            // same decomposition reproduces the same trajectory.
            let seed = 0x5EED_0000_0000_0000u64 ^ (i as u64);

            scope.spawn(move |_| {
                run_patch((y0, y1), width, data, lut, links, steps_per_patch, seed, &tx);
            });
        }
    });

    // Dropping our own sender lets the writer thread's `for record in rx`
    // loop see the channel close once every patch's clone is also
    // dropped, so it can flush and return instead of blocking forever.
    drop(trajectory_tx);
    writer_handle
        .join()
        .expect("trajectory writer thread panicked")?;

    Ok(())
}

// ------------------------------------------------------------------------
// Double-buffered io_uring trajectory writer
// ------------------------------------------------------------------------

/// One page's worth of trajectory data: `O_DIRECT` requires every write to
/// be aligned to the storage device's logical block size (4096 bytes on
/// essentially every modern NVMe part).
pub const PAGE_SIZE: usize = 4096;

/// A fixed-size, 4096-byte-aligned page. `#[repr(align(4096))]` is what
/// actually guarantees the alignment `O_DIRECT` demands -- a bare
/// `[u8; PAGE_SIZE]` field has no address-alignment guarantee beyond `u8`'s
/// natural alignment of 1, which is not sufficient on its own.
#[repr(align(4096))]
struct Page([u8; PAGE_SIZE]);

impl Page {
    const fn zeroed() -> Self {
        Page([0u8; PAGE_SIZE])
    }
}

/// A single fired-reaction record as logged to `trajectory.bin`: the
/// simulation time (as raw `f64` bits, to keep this `repr(C)` and free of
/// float-specific padding surprises), the global lattice site index, and
/// which reaction fired there.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct TrajectoryRecord {
    sim_time_bits: u64,
    site_idx: u32,
    reaction_id: u32,
}

const RECORD_SIZE: usize = std::mem::size_of::<TrajectoryRecord>();
const RECORDS_PER_PAGE: usize = PAGE_SIZE / RECORD_SIZE;
const _: () = assert!(PAGE_SIZE.is_multiple_of(RECORD_SIZE));

fn write_record(page: &mut [u8; PAGE_SIZE], slot: usize, record: &TrajectoryRecord) {
    let start = slot * RECORD_SIZE;
    page[start..start + 8].copy_from_slice(&record.sim_time_bits.to_le_bytes());
    page[start + 8..start + 12].copy_from_slice(&record.site_idx.to_le_bytes());
    page[start + 12..start + 16].copy_from_slice(&record.reaction_id.to_le_bytes());
}

/// Linux's `O_DIRECT` flag value on x86_64/aarch64 (see
/// `include/uapi/asm-generic/fcntl.h`); a few legacy architectures (alpha,
/// sparc, mips, parisc) define a different bit pattern, but none of them
/// are targets for this crate. Hardcoded rather than pulling in `libc` as
/// an extra dependency purely for one flag constant.
const O_DIRECT: i32 = 0o40000;

fn open_direct_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(O_DIRECT)
        .open(path)
}

/// A single dedicated page-writing thread: exclusively owns nothing but
/// its `Rio` handle and file handle, and simply blocks on each
/// `(offset, page)` it receives until that page has landed on the NVMe
/// device via Direct I/O, then returns the drained page to the shared free
/// pool for reuse.
///
/// Because this loop only ever submits a write and waits on it within the
/// same iteration -- never holding a `Completion` across iterations -- the
/// borrow checker has no trouble with it. The actual double-buffering
/// overlap comes from *two* of these running concurrently on their own OS
/// threads (see `run_trajectory_writer`): while one is blocked in `wait()`
/// on page A, the other is free to be submitting or waiting on page B, and
/// neither blocks any compute (Gillespie) thread.
fn run_page_writer(
    ring: rio::Rio,
    file: std::sync::Arc<File>,
    rx: Receiver<(u64, Box<Page>)>,
    free_tx: Sender<Box<Page>>,
) -> io::Result<()> {
    for (offset, filled) in rx {
        let completion = ring.write_at(&*file, &filled.0, offset);
        completion.wait()?;
        let _ = free_tx.send(filled);
    }
    Ok(())
}

/// Own the trajectory file and a small fixed pool of page buffers for the
/// entire run, draining fired-reaction records from `rx` into whichever
/// page is currently being filled, and handing full pages off to one of
/// two dedicated writer threads for zero-copy Direct I/O.
///
/// The double-buffering asked for here is implemented as ownership
/// transfer through a 3-slot `Box<Page>` pool (one page always "being
/// filled" here, up to two more "in flight" to the two writer threads)
/// rather than a `rio::Completion` borrow persisted across loop
/// iterations of a single function: the compiler can't prove a
/// runtime-computed flag alternates strictly enough to let two named
/// buffers take turns being mutated and borrowed, but it *can* trivially
/// see that a `Box<Page>` moved out over a channel is no longer aliased by
/// anything here. Real overlap -- the actual point of double buffering --
/// comes from the two writer threads each blocking on their own `wait()`
/// concurrently, on separate OS threads, never stalling this filling loop
/// or any compute thread.
fn run_trajectory_writer(rx: Receiver<TrajectoryRecord>, path: &Path) -> io::Result<()> {
    let ring = rio::new()?;
    let file = std::sync::Arc::new(open_direct_file(path)?);

    // Three heap-allocated, 4096-byte-aligned pages, allocated once up
    // front and never reallocated afterward -- they only ever move (by
    // ownership, through channels) between "being filled here", "in
    // flight to the SSD on one of the two writer threads", and "sitting
    // free in the pool".
    let (free_tx, free_rx) = crossbeam_channel::bounded::<Box<Page>>(3);
    for _ in 0..3 {
        free_tx
            .send(Box::new(Page::zeroed()))
            .expect("pool channel has capacity for all 3 initial pages");
    }

    let (tx_a, rx_a) = crossbeam_channel::bounded::<(u64, Box<Page>)>(1);
    let (tx_b, rx_b) = crossbeam_channel::bounded::<(u64, Box<Page>)>(1);

    let writer_a = std::thread::spawn({
        let ring = ring.clone();
        let file = std::sync::Arc::clone(&file);
        let free_tx = free_tx.clone();
        move || run_page_writer(ring, file, rx_a, free_tx)
    });
    let writer_b = std::thread::spawn({
        let ring = ring.clone();
        let file = std::sync::Arc::clone(&file);
        let free_tx = free_tx.clone();
        move || run_page_writer(ring, file, rx_b, free_tx)
    });

    let mut cursor = 0usize;
    let mut offset = 0u64;
    let mut page = free_rx.recv().expect("pool primed with 3 pages above");
    // Flips every page-fill cycle to alternate which writer thread ("the
    // alternate page") the next full page is dispatched to.
    let mut dispatch_to_a = true;

    for record in rx {
        write_record(&mut page.0, cursor, &record);
        cursor += 1;

        if cursor == RECORDS_PER_PAGE {
            // Grab a fresh page from the pool *before* handing the filled
            // one off -- this only blocks if both writer threads are
            // simultaneously behind, which the 3-page pool is sized to
            // avoid in steady state.
            let next_page = free_rx
                .recv()
                .expect("3-page pool covers 1 filling + 2 in-flight writers");
            let filled = std::mem::replace(&mut page, next_page);

            let dest = if dispatch_to_a { &tx_a } else { &tx_b };
            let _ = dest.send((offset, filled));

            offset += PAGE_SIZE as u64;
            dispatch_to_a = !dispatch_to_a;
            cursor = 0;
        }
    }

    // Closing both dispatch channels lets each writer thread's `for (..)
    // in rx` loop end once it drains whatever was already in flight to
    // it, so `join` below returns instead of blocking forever.
    drop(tx_a);
    drop(tx_b);
    writer_a.join().expect("page writer thread A panicked")?;
    writer_b.join().expect("page writer thread B panicked")?;

    if cursor > 0 {
        // No more incoming work to overlap with -- flush the trailing
        // partial page directly and synchronously.
        let completion = ring.write_at(&*file, &page.0, offset);
        completion.wait()?;
    }

    Ok(())
}
