//! Grid neighbor topology for the catalyst lattice: what "adjacent site"
//! means, centralized so `occupancy.rs` (pair counting, incremental
//! counters, bimolecular partner search) and `engine.rs` (same-patch
//! bimolecular partner pick) can't drift out of sync on the definition,
//! and so a change to the lattice's physical geometry is one place to
//! make, not four.
//!
//! All indices are into a flat row-major byte matrix
//! (`layout::SiteLattice`, or one patch's own slice of it):
//! `idx = row * width + col`. `rows` is the number of rows *in whatever
//! slice `idx` lives in* -- a whole lattice's `height`, or one patch's
//! `rows_in_band` -- since a patch's own local coordinate space is what
//! actually bounds a neighbor search within it.
//!
//! **Hexagonal (fcc(111)) topology.** The reaction data this engine now
//! targets (see `oc20_ingest --metal Pd --facet 111`) is a real fcc(111)
//! close-packed surface -- six equidistant nearest neighbors per site, not
//! four. Rather than introduce a second storage layout, this stays on the
//! same flat row-major mmap and reinterprets it as an "offset-coordinate"
//! (odd-r) hex grid: even rows and odd rows are horizontally offset by
//! half a hex-width from each other, so each row zig-zags into true
//! close-packed alignment with its neighbors while every site is still
//! addressable as `row * width + col`. Six neighbor directions per site
//! (W, E, and two each of "up"/"down") whose exact column offset depends
//! on the current row's parity -- see `EVEN_ROW_DELTAS`/`ODD_ROW_DELTAS`.

/// Upper bound on neighbors any supported topology can produce for one
/// site -- six, for the hexagonal topology below. Every function here
/// returns a fixed-size `[Option<usize>; MAX_NEIGHBORS]` (no heap
/// allocation on this hot path) with unused trailing slots `None`.
pub const MAX_NEIGHBORS: usize = 6;

#[inline]
fn row_col(idx: usize, width: usize) -> (usize, usize) {
    (idx / width, idx % width)
}

/// Bounds-check a `(row, col)` computed via signed deltas and convert it
/// back to a flat index, or `None` if it fell outside `width` x `rows`.
#[inline]
fn try_idx(row: isize, col: isize, width: usize, rows: usize) -> Option<usize> {
    if row < 0 || col < 0 {
        return None;
    }
    let (row, col) = (row as usize, col as usize);
    if row >= rows || col >= width {
        return None;
    }
    Some(row * width + col)
}

/// Offset-coordinate ("odd-r") hex deltas as `(dRow, dCol)`, in a fixed
/// `[W, E, NW, NE, SW, SE]` order, for a site on an *even*-indexed row.
/// Even rows are the "reference" columns; odd rows are shifted a half
/// step to the right, which is why the diagonal (NW/NE/SW/SE) column
/// offsets differ between `EVEN_ROW_DELTAS` and `ODD_ROW_DELTAS` even
/// though both represent the same real, symmetric hex adjacency.
const EVEN_ROW_DELTAS: [(isize, isize); MAX_NEIGHBORS] = [
    (0, -1),
    (0, 1), // W, E
    (-1, -1),
    (-1, 0), // NW, NE
    (1, -1),
    (1, 0), // SW, SE
];

/// Same as `EVEN_ROW_DELTAS`, for a site on an *odd*-indexed row.
const ODD_ROW_DELTAS: [(isize, isize); MAX_NEIGHBORS] = [
    (0, -1),
    (0, 1), // W, E
    (-1, 0),
    (-1, 1), // NW, NE
    (1, 0),
    (1, 1), // SW, SE
];

/// Shared walk: bounds-check each of `deltas` against `idx`'s `(row, col)`
/// and pack the surviving neighbors at the front of a fixed-size array --
/// the common tail of both `all_neighbors` (given all six deltas) and
/// `forward_neighbors` (given the canonical half). `deltas.len()` is at
/// most `MAX_NEIGHBORS`, so `out` never overflows.
#[inline]
fn gather(
    idx: usize,
    width: usize,
    rows: usize,
    deltas: &[(isize, isize)],
) -> [Option<usize>; MAX_NEIGHBORS] {
    let (row, col) = row_col(idx, width);

    let mut out = [None; MAX_NEIGHBORS];
    let mut n = 0;
    for &(dr, dc) in deltas {
        if let Some(neighbor) = try_idx(row as isize + dr, col as isize + dc, width, rows) {
            out[n] = Some(neighbor);
            n += 1;
        }
    }
    out
}

/// Every grid-adjacent neighbor of `idx` that exists within `width` x
/// `rows`, in the fixed `[W, E, NW, NE, SW, SE]` order (existing
/// neighbors packed at the front of the array; the rest `None`). Used
/// wherever *all* adjacent sites of one site need visiting (bimolecular
/// partner search, incremental counter updates after a single site
/// changes).
#[inline]
pub fn all_neighbors(idx: usize, width: usize, rows: usize) -> [Option<usize>; MAX_NEIGHBORS] {
    let row = idx / width;
    let deltas = if row.is_multiple_of(2) {
        &EVEN_ROW_DELTAS
    } else {
        &ODD_ROW_DELTAS
    };
    gather(idx, width, rows, deltas)
}

/// The canonical *half* of `all_neighbors` -- E, SW, SE -- for a
/// full-grid scan that must visit every unordered adjacent pair exactly
/// once (not twice, once from each side). Checking all six from every
/// site during a full scan would double-count each edge; this subset is
/// exactly the complement (W, NW, NE) that every other site's own E/SW/SE
/// already reaches from the other direction -- see the module tests for
/// the reciprocity/edge-count invariant this relies on.
#[inline]
pub fn forward_neighbors(idx: usize, width: usize, rows: usize) -> [Option<usize>; MAX_NEIGHBORS] {
    let row = idx / width;
    let deltas: [(isize, isize); 3] = if row.is_multiple_of(2) {
        [(0, 1), (1, -1), (1, 0)] // E, SW, SE
    } else {
        [(0, 1), (1, 0), (1, 1)] // E, SW, SE
    };
    gather(idx, width, rows, &deltas)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_neighbors_interior_even_row_site_matches_expected_hex_set() {
        // width=6, rows=6, site (row=2, col=2) -- interior, even row.
        let width = 6;
        let rows = 6;
        let idx = 2 * width + 2;
        let neighbors: std::collections::BTreeSet<usize> = all_neighbors(idx, width, rows)
            .into_iter()
            .flatten()
            .collect();
        // W=13, E=15, NW=7, NE=8, SW=19, SE=20
        assert_eq!(neighbors, [13, 15, 7, 8, 19, 20].into_iter().collect());
    }

    #[test]
    fn all_neighbors_interior_odd_row_site_matches_expected_hex_set() {
        // width=6, rows=6, site (row=3, col=2) -- interior, odd row.
        let width = 6;
        let rows = 6;
        let idx = 3 * width + 2;
        let neighbors: std::collections::BTreeSet<usize> = all_neighbors(idx, width, rows)
            .into_iter()
            .flatten()
            .collect();
        // W=19, E=21, NW=14, NE=15, SW=26, SE=27
        assert_eq!(neighbors, [19, 21, 14, 15, 26, 27].into_iter().collect());
    }

    #[test]
    fn all_neighbors_corner_site_has_two() {
        let width = 5;
        let rows = 5;
        let neighbors: Vec<usize> = all_neighbors(0, width, rows)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&width));
    }

    #[test]
    fn all_neighbors_reciprocal_across_full_grid() {
        let width = 6;
        let rows = 6;
        for idx in 0..width * rows {
            for neighbor in all_neighbors(idx, width, rows).into_iter().flatten() {
                let back: Vec<usize> = all_neighbors(neighbor, width, rows)
                    .into_iter()
                    .flatten()
                    .collect();
                assert!(back.contains(&idx), "{idx} -> {neighbor} not reciprocal");
            }
        }
    }

    #[test]
    fn forward_neighbors_full_scan_visits_every_edge_exactly_once() {
        let width = 6;
        let rows = 6;
        let mut edges = std::collections::HashSet::new();
        for idx in 0..width * rows {
            for neighbor in forward_neighbors(idx, width, rows).into_iter().flatten() {
                let edge = (idx.min(neighbor), idx.max(neighbor));
                assert!(edges.insert(edge), "edge {edge:?} visited more than once");
            }
        }
        // Every edge counted via `forward_neighbors` must also show up from
        // both sides in `all_neighbors` -- i.e. the forward-only scan found
        // exactly half of what a full (both-direction) adjacency count
        // would find.
        let mut full_count = 0usize;
        for idx in 0..width * rows {
            full_count += all_neighbors(idx, width, rows)
                .into_iter()
                .flatten()
                .count();
        }
        assert_eq!(full_count, edges.len() * 2);
    }

    #[test]
    fn single_row_has_no_vertical_neighbors() {
        let width = 4;
        let rows = 1;
        let neighbors: Vec<usize> = all_neighbors(1, width, rows)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(neighbors, vec![0, 2]);
    }

    #[test]
    fn interior_site_has_exactly_six_neighbors() {
        let width = 8;
        let rows = 8;
        let idx = 4 * width + 4;
        assert_eq!(
            all_neighbors(idx, width, rows)
                .into_iter()
                .flatten()
                .count(),
            6
        );
    }
}
