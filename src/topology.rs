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

/// Upper bound on neighbors any supported topology can produce for one
/// site. Every function here returns a fixed-size
/// `[Option<usize>; MAX_NEIGHBORS]` (no heap allocation on this hot path)
/// with unused trailing slots `None`.
pub const MAX_NEIGHBORS: usize = 4;

#[inline]
fn row_col(idx: usize, width: usize) -> (usize, usize) {
    (idx / width, idx % width)
}

/// Every grid-adjacent neighbor of `idx` that exists within `width` x
/// `rows` -- left, right, up, down, in that fixed order (existing
/// neighbors packed at the front of the array; the rest `None`). Used
/// wherever *all* adjacent sites of one site need visiting (bimolecular
/// partner search, incremental counter updates after a single site
/// changes).
#[inline]
pub fn all_neighbors(idx: usize, width: usize, rows: usize) -> [Option<usize>; MAX_NEIGHBORS] {
    let (row, col) = row_col(idx, width);
    let mut out = [None; MAX_NEIGHBORS];
    let mut n = 0;
    if col > 0 {
        out[n] = Some(idx - 1);
        n += 1;
    }
    if col + 1 < width {
        out[n] = Some(idx + 1);
        n += 1;
    }
    if row > 0 {
        out[n] = Some(idx - width);
        n += 1;
    }
    if row + 1 < rows {
        out[n] = Some(idx + width);
    }
    out
}

/// The canonical *half* of `all_neighbors` -- right and down only -- for a
/// full-grid scan that must visit every unordered adjacent pair exactly
/// once (not twice, once from each side). Checking all four from every
/// site during a full scan would double-count each edge.
#[inline]
pub fn forward_neighbors(idx: usize, width: usize, rows: usize) -> [Option<usize>; MAX_NEIGHBORS] {
    let (row, col) = row_col(idx, width);
    let mut out = [None; MAX_NEIGHBORS];
    let mut n = 0;
    if col + 1 < width {
        out[n] = Some(idx + 1);
        n += 1;
    }
    if row + 1 < rows {
        out[n] = Some(idx + width);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_neighbors_interior_site_has_all_four() {
        let width = 5;
        let rows = 5;
        let idx = 2 * width + 2; // (row=2, col=2), fully interior
        let neighbors: Vec<usize> = all_neighbors(idx, width, rows).into_iter().flatten().collect();
        assert_eq!(neighbors.len(), 4);
        assert!(neighbors.contains(&(idx - 1)));
        assert!(neighbors.contains(&(idx + 1)));
        assert!(neighbors.contains(&(idx - width)));
        assert!(neighbors.contains(&(idx + width)));
    }

    #[test]
    fn all_neighbors_corner_site_has_two() {
        let width = 5;
        let rows = 5;
        let neighbors: Vec<usize> = all_neighbors(0, width, rows).into_iter().flatten().collect();
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
                let back: Vec<usize> = all_neighbors(neighbor, width, rows).into_iter().flatten().collect();
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
            full_count += all_neighbors(idx, width, rows).into_iter().flatten().count();
        }
        assert_eq!(full_count, edges.len() * 2);
    }

    #[test]
    fn single_row_has_no_vertical_neighbors() {
        let width = 4;
        let rows = 1;
        let neighbors: Vec<usize> = all_neighbors(1, width, rows).into_iter().flatten().collect();
        assert_eq!(neighbors, vec![0, 2]);
    }
}
