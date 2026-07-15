//! Occupancy-gated event selection for `KMCOCC01` reaction LUTs.
//!
//! `gillespie.rs`'s composition-rejection sampler treats every reaction as
//! an independent, always-available channel with a fixed propensity,
//! entirely decoupled from the lattice's live occupancy -- a reaction can
//! be selected and applied to a site that never actually held its
//! reactant. This module exists to close that gap for real-data
//! (`oc20_ingest`-built) LUTs: propensity is `rate_q16` scaled by how many
//! lattice sites *currently* match a reaction's required starting state,
//! and a reaction can only ever fire where that state genuinely holds.
//!
//! The reaction catalogue `oc20_ingest` builds for this path is small
//! (~25 elementary-step templates -- 4 quantile buckets per species per
//! direction, plus a handful of un-bucketed real bimolecular barriers; see
//! its own docs for why), so this module doesn't need `gillespie.rs`'s
//! 32-bin magnitude-class machinery at all: a simple weighted linear scan
//! over the template list is effectively O(1) for any M this small, and
//! -- unlike the static scheme's build-once bin index -- needs to reflect
//! a *changing* total propensity as the lattice evolves, which a
//! precomputed bin index isn't designed to do.
//!
//! Each site's "which quantile bucket does this site's chemistry belong
//! to, for a given species" identity is a deterministic hash of its own
//! index -- never stored, so there is no per-site memory cost beyond the
//! base occupancy byte `layout::SiteLattice` already carries.
//!
//! Site/pair selection uses bounded rejection sampling (pick a uniformly
//! random candidate, retry on a miss) rather than an exact O(1) live
//! free-list per bucket. This is simpler and correct -- a fired reaction
//! never touches the wrong site, because every candidate is verified
//! against the live lattice state before being accepted -- at a known,
//! honest cost: if a bucket's live count is extremely sparse relative to
//! the whole patch, rejection can need many attempts before (or a full
//! deterministic scan after) finding a match. The natural next
//! optimization, if that ever matters in practice, is explicit per-bucket
//! free-lists (a "sparse set" with O(1) removal) -- not built this pass.

use crate::gillespie::Rng;
use crate::layout::{self, ReactionRecord, ADS_CO, ADS_H, ADS_O, SPECIES_BITS};

/// Quantile buckets `oc20_ingest`'s `bucket_by_quantile` splits each
/// species' real DFT samples into. Must agree with the ingest tool: the
/// bucket a site is assigned to here has to line up with which bucket's
/// rate a template represents there. Monomolecular `ReactionRecord`s
/// carry their bucket index in `bin_id` (0..`BUCKETS_PER_SPECIES`);
/// bimolecular ones don't use `bin_id` at all (see `live_count`).
pub const BUCKETS_PER_SPECIES: usize = 4;

/// Cap on rejection-sampling attempts before falling back to a
/// deterministic full scan. Large enough that the common case (a bucket
/// with a non-trivial fraction of the patch matching) essentially never
/// hits it, small enough that the fallback -- guaranteed to succeed, since
/// callers only search when the corresponding live count is already known
/// to be positive -- kicks in quickly for genuinely sparse cases rather
/// than burning many wasted draws first.
const MAX_REJECTION_ATTEMPTS: u32 = 256;

fn species_index(species_bit: u8) -> Option<usize> {
    SPECIES_BITS.iter().position(|&b| b == species_bit)
}

/// Deterministic, storage-free hash mapping one lattice site's global
/// index to a quantile bucket for one species. A single-shot mix (not a
/// stateful stream) reusing `gillespie::Rng`'s SplitMix64 constants for
/// consistency with the rest of the codebase: the same `(site_idx,
/// species_bit, seed)` always produces the same bucket, so a site's
/// bucket assignment never needs to be persisted anywhere.
#[inline]
pub fn site_bucket(site_idx: usize, species_bit: u8, seed: u64) -> usize {
    let mut z = (site_idx as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((species_bit as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(seed);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z % BUCKETS_PER_SPECIES as u64) as usize
}

/// True iff the unordered pair `{a, b}` equals `{x, y}` -- used to count
/// an adjacent-site pair regardless of which of the two sites is "first"
/// in whatever order they happen to be visited.
#[inline]
fn pair_matches(a: u8, b: u8, x: u8, y: u8) -> bool {
    (a == x && b == y) || (a == y && b == x)
}

/// Live, per-patch counts the occupancy-gated selector needs to weight
/// propensities and find matching sites. Built once from a full scan when
/// a patch starts (`new`), then kept in sync in O(1) amortized time per
/// event (`on_occupancy_change`) rather than ever rescanned.
pub struct OccupancyCounters {
    /// `vacant_count[species][bucket]`: how many sites assigned to
    /// `(species, bucket)` are currently vacant -- the live count an
    /// adsorption template for that species/bucket fires against.
    vacant_count: [[u32; BUCKETS_PER_SPECIES]; 3],
    /// `occupied_count[species][bucket]`: how many sites assigned to
    /// `(species, bucket)` are currently occupied *by that species* --
    /// the live count a desorption template fires against.
    occupied_count: [[u32; BUCKETS_PER_SPECIES]; 3],
    /// Live count of adjacent (O*, CO*) site pairs -- CO-oxidation's
    /// propensity. Not bucketed: `oc20_ingest` keeps bimolecular real
    /// barriers as individually-real, un-averaged records (there are only
    /// a handful of them), so there is one shared pool of matching pairs
    /// for the CO-oxidation templates to draw from, not per-bucket ones.
    co_ox_pairs: u32,
    /// Live count of adjacent (H*, H*) site pairs -- H2-recombination's
    /// propensity.
    h2_pairs: u32,
}

impl OccupancyCounters {
    /// One O(N) pass over `patch_data`'s initial state, seeding every
    /// counter from scratch. Pair counting only looks at each site's
    /// `topology::forward_neighbors` (not all up to six) specifically so a
    /// full scan counts every unordered adjacent pair exactly once --
    /// checking every neighbor from every site would double-count each
    /// pair, once from each side.
    pub fn new(patch_data: &[u8], width: usize, seed: u64) -> Self {
        let mut counters = OccupancyCounters {
            vacant_count: [[0; BUCKETS_PER_SPECIES]; 3],
            occupied_count: [[0; BUCKETS_PER_SPECIES]; 3],
            co_ox_pairs: 0,
            h2_pairs: 0,
        };
        if width == 0 {
            return counters;
        }
        let rows = patch_data.len() / width;

        for site_idx in 0..patch_data.len() {
            let state = patch_data[site_idx];
            if state == layout::VACANT {
                for (species, &bit) in SPECIES_BITS.iter().enumerate() {
                    let bucket = site_bucket(site_idx, bit, seed);
                    counters.vacant_count[species][bucket] += 1;
                }
            } else if let Some(species) = species_index(state) {
                let bucket = site_bucket(site_idx, state, seed);
                counters.occupied_count[species][bucket] += 1;
            }
            // Any other bit pattern is a corrupted/unknown site state
            // (see `layout::OCCUPANCY_MASK`'s doc comment) -- not counted
            // toward any template's live count, so it simply can never be
            // selected as a reactant; the engine never writes such a state.

            for neighbor_idx in crate::topology::forward_neighbors(site_idx, width, rows)
                .into_iter()
                .flatten()
            {
                counters.add_pair(state, patch_data[neighbor_idx]);
            }
        }

        counters
    }

    fn add_pair(&mut self, a: u8, b: u8) {
        if pair_matches(a, b, ADS_O, ADS_CO) {
            self.co_ox_pairs += 1;
        }
        if a == ADS_H && b == ADS_H {
            self.h2_pairs += 1;
        }
    }

    fn remove_pair(&mut self, a: u8, b: u8) {
        if pair_matches(a, b, ADS_O, ADS_CO) {
            self.co_ox_pairs = self.co_ox_pairs.saturating_sub(1);
        }
        if a == ADS_H && b == ADS_H {
            self.h2_pairs = self.h2_pairs.saturating_sub(1);
        }
    }

    /// Incrementally update every counter affected by one site's
    /// occupancy changing from `old_state` to `new_state`. Must be called
    /// with `patch_data` already reflecting `new_state` at `site_idx` --
    /// neighbor lookups read `patch_data` directly, so for a bimolecular
    /// event touching two sites, call this once per site immediately
    /// after that site's own mutation (not batched), so each call sees
    /// accurate neighbor state; the two sequential single-site deltas sum
    /// to the correct net change regardless of call order.
    #[allow(clippy::too_many_arguments)]
    pub fn on_occupancy_change(
        &mut self,
        patch_data: &[u8],
        site_idx: usize,
        width: usize,
        rows_in_band: usize,
        old_state: u8,
        new_state: u8,
        seed: u64,
    ) {
        if old_state == new_state {
            return;
        }

        if old_state == layout::VACANT {
            for (species, &bit) in SPECIES_BITS.iter().enumerate() {
                let bucket = site_bucket(site_idx, bit, seed);
                self.vacant_count[species][bucket] =
                    self.vacant_count[species][bucket].saturating_sub(1);
            }
        } else if let Some(species) = species_index(old_state) {
            let bucket = site_bucket(site_idx, old_state, seed);
            self.occupied_count[species][bucket] =
                self.occupied_count[species][bucket].saturating_sub(1);
        }

        if new_state == layout::VACANT {
            for (species, &bit) in SPECIES_BITS.iter().enumerate() {
                let bucket = site_bucket(site_idx, bit, seed);
                self.vacant_count[species][bucket] += 1;
            }
        } else if let Some(species) = species_index(new_state) {
            let bucket = site_bucket(site_idx, new_state, seed);
            self.occupied_count[species][bucket] += 1;
        }

        // Re-evaluate every pair touching this site (up to all of its
        // neighbors this time -- unlike `new`'s full-scan pass, this only
        // ever looks at one site's incident edges, so there's no
        // double-counting risk to avoid).
        for neighbor_idx in crate::topology::all_neighbors(site_idx, width, rows_in_band)
            .into_iter()
            .flatten()
        {
            let neighbor_state = patch_data[neighbor_idx];
            self.remove_pair(old_state, neighbor_state);
            self.add_pair(new_state, neighbor_state);
        }
    }

    /// How many lattice sites currently satisfy `template`'s reactant
    /// pattern -- the live weight its propensity is scaled by.
    fn live_count(&self, template: &ReactionRecord) -> u64 {
        if template.is_bimolecular {
            let reactant_a = template.transition_a >> 4;
            if reactant_a == ADS_O || reactant_a == ADS_CO {
                self.co_ox_pairs as u64
            } else {
                self.h2_pairs as u64
            }
        } else {
            let reactant_mask = template.transition_a >> 4;
            let product_mask = template.transition_a & 0x0F;
            let bucket = template.bin_id as usize;
            if bucket >= BUCKETS_PER_SPECIES {
                return 0;
            }
            if reactant_mask == 0 {
                // Adsorption: reactant is VACANT, species comes from the
                // product side.
                species_index(product_mask)
                    .map(|s| self.vacant_count[s][bucket] as u64)
                    .unwrap_or(0)
            } else {
                species_index(reactant_mask)
                    .map(|s| self.occupied_count[s][bucket] as u64)
                    .unwrap_or(0)
            }
        }
    }

    /// Sum of every template's live weight (`rate_q16 * live_count`) --
    /// the total propensity the exponential waiting-time draw needs.
    /// Zero means the domain has gone fully quiescent: no template has a
    /// site currently matching its reactant pattern.
    pub fn total_propensity(&self, templates: &[ReactionRecord]) -> f64 {
        templates
            .iter()
            .map(|t| t.rate_q16 as f64 * self.live_count(t) as f64)
            .sum()
    }

    /// Select which reaction fires next (weighted by `rate_q16 *
    /// live_count`) and which site(s) it fires on, without applying
    /// anything -- callers are expected to apply the transition(s) via
    /// the existing `layout::apply_transition`/trajectory-logging path,
    /// then call `on_occupancy_change` per touched site. Returns `None`
    /// when every template's live weight is zero (domain quiescent, same
    /// semantics as `gillespie::GillespieDomain::step`).
    pub fn select_event(
        &self,
        templates: &[ReactionRecord],
        patch_data: &[u8],
        width: usize,
        rows_in_band: usize,
        rng: &mut Rng,
        seed: u64,
    ) -> Option<(u32, usize, Option<usize>)> {
        let weights: Vec<f64> = templates
            .iter()
            .map(|t| t.rate_q16 as f64 * self.live_count(t) as f64)
            .collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return None;
        }

        let draw = (rng.next_u64() as f64 / (u64::MAX as f64 + 1.0)) * total;
        let mut acc = 0.0;
        let mut chosen = templates.len() - 1;
        for (i, &w) in weights.iter().enumerate() {
            acc += w;
            if draw < acc {
                chosen = i;
                break;
            }
        }
        let template = &templates[chosen];

        if template.is_bimolecular {
            let (site_a, site_b) =
                self.find_bimolecular_pair(template, patch_data, width, rows_in_band, rng)?;
            Some((chosen as u32, site_a, Some(site_b)))
        } else {
            let site = self.find_monomolecular_site(template, patch_data, seed, rng)?;
            Some((chosen as u32, site, None))
        }
    }

    fn find_monomolecular_site(
        &self,
        template: &ReactionRecord,
        patch_data: &[u8],
        seed: u64,
        rng: &mut Rng,
    ) -> Option<usize> {
        let reactant_mask = template.transition_a >> 4;
        let product_mask = template.transition_a & 0x0F;
        let (expected_state, species_bit) = if reactant_mask == 0 {
            (layout::VACANT, product_mask)
        } else {
            (reactant_mask, reactant_mask)
        };
        let bucket = template.bin_id as usize;
        let n = patch_data.len();
        if n == 0 {
            return None;
        }

        for _ in 0..MAX_REJECTION_ATTEMPTS {
            let candidate = rng.next_u32_below(n as u32) as usize;
            if patch_data[candidate] == expected_state
                && site_bucket(candidate, species_bit, seed) == bucket
            {
                return Some(candidate);
            }
        }
        // Deterministic fallback: guaranteed to find a match, since
        // `select_event` only searches for a template whose `live_count`
        // (computed from these exact same counters) was already positive.
        (0..n).find(|&site| {
            patch_data[site] == expected_state && site_bucket(site, species_bit, seed) == bucket
        })
    }

    fn find_bimolecular_pair(
        &self,
        template: &ReactionRecord,
        patch_data: &[u8],
        width: usize,
        rows_in_band: usize,
        rng: &mut Rng,
    ) -> Option<(usize, usize)> {
        let species_a_bit = template.transition_a >> 4;
        let species_b_bit = template.transition_b >> 4;
        let n = patch_data.len();
        if n == 0 {
            return None;
        }

        for _ in 0..MAX_REJECTION_ATTEMPTS {
            let candidate = rng.next_u32_below(n as u32) as usize;
            if patch_data[candidate] != species_a_bit {
                continue;
            }
            if let Some(partner) =
                neighbor_with_state(patch_data, candidate, width, rows_in_band, species_b_bit)
            {
                return Some((candidate, partner));
            }
        }
        // Deterministic fallback, same guarantee as the monomolecular
        // case: `co_ox_pairs`/`h2_pairs` being positive means a matching
        // pair genuinely exists somewhere in the patch.
        (0..n).find_map(|candidate| {
            if patch_data[candidate] != species_a_bit {
                return None;
            }
            neighbor_with_state(patch_data, candidate, width, rows_in_band, species_b_bit)
                .map(|partner| (candidate, partner))
        })
    }
}

/// First (if any) of `site_idx`'s up to six grid neighbors currently in
/// `state`. Deterministic rather than randomized among multiple matches --
/// the caller already randomized *which* first site it tried, so picking
/// the first matching neighbor deterministically doesn't introduce
/// meaningful bias for this purpose.
fn neighbor_with_state(
    patch_data: &[u8],
    site_idx: usize,
    width: usize,
    rows_in_band: usize,
    state: u8,
) -> Option<usize> {
    crate::topology::all_neighbors(site_idx, width, rows_in_band)
        .into_iter()
        .flatten()
        .find(|&idx| patch_data[idx] == state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{ADS_CO, ADS_H, ADS_O, VACANT};

    fn rng() -> Rng {
        Rng::seeded(42)
    }

    #[test]
    fn site_bucket_is_deterministic() {
        for site in [0usize, 1, 1000, 999_999] {
            for species in SPECIES_BITS {
                assert_eq!(
                    site_bucket(site, species, 7),
                    site_bucket(site, species, 7)
                );
            }
        }
    }

    #[test]
    fn site_bucket_spreads_roughly_uniformly() {
        let mut counts = [0u32; BUCKETS_PER_SPECIES];
        let n = 40_000;
        for site in 0..n {
            counts[site_bucket(site, ADS_O, 123)] += 1;
        }
        let expected = n as f64 / BUCKETS_PER_SPECIES as f64;
        for &c in &counts {
            let ratio = c as f64 / expected;
            assert!(
                (0.8..1.2).contains(&ratio),
                "bucket count {c} too far from expected {expected}"
            );
        }
    }

    #[test]
    fn site_bucket_differs_by_species_for_the_same_site() {
        // Not a strict requirement, but the three species' hashes should
        // not be trivially identical for every site (would collapse all
        // three species onto the same bucket assignment).
        let mismatches = (0..1000)
            .filter(|&s| site_bucket(s, ADS_O, 1) != site_bucket(s, ADS_H, 1))
            .count();
        assert!(mismatches > 100);
    }

    fn brute_force_counters(patch_data: &[u8], width: usize, seed: u64) -> OccupancyCounters {
        OccupancyCounters::new(patch_data, width, seed)
    }

    #[test]
    fn new_counts_vacant_and_occupied_sites_correctly() {
        let width = 4;
        let data = vec![VACANT, ADS_O, ADS_H, ADS_CO, VACANT, VACANT, ADS_O, ADS_O];
        let counters = OccupancyCounters::new(&data, width, 99);

        let total_vacant: u32 = (0..3).map(|s| counters.vacant_count[s][0..].iter().sum::<u32>()).sum();
        // Every vacant site contributes to all 3 species' vacant counts.
        let vacant_sites = data.iter().filter(|&&s| s == VACANT).count() as u32;
        assert_eq!(total_vacant, vacant_sites * 3);

        let o_occupied: u32 = counters.occupied_count[0].iter().sum();
        assert_eq!(o_occupied, data.iter().filter(|&&s| s == ADS_O).count() as u32);
        let h_occupied: u32 = counters.occupied_count[1].iter().sum();
        assert_eq!(h_occupied, data.iter().filter(|&&s| s == ADS_H).count() as u32);
        let co_occupied: u32 = counters.occupied_count[2].iter().sum();
        assert_eq!(co_occupied, data.iter().filter(|&&s| s == ADS_CO).count() as u32);
    }

    #[test]
    fn new_counts_adjacent_pairs_exactly_once() {
        let width = 2;
        // Grid:
        // O  CO
        // H  H
        let data = vec![ADS_O, ADS_CO, ADS_H, ADS_H];
        let counters = OccupancyCounters::new(&data, width, 5);
        assert_eq!(counters.co_ox_pairs, 1); // (0,1) horizontal
        assert_eq!(counters.h2_pairs, 1); // (2,3) horizontal
    }

    #[test]
    fn on_occupancy_change_matches_brute_force_after_random_events() {
        let width = 6usize;
        let height = 6usize;
        let seed = 4242u64;
        let mut data = vec![VACANT; width * height];
        let mut rng = Rng::seeded(1);
        let states = [VACANT, ADS_O, ADS_H, ADS_CO];
        for s in data.iter_mut() {
            *s = states[rng.next_u32_below(4) as usize];
        }

        let mut counters = OccupancyCounters::new(&data, width, seed);

        for _ in 0..500 {
            let site = rng.next_u32_below((width * height) as u32) as usize;
            let old = data[site];
            let new = states[rng.next_u32_below(4) as usize];
            data[site] = new;
            counters.on_occupancy_change(&data, site, width, height, old, new, seed);
        }

        let brute = brute_force_counters(&data, width, seed);
        assert_eq!(counters.vacant_count, brute.vacant_count);
        assert_eq!(counters.occupied_count, brute.occupied_count);
        assert_eq!(counters.co_ox_pairs, brute.co_ox_pairs);
        assert_eq!(counters.h2_pairs, brute.h2_pairs);
    }

    fn ads_template(species_bit: u8, bucket: u8, rate: u32) -> ReactionRecord {
        ReactionRecord {
            rate_q16: rate,
            bin_id: bucket,
            transition_a: species_bit, // reactant=VACANT(0), product=species
            transition_b: 0,
            is_bimolecular: false,
        }
    }

    fn des_template(species_bit: u8, bucket: u8, rate: u32) -> ReactionRecord {
        ReactionRecord {
            rate_q16: rate,
            bin_id: bucket,
            transition_a: species_bit << 4, // reactant=species, product=VACANT
            transition_b: 0,
            is_bimolecular: false,
        }
    }

    #[test]
    fn select_event_never_fires_desorption_on_a_site_lacking_the_reactant() {
        let width = 8;
        let height = 8;
        let seed = 17u64;
        // Only ONE O*-occupied site in the whole patch; everything else
        // vacant. If site selection weren't occupancy-gated, a desorption
        // template with a huge rate would very likely pick the wrong site.
        let mut data = vec![VACANT; width * height];
        let o_site = 37usize;
        data[o_site] = ADS_O;
        let counters = OccupancyCounters::new(&data, width, seed);

        let bucket = site_bucket(o_site, ADS_O, seed) as u8;
        let templates = vec![des_template(ADS_O, bucket, u32::MAX)];

        let mut rng = rng();
        for _ in 0..200 {
            if let Some((_, site, site_b)) =
                counters.select_event(&templates, &data, width, height, &mut rng, seed)
            {
                assert_eq!(site, o_site);
                assert_eq!(site_b, None);
            }
        }
    }

    #[test]
    fn select_event_never_fires_adsorption_on_an_occupied_site() {
        let width = 8;
        let height = 8;
        let seed = 91u64;
        // Only ONE vacant site; everything else occupied by CO*.
        let mut data = vec![ADS_CO; width * height];
        let vacant_site = 5usize;
        data[vacant_site] = VACANT;
        let counters = OccupancyCounters::new(&data, width, seed);

        let bucket = site_bucket(vacant_site, ADS_O, seed) as u8;
        let templates = vec![ads_template(ADS_O, bucket, u32::MAX)];

        let mut rng = rng();
        for _ in 0..200 {
            if let Some((_, site, site_b)) =
                counters.select_event(&templates, &data, width, height, &mut rng, seed)
            {
                assert_eq!(site, vacant_site);
                assert_eq!(site_b, None);
            }
        }
    }

    #[test]
    fn select_event_bimolecular_only_fires_on_a_genuinely_adjacent_pair() {
        let width = 6;
        let height = 6;
        let seed = 3u64;
        let mut data = vec![VACANT; width * height];
        // One real O*-CO* adjacent pair at (10, 11); a lone, non-adjacent
        // O* elsewhere that must never be chosen as a bimolecular partner
        // site since it has no CO* neighbor.
        data[10] = ADS_O;
        data[11] = ADS_CO;
        data[30] = ADS_O;
        let counters = OccupancyCounters::new(&data, width, seed);
        assert_eq!(counters.co_ox_pairs, 1);

        let template = ReactionRecord {
            rate_q16: u32::MAX,
            bin_id: 0,
            transition_a: ADS_O << 4,
            transition_b: ADS_CO << 4,
            is_bimolecular: true,
        };
        let templates = vec![template];

        let mut rng = rng();
        for _ in 0..200 {
            if let Some((_, site_a, site_b)) =
                counters.select_event(&templates, &data, width, height, &mut rng, seed)
            {
                let site_b = site_b.expect("bimolecular event must return a second site");
                let pair = (data[site_a], data[site_b]);
                assert!(
                    pair == (ADS_O, ADS_CO) || pair == (ADS_CO, ADS_O),
                    "fired on non-matching pair {pair:?}"
                );
                let neighbors: Vec<usize> = crate::topology::all_neighbors(site_a, width, height)
                    .into_iter()
                    .flatten()
                    .collect();
                assert!(
                    neighbors.contains(&site_b),
                    "bimolecular sites must be topology-adjacent"
                );
            }
        }
    }

    #[test]
    fn select_event_returns_none_when_every_template_has_zero_live_count() {
        let width = 4;
        let data = vec![VACANT; 16];
        let counters = OccupancyCounters::new(&data, width, 1);
        // Desorption templates for a species with nothing on the lattice.
        let templates = vec![
            des_template(ADS_O, 0, 1000),
            des_template(ADS_H, 1, 1000),
            des_template(ADS_CO, 2, 1000),
        ];
        let mut rng = rng();
        assert_eq!(
            counters.select_event(&templates, &data, width, 4, &mut rng, 1),
            None
        );
    }
}
