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
//! **Site selection for bucketed (monomolecular) reactions is an exact,
//! guaranteed-O(1) live free-list**, not rejection sampling: `BucketedSet`
//! is a "sparse set" per `(species, bucket)` -- a dense `Vec<u32>` of the
//! matching site indices plus a `site_idx -> position`-in-dense lookup --
//! so insert/remove/random-pick are all O(1), with no retry loop and no
//! pathological case when a bucket is sparse. This replaced an earlier
//! bounded-rejection design (pick a uniformly random candidate, retry on a
//! miss, fall back to a deterministic scan after enough misses): correct,
//! but its cost wasn't bounded -- a bucket with a tiny live fraction of the
//! patch could burn many wasted draws before landing a hit. The free-list
//! trades that uncertainty for O(site count) extra memory (the `position`
//! lookup, sized to the patch once per species/vacant-or-occupied) -- a
//! deliberate space-for-a-worst-case-time-guarantee trade, not free.
//!
//! **Bimolecular pair selection still uses bounded rejection** (see
//! `find_bimolecular_pair`, `MAX_REJECTION_ATTEMPTS`). `oc20_ingest` keeps
//! real bimolecular barriers as a handful of un-bucketed records (no
//! `bin_id`), so there is no per-bucket structure for a pair free-list to
//! key off in the first place -- building one would mean inventing a
//! separate edge-indexed sparse set (keyed by, say, `site_idx * 3 +
//! forward-neighbor-direction`), a materially different data structure
//! from the one this pass adds. Left as a future item of its own if
//! bimolecular pair search ever actually shows up as a bottleneck -- it
//! hasn't (real bimolecular record counts are small, and pairs are rarely
//! anywhere near as sparse as the worst-case single-bucket scenario the
//! free-list above was built for).

use crate::gillespie::Rng;
use crate::layout::{self, ReactionRecord, ADS_CO, ADS_H, ADS_O, ADS_OH, NUM_SPECIES, SPECIES_BITS};

/// Quantile buckets `oc20_ingest`'s `bucket_by_quantile` splits each
/// species' real DFT samples into. Must agree with the ingest tool: the
/// bucket a site is assigned to here has to line up with which bucket's
/// rate a template represents there. Monomolecular `ReactionRecord`s
/// carry their bucket index in `bin_id` (0..`BUCKETS_PER_SPECIES`);
/// bimolecular ones don't use `bin_id` at all (see `live_count`).
pub const BUCKETS_PER_SPECIES: usize = 4;

/// Cap on rejection-sampling attempts before falling back to a
/// deterministic full scan, for **bimolecular pair search only**
/// (`find_bimolecular_pair`) -- monomolecular site selection no longer
/// rejection-samples at all, see `BucketedSet`. Large enough that the
/// common case (a pair pool with a non-trivial fraction of the patch
/// matching) essentially never hits it, small enough that the fallback --
/// guaranteed to succeed, since callers only search when the corresponding
/// live count is already known to be positive -- kicks in quickly for
/// genuinely sparse cases rather than burning many wasted draws first.
const MAX_REJECTION_ATTEMPTS: u32 = 256;

/// A "sparse set" of lattice site indices, giving O(1) insert, O(1)
/// swap-remove, and O(1) uniform-random pick over whichever sites are
/// currently members -- the free-list `OccupancyCounters` uses in place of
/// rejection sampling for bucketed (monomolecular) reactant lookups.
/// `BUCKETS_PER_SPECIES` independent dense lists, one per quantile bucket,
/// sharing a single `position` lookup since a site is a member of at most
/// one bucket's list at a time for a given species+vacant-or-occupied
/// category (which bucket that is doesn't need to be stored -- it's always
/// recoverable from `site_bucket`, so callers pass it in rather than this
/// type tracking it redundantly).
#[derive(Debug)]
struct BucketedSet {
    dense: [Vec<u32>; BUCKETS_PER_SPECIES],
    /// `position[site_idx]`: that site's index within whichever bucket's
    /// `dense` list currently holds it. Only meaningful while the site is
    /// actually a member of *some* bucket in this set -- callers never
    /// consult it otherwise.
    position: Vec<u32>,
}

impl BucketedSet {
    fn new(site_count: usize) -> Self {
        BucketedSet {
            dense: std::array::from_fn(|_| Vec::new()),
            position: vec![u32::MAX; site_count],
        }
    }

    fn insert(&mut self, site_idx: usize, bucket: usize) {
        let pos = self.dense[bucket].len() as u32;
        self.dense[bucket].push(site_idx as u32);
        self.position[site_idx] = pos;
    }

    /// Swap-remove `site_idx` from `bucket`'s dense list: move the last
    /// element into the removed slot (O(1), no shifting) and fix up that
    /// moved element's recorded position.
    fn remove(&mut self, site_idx: usize, bucket: usize) {
        let pos = self.position[site_idx] as usize;
        let last = self.dense[bucket].len() - 1;
        if pos != last {
            let moved = self.dense[bucket][last];
            self.dense[bucket][pos] = moved;
            self.position[moved as usize] = pos as u32;
        }
        self.dense[bucket].pop();
    }

    fn len(&self, bucket: usize) -> usize {
        self.dense[bucket].len()
    }

    /// Uniformly random member of `bucket`'s live set, or `None` if empty.
    /// The whole point of this type: O(1), no retry loop, ever.
    fn pick(&self, bucket: usize, rng: &mut Rng) -> Option<usize> {
        let d = &self.dense[bucket];
        if d.is_empty() {
            return None;
        }
        Some(d[rng.next_u32_below(d.len() as u32) as usize] as usize)
    }
}

fn species_index(species_bit: u8) -> Option<usize> {
    SPECIES_BITS.iter().position(|&b| b == species_bit)
}

/// Relative gas-phase partial-pressure multipliers, indexed by species the
/// same way `SPECIES_BITS`/`occupancy::OccupancyCounters` are -- a runtime
/// simulator parameter (`kinetica --pressure-o2 --pressure-h2 --pressure-co
/// --pressure-h2o`), not baked into `reactions.lut`, so changing the
/// feed-gas composition never requires rebuilding it. Without this, two
/// runs meant to represent different partial pressures produced identical
/// adsorption kinetics -- the same rate-constant table applied regardless
/// of how much of each gas was actually being fed in.
///
/// Sized `NUM_SPECIES`, but index 3 (OH) is always ignored: OH only ever
/// forms via the heteroatomic dissociative-adsorption path (water
/// splitting), which `pressure_factor` short-circuits to `1.0` *before*
/// ever indexing into this array -- see its doc comment for why a species
/// that only ever appears as one side of a two-species gas reaction can't
/// correctly be gated by looking up its own slot. H2O (index 4) *is* a
/// real, used slot: `star + H2O(g) -> H2Ostar` is an ordinary single-gas
/// monomolecular adsorption, same shape as O2/H2/CO, so it gates exactly
/// like they do.
///
/// Desorption and bimolecular *recombination* templates (CO-oxidation,
/// H2-recombination) are untouched -- pressure only gates a reaction that
/// consumes a gas-phase molecule, which includes both monomolecular
/// adsorption *and* genuine two-site dissociative adsorption
/// (`2* + O2(g)/H2(g) -> 2 species*`, see `oc20_ingest`), but not a
/// reaction that's already conditioned on something already sitting on
/// the surface. `ones()` reproduces exactly the pre-pressure-coupling
/// propensities (every multiplier 1.0).
#[derive(Clone, Copy, Debug)]
pub struct Pressures {
    pub values: [f64; NUM_SPECIES],
}

impl Pressures {
    pub const fn ones() -> Self {
        Pressures { values: [1.0; NUM_SPECIES] }
    }
}

/// `1.0` for every template except one that consumes a gas-phase
/// molecule -- monomolecular adsorption, or genuine *homoatomic* two-site
/// dissociative adsorption (`2* + O2(g)/H2(g) -> 2 species*`) -- where
/// it's that species' relative partial pressure. Kept separate from
/// `live_count` since a pressure multiplier isn't a physical site count.
///
/// A bimolecular *recombination* template (CO-oxidation, H2-recombination,
/// or the reverse/associative-desorption half of water splitting) consumes
/// an occupied pair, not a gas-phase reactant, so it's untouched, same as
/// monomolecular desorption. A *heteroatomic* dissociative-adsorption
/// template -- currently only water splitting, `2* + H2O(g) -> H* +
/// OH*` -- genuinely does consume a gas-phase molecule (H2O), and
/// `Pressures` *does* now carry a real H2O slot (index 4, gating H2O*'s
/// own ordinary monomolecular adsorption below), but water splitting still
/// can't use it: neither site's product species alone identifies "this
/// reaction's gas is H2O" (site A produces H*, which has its own pressure
/// slot, but gating on it would incorrectly tie water-splitting propensity
/// to H2 pressure instead of H2O pressure; there's no single product
/// species here that *means* "H2O consumed" the way H2O* itself would).
/// Rather than get that wrong silently, heteroatomic dissociative
/// adsorption stays pressure-neutral -- a documented simplification, not a
/// bug (see README).
fn pressure_factor(template: &ReactionRecord, pressures: &Pressures) -> f64 {
    let reactant_mask = (template.transition_a >> 8) as u8;
    if reactant_mask != 0 {
        return 1.0; // desorption or recombination: no gas-phase reactant
    }
    let product_mask = (template.transition_a & 0xFF) as u8;
    if template.is_bimolecular && product_mask != (template.transition_b & 0xFF) as u8 {
        return 1.0; // heteroatomic dissociative adsorption -- see doc comment
    }
    species_index(product_mask)
        .and_then(|s| pressures.values.get(s))
        .copied()
        .unwrap_or(1.0)
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
    /// `vacant_sets[species]`: free-list of currently-vacant sites,
    /// bucketed by that species' `site_bucket` hash -- the live pool an
    /// adsorption template for that species/bucket fires against. A single
    /// vacant site is a member of all `NUM_SPECIES` of these simultaneously
    /// (once per species, generally in a different bucket per species,
    /// since any species could in principle adsorb there).
    vacant_sets: [BucketedSet; NUM_SPECIES],
    /// `occupied_sets[species]`: free-list of sites currently occupied *by
    /// that species*, bucketed the same way -- the live pool a desorption
    /// template fires against. A site is a member of at most one of these
    /// at a time (whichever species currently occupies it, if any).
    occupied_sets: [BucketedSet; NUM_SPECIES],
    /// Live count of adjacent (O*, CO*) site pairs -- CO-oxidation's
    /// propensity. Not bucketed: `oc20_ingest` keeps bimolecular real
    /// barriers as individually-real, un-averaged records (there are only
    /// a handful of them), so there is one shared pool of matching pairs
    /// for the CO-oxidation templates to draw from, not per-bucket ones.
    co_ox_pairs: u32,
    /// Live count of adjacent (H*, H*) site pairs -- H2-recombination's
    /// propensity.
    h2_pairs: u32,
    /// Live count of adjacent (H*, OH*) site pairs -- the associative-
    /// desorption (reverse) half of water splitting's propensity
    /// (`H* + OH* -> H2O(g) + 2*`). Mirrors `co_ox_pairs`/`h2_pairs`.
    h_oh_pairs: u32,
    /// Live count of adjacent (VACANT, VACANT) site pairs -- the shared
    /// pool every genuine two-site dissociative-adsorption template
    /// (`2* + O2(g) -> 2 O*`, `2* + H2(g) -> 2 H*`, `2* + H2O(g) -> H* +
    /// OH*`) draws its propensity from. Deliberately one shared counter,
    /// not per-species/per-reaction: any adjacent vacant pair is a real
    /// landing site for *any* of these, so their templates compete for
    /// the same physical pool, same as monomolecular adsorption templates
    /// already compete for `vacant_count`'s single-site pool -- only each
    /// template's own `rate_q16` (and, via `pressure_factor`, its own
    /// species' relative pressure where applicable) differentiates them.
    vacant_pairs: u32,
}

impl OccupancyCounters {
    /// One O(N) pass over `patch_data`'s initial state, seeding every
    /// counter from scratch. Pair counting only looks at each site's
    /// `topology::forward_neighbors` (not all up to six) specifically so a
    /// full scan counts every unordered adjacent pair exactly once --
    /// checking every neighbor from every site would double-count each
    /// pair, once from each side.
    pub fn new(patch_data: &[u8], width: usize, seed: u64) -> Self {
        let site_count = patch_data.len();
        let mut counters = OccupancyCounters {
            vacant_sets: std::array::from_fn(|_| BucketedSet::new(site_count)),
            occupied_sets: std::array::from_fn(|_| BucketedSet::new(site_count)),
            co_ox_pairs: 0,
            h2_pairs: 0,
            h_oh_pairs: 0,
            vacant_pairs: 0,
        };
        if width == 0 {
            return counters;
        }
        let rows = patch_data.len() / width;

        for site_idx in 0..patch_data.len() {
            let state = patch_data[site_idx];
            // Any bit pattern other than VACANT or a single known species
            // is a corrupted/unknown site state (see
            // `layout::OCCUPANCY_MASK`'s doc comment) -- `update_membership`
            // simply doesn't add it to any set, so it can never be selected
            // as a reactant; the engine never writes such a state.
            counters.update_membership(site_idx, state, seed, true);

            for neighbor_idx in crate::topology::forward_neighbors(site_idx, width, rows)
                .into_iter()
                .flatten()
            {
                counters.add_pair(state, patch_data[neighbor_idx]);
            }
        }

        counters
    }

    /// Add (`insert = true`) or remove (`insert = false`) `site_idx` --
    /// currently holding `state` -- from whichever `vacant_sets`/
    /// `occupied_sets` bucket that state belongs to. The one place this
    /// branch (VACANT -> every species' vacant set, a known species ->
    /// just that species' occupied set, anything else -> no set at all)
    /// is written, shared by `new`'s seeding pass and
    /// `on_occupancy_change`'s old-state removal and new-state insertion.
    fn update_membership(&mut self, site_idx: usize, state: u8, seed: u64, insert: bool) {
        if state == layout::VACANT {
            for (species, &bit) in SPECIES_BITS.iter().enumerate() {
                let bucket = site_bucket(site_idx, bit, seed);
                if insert {
                    self.vacant_sets[species].insert(site_idx, bucket);
                } else {
                    self.vacant_sets[species].remove(site_idx, bucket);
                }
            }
        } else if let Some(species) = species_index(state) {
            let bucket = site_bucket(site_idx, state, seed);
            if insert {
                self.occupied_sets[species].insert(site_idx, bucket);
            } else {
                self.occupied_sets[species].remove(site_idx, bucket);
            }
        }
    }

    fn add_pair(&mut self, a: u8, b: u8) {
        if pair_matches(a, b, ADS_O, ADS_CO) {
            self.co_ox_pairs += 1;
        }
        if a == ADS_H && b == ADS_H {
            self.h2_pairs += 1;
        }
        if pair_matches(a, b, ADS_H, ADS_OH) {
            self.h_oh_pairs += 1;
        }
        if a == layout::VACANT && b == layout::VACANT {
            self.vacant_pairs += 1;
        }
    }

    fn remove_pair(&mut self, a: u8, b: u8) {
        if pair_matches(a, b, ADS_O, ADS_CO) {
            self.co_ox_pairs = self.co_ox_pairs.saturating_sub(1);
        }
        if a == ADS_H && b == ADS_H {
            self.h2_pairs = self.h2_pairs.saturating_sub(1);
        }
        if pair_matches(a, b, ADS_H, ADS_OH) {
            self.h_oh_pairs = self.h_oh_pairs.saturating_sub(1);
        }
        if a == layout::VACANT && b == layout::VACANT {
            self.vacant_pairs = self.vacant_pairs.saturating_sub(1);
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

        self.update_membership(site_idx, old_state, seed, false);
        self.update_membership(site_idx, new_state, seed, true);

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
            let reactant_a = (template.transition_a >> 8) as u8;
            let reactant_b = (template.transition_b >> 8) as u8;
            // Exhaustive match on which live pair-count pool this
            // template's reactant *pair* draws from, keyed off the
            // reactant species on both sides (not just site A) -- an
            // explicit match rather than an if/else chain so adding a new
            // bimolecular reaction kind that doesn't fit any existing
            // pool fails safe (0, never selected) instead of silently
            // being miscounted against an unrelated pool.
            if reactant_a == layout::VACANT && reactant_b == layout::VACANT {
                // Any dissociative adsorption (2* + gas -> 2 species*,
                // homoatomic like O2/H2 or heteroatomic like water
                // splitting): both sites start VACANT -- draws from the
                // shared adjacent-vacant-pair pool every such template
                // competes for (see `vacant_pairs`'s doc comment).
                self.vacant_pairs as u64
            } else if pair_matches(reactant_a, reactant_b, ADS_O, ADS_CO) {
                self.co_ox_pairs as u64
            } else if reactant_a == ADS_H && reactant_b == ADS_H {
                self.h2_pairs as u64
            } else if pair_matches(reactant_a, reactant_b, ADS_H, ADS_OH) {
                self.h_oh_pairs as u64
            } else {
                0
            }
        } else {
            let reactant_mask = (template.transition_a >> 8) as u8;
            let product_mask = (template.transition_a & 0xFF) as u8;
            let bucket = template.bin_id as usize;
            if bucket >= BUCKETS_PER_SPECIES {
                return 0;
            }
            if reactant_mask == 0 {
                // Adsorption: reactant is VACANT, species comes from the
                // product side.
                species_index(product_mask)
                    .map(|s| self.vacant_sets[s].len(bucket) as u64)
                    .unwrap_or(0)
            } else {
                species_index(reactant_mask)
                    .map(|s| self.occupied_sets[s].len(bucket) as u64)
                    .unwrap_or(0)
            }
        }
    }

    /// Per-template live weight (`rate_q16 * live_count * pressure_factor`)
    /// -- the one place this formula is computed. `total_propensity` and
    /// `select_event` both need it every step (the former for the
    /// exponential waiting-time draw, the latter for the weighted
    /// reaction-selection draw); a caller doing both in the same step
    /// (like `engine.rs`'s Gillespie loop) should call this once and pass
    /// the result to both, rather than let each recompute it -- see
    /// `select_event`'s doc comment.
    pub(crate) fn weights(&self, templates: &[ReactionRecord], pressures: &Pressures) -> Vec<f64> {
        templates
            .iter()
            .map(|t| t.rate_q16 as f64 * self.live_count(t) as f64 * pressure_factor(t, pressures))
            .collect()
    }

    /// Sum of every template's live weight -- the total propensity the
    /// exponential waiting-time draw needs. Zero means the domain has gone
    /// fully quiescent: no template has a site currently matching its
    /// reactant pattern (or every matching template's gas-phase pressure
    /// is zero).
    pub fn total_propensity(&self, templates: &[ReactionRecord], pressures: &Pressures) -> f64 {
        self.weights(templates, pressures).iter().sum()
    }

    /// Select which reaction fires next (weighted by `rate_q16 *
    /// live_count * pressure_factor`) and which site(s) it fires on,
    /// without applying anything -- callers are expected to apply the
    /// transition(s) via the existing
    /// `layout::apply_transition`/trajectory-logging path, then call
    /// `on_occupancy_change` per touched site. Returns `None` when every
    /// template's live weight is zero (domain quiescent, same semantics
    /// as `gillespie::GillespieDomain::step`).
    ///
    /// Takes the already-computed per-template `weights` (see `weights`)
    /// and their `total` rather than recomputing them from scratch --
    /// `engine.rs`'s Gillespie loop already needs `total` for the
    /// waiting-time draw before it gets here, so passing both through
    /// avoids a second O(template count) pass (and a second heap
    /// allocation) over the same formula every single fired event.
    #[allow(clippy::too_many_arguments)]
    pub fn select_event(
        &self,
        templates: &[ReactionRecord],
        weights: &[f64],
        total: f64,
        patch_data: &[u8],
        width: usize,
        rows_in_band: usize,
        rng: &mut Rng,
    ) -> Option<(u32, usize, Option<usize>)> {
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
            let site = self.find_monomolecular_site(template, rng)?;
            Some((chosen as u32, site, None))
        }
    }

    /// O(1): looks up the exact free-list `live_count` already used to
    /// weight this template (see `BucketedSet`) and picks a uniformly
    /// random member. No verification pass against `patch_data` needed --
    /// unlike the rejection-sampling design this replaced, set membership
    /// *is* the ground truth (kept exactly in sync by `new`/
    /// `on_occupancy_change`), not a guess to be checked.
    fn find_monomolecular_site(&self, template: &ReactionRecord, rng: &mut Rng) -> Option<usize> {
        let reactant_mask = (template.transition_a >> 8) as u8;
        let product_mask = (template.transition_a & 0xFF) as u8;
        let bucket = template.bin_id as usize;
        if bucket >= BUCKETS_PER_SPECIES {
            return None;
        }
        if reactant_mask == 0 {
            let species = species_index(product_mask)?;
            self.vacant_sets[species].pick(bucket, rng)
        } else {
            let species = species_index(reactant_mask)?;
            self.occupied_sets[species].pick(bucket, rng)
        }
    }

    fn find_bimolecular_pair(
        &self,
        template: &ReactionRecord,
        patch_data: &[u8],
        width: usize,
        rows_in_band: usize,
        rng: &mut Rng,
    ) -> Option<(usize, usize)> {
        let species_a_bit = (template.transition_a >> 8) as u8;
        let species_b_bit = (template.transition_b >> 8) as u8;
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
        // Deterministic fallback: `co_ox_pairs`/`h2_pairs`/`vacant_pairs`
        // (whichever backs this template's `live_count`) being positive
        // means a matching pair genuinely exists somewhere in the patch --
        // guaranteed to succeed, same as `find_monomolecular_site`'s
        // free-list lookup, just via a scan rather than an O(1) pick since
        // pairs aren't bucketed (see the module-level doc comment).
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
    use crate::layout::{ADS_CO, ADS_H, ADS_H2O, ADS_O, VACANT};

    fn rng() -> Rng {
        Rng::seeded(42)
    }

    /// Test convenience wrapper matching `select_event`'s pre-refactor
    /// signature: computes `weights`/`total` itself rather than making
    /// every test call site do it. Production code (`engine.rs`) computes
    /// them once per step and passes both through instead -- see
    /// `select_event`'s doc comment for why that matters there but not here.
    #[allow(clippy::too_many_arguments)]
    fn select(
        counters: &OccupancyCounters,
        templates: &[ReactionRecord],
        patch_data: &[u8],
        width: usize,
        rows_in_band: usize,
        rng: &mut Rng,
        pressures: &Pressures,
    ) -> Option<(u32, usize, Option<usize>)> {
        let weights = counters.weights(templates, pressures);
        let total: f64 = weights.iter().sum();
        counters.select_event(templates, &weights, total, patch_data, width, rows_in_band, rng)
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

    fn vacant_total(counters: &OccupancyCounters, species: usize) -> u32 {
        (0..BUCKETS_PER_SPECIES).map(|b| counters.vacant_sets[species].len(b) as u32).sum()
    }

    fn occupied_total(counters: &OccupancyCounters, species: usize) -> u32 {
        (0..BUCKETS_PER_SPECIES).map(|b| counters.occupied_sets[species].len(b) as u32).sum()
    }

    #[test]
    fn new_counts_vacant_and_occupied_sites_correctly() {
        let width = 4;
        let data = vec![VACANT, ADS_O, ADS_H, ADS_CO, VACANT, VACANT, ADS_O, ADS_O];
        let counters = OccupancyCounters::new(&data, width, 99);

        let total_vacant: u32 = (0..3).map(|s| vacant_total(&counters, s)).sum();
        // Every vacant site contributes to all 3 species' vacant counts.
        let vacant_sites = data.iter().filter(|&&s| s == VACANT).count() as u32;
        assert_eq!(total_vacant, vacant_sites * 3);

        let o_occupied = occupied_total(&counters, 0);
        assert_eq!(o_occupied, data.iter().filter(|&&s| s == ADS_O).count() as u32);
        let h_occupied = occupied_total(&counters, 1);
        assert_eq!(h_occupied, data.iter().filter(|&&s| s == ADS_H).count() as u32);
        let co_occupied = occupied_total(&counters, 2);
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
        // Compare exact set *membership*, not raw dense-array order --
        // swap-remove reorders a bucket's dense list, so two `BucketedSet`s
        // holding the same sites can differ in element order depending on
        // insert/remove history. Sorting is a stronger check than the old
        // count-only comparison this replaced: it now also catches a bug
        // that miscounted the same total but selected the wrong sites.
        fn sorted_members(set: &BucketedSet, bucket: usize) -> Vec<u32> {
            let mut v = set.dense[bucket].clone();
            v.sort_unstable();
            v
        }
        for species in 0..NUM_SPECIES {
            for bucket in 0..BUCKETS_PER_SPECIES {
                assert_eq!(
                    sorted_members(&counters.vacant_sets[species], bucket),
                    sorted_members(&brute.vacant_sets[species], bucket),
                    "vacant_sets[{species}][{bucket}] diverged from brute force"
                );
                assert_eq!(
                    sorted_members(&counters.occupied_sets[species], bucket),
                    sorted_members(&brute.occupied_sets[species], bucket),
                    "occupied_sets[{species}][{bucket}] diverged from brute force"
                );
            }
        }
        assert_eq!(counters.co_ox_pairs, brute.co_ox_pairs);
        assert_eq!(counters.h2_pairs, brute.h2_pairs);
    }

    #[test]
    fn bucketed_set_pick_and_len_reflect_swap_remove_reordering() {
        let mut set = BucketedSet::new(10);
        for site in [1usize, 4, 7] {
            set.insert(site, 2);
        }
        assert_eq!(set.len(2), 3);

        // Remove the middle element by insertion order -- exercises the
        // swap-with-last path, not just the simple "remove the last" case.
        set.remove(4, 2);
        assert_eq!(set.len(2), 2);
        let mut remaining = set.dense[2].clone();
        remaining.sort_unstable();
        assert_eq!(remaining, vec![1, 7]);

        set.remove(1, 2);
        set.remove(7, 2);
        assert_eq!(set.len(2), 0);
        let mut rng = Rng::seeded(3);
        assert_eq!(set.pick(2, &mut rng), None);
    }

    fn ads_template(species_bit: u8, bucket: u8, rate: u32) -> ReactionRecord {
        ReactionRecord {
            rate_q16: rate,
            bin_id: bucket,
            transition_a: species_bit as u16, // reactant=VACANT(0), product=species
            transition_b: 0,
            is_bimolecular: false,
        }
    }

    fn des_template(species_bit: u8, bucket: u8, rate: u32) -> ReactionRecord {
        ReactionRecord {
            rate_q16: rate,
            bin_id: bucket,
            transition_a: (species_bit as u16) << 8, // reactant=species, product=VACANT
            transition_b: 0,
            is_bimolecular: false,
        }
    }

    fn bimolecular_template(species_a_bit: u8, species_b_bit: u8, rate: u32) -> ReactionRecord {
        ReactionRecord {
            rate_q16: rate,
            bin_id: 0,
            transition_a: (species_a_bit as u16) << 8,
            transition_b: (species_b_bit as u16) << 8,
            is_bimolecular: true,
        }
    }

    #[test]
    fn pressure_factor_only_scales_adsorption_templates() {
        let pressures = Pressures { values: [2.0, 3.0, 5.0, 1.0, 11.0] };
        // Adsorption: pressure_factor equals that species' own pressure.
        assert_eq!(pressure_factor(&ads_template(ADS_O, 0, 1), &pressures), 2.0);
        assert_eq!(pressure_factor(&ads_template(ADS_H, 0, 1), &pressures), 3.0);
        assert_eq!(pressure_factor(&ads_template(ADS_CO, 0, 1), &pressures), 5.0);
        // H2O* adsorption is an ordinary single-gas monomolecular channel,
        // same shape as O2/H2/CO -- gates on its own pressure slot exactly
        // like they do (see `Pressures`' doc comment for why this differs
        // from water-splitting's H* + OH* products, which stay neutral).
        assert_eq!(pressure_factor(&ads_template(ADS_H2O, 0, 1), &pressures), 11.0);
        // Desorption and bimolecular: untouched by pressure, always 1.0.
        assert_eq!(pressure_factor(&des_template(ADS_O, 0, 1), &pressures), 1.0);
        assert_eq!(pressure_factor(&des_template(ADS_CO, 0, 1), &pressures), 1.0);
        assert_eq!(pressure_factor(&des_template(ADS_H2O, 0, 1), &pressures), 1.0);
        assert_eq!(pressure_factor(&bimolecular_template(ADS_O, ADS_CO, 1), &pressures), 1.0);
    }

    #[test]
    fn total_propensity_with_ones_pressure_matches_unpressured_formula() {
        let width = 4;
        let data = vec![VACANT, ADS_O, ADS_H, ADS_CO, VACANT, VACANT, ADS_O, ADS_O];
        let counters = OccupancyCounters::new(&data, width, 3);
        let bucket = site_bucket(0, ADS_O, 3) as u8;
        let templates = vec![ads_template(ADS_O, bucket, 1000), des_template(ADS_O, bucket, 500)];

        let with_ones = counters.total_propensity(&templates, &Pressures::ones());
        let manual: f64 = templates
            .iter()
            .map(|t| t.rate_q16 as f64 * counters.live_count(t) as f64)
            .sum();
        assert_eq!(with_ones, manual);
    }

    #[test]
    fn total_propensity_scales_linearly_with_adsorption_pressure() {
        let width = 4;
        let data = vec![VACANT; 16];
        let counters = OccupancyCounters::new(&data, width, 9);
        let bucket = site_bucket(0, ADS_CO, 9) as u8;
        // Only a CO adsorption template -- every VACANT site matches, so
        // this isolates the pressure multiplier's effect cleanly.
        let templates = vec![ads_template(ADS_CO, bucket, 1000)];

        let baseline = counters.total_propensity(&templates, &Pressures::ones());
        let doubled = counters.total_propensity(&templates, &Pressures { values: [1.0, 1.0, 2.0, 1.0, 1.0] });
        assert!(baseline > 0.0, "adsorption template should have nonzero live count");
        assert!((doubled - 2.0 * baseline).abs() < 1e-9);

        // Desorption is untouched by the same pressure change.
        let des_templates = vec![des_template(ADS_CO, bucket, 1000)];
        let des_baseline = counters.total_propensity(&des_templates, &Pressures::ones());
        let des_under_pressure =
            counters.total_propensity(&des_templates, &Pressures { values: [1.0, 1.0, 2.0, 1.0, 1.0] });
        assert_eq!(des_baseline, des_under_pressure);
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
                select(&counters, &templates, &data, width, height, &mut rng, &Pressures::ones())
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
                select(&counters, &templates, &data, width, height, &mut rng, &Pressures::ones())
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
            transition_a: (ADS_O as u16) << 8,
            transition_b: (ADS_CO as u16) << 8,
            is_bimolecular: true,
        };
        let templates = vec![template];

        let mut rng = rng();
        for _ in 0..200 {
            if let Some((_, site_a, site_b)) =
                select(&counters, &templates, &data, width, height, &mut rng, &Pressures::ones())
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
            select(&counters, &templates, &data, width, 4, &mut rng, &Pressures::ones()),
            None
        );
    }

    fn dissociative_ads_template(species_bit: u8, rate: u32) -> ReactionRecord {
        ReactionRecord {
            rate_q16: rate,
            bin_id: 0,
            transition_a: species_bit as u16, // reactant=VACANT(0), product=species, both sites
            transition_b: species_bit as u16,
            is_bimolecular: true,
        }
    }

    /// The falsifying test Phase 3 (genuine two-site dissociative
    /// adsorption) exists to pass: a dissociative-adsorption template must
    /// only ever fire on a real adjacent *vacant* pair, never on a lone
    /// vacant site with no vacant neighbor -- mirroring the existing
    /// recombination-direction bimolecular test, but for the reverse
    /// (gas-consuming) direction this phase adds.
    #[test]
    fn select_event_dissociative_adsorption_only_fires_on_a_genuinely_adjacent_vacant_pair() {
        let width = 6;
        let height = 6;
        let seed = 11u64;
        let mut data = vec![ADS_O; width * height];
        // One real adjacent vacant pair at (10, 11); a lone, non-adjacent
        // vacant site elsewhere that must never be chosen as a partner
        // since it has no vacant neighbor.
        data[10] = VACANT;
        data[11] = VACANT;
        data[30] = VACANT;
        let counters = OccupancyCounters::new(&data, width, seed);
        assert_eq!(counters.vacant_pairs, 1);

        let templates = vec![dissociative_ads_template(ADS_H, u32::MAX)];

        let mut rng = rng();
        for _ in 0..200 {
            if let Some((_, site_a, site_b)) =
                select(&counters, &templates, &data, width, height, &mut rng, &Pressures::ones())
            {
                let site_b = site_b.expect("dissociative adsorption must return a second site");
                assert_eq!(data[site_a], VACANT);
                assert_eq!(data[site_b], VACANT);
                let neighbors: Vec<usize> = crate::topology::all_neighbors(site_a, width, height)
                    .into_iter()
                    .flatten()
                    .collect();
                assert!(
                    neighbors.contains(&site_b),
                    "dissociative-adsorption sites must be topology-adjacent"
                );
            }
        }
    }

    /// Coverage-dependent rate: a dissociative-adsorption template's
    /// propensity must genuinely track how many adjacent vacant pairs
    /// currently exist, not fire at a fixed rate regardless of coverage --
    /// the whole point of this phase versus the pseudo-monomolecular
    /// approximation it replaces.
    #[test]
    fn total_propensity_for_dissociative_adsorption_tracks_vacant_pair_count() {
        let width = 6;
        let height = 6;
        let seed = 21u64;

        let mostly_vacant = vec![VACANT; width * height];
        let counters_high = OccupancyCounters::new(&mostly_vacant, width, seed);

        let mut mostly_occupied = vec![ADS_O; width * height];
        mostly_occupied[10] = VACANT;
        mostly_occupied[11] = VACANT; // exactly one adjacent vacant pair
        let counters_low = OccupancyCounters::new(&mostly_occupied, width, seed);
        assert_eq!(counters_low.vacant_pairs, 1);
        assert!(counters_high.vacant_pairs > counters_low.vacant_pairs);

        let templates = vec![dissociative_ads_template(ADS_H, 1000)];
        let propensity_high = counters_high.total_propensity(&templates, &Pressures::ones());
        let propensity_low = counters_low.total_propensity(&templates, &Pressures::ones());

        assert!(
            propensity_high > propensity_low,
            "a lattice with more adjacent vacant pairs should have higher dissociative-adsorption \
             propensity: high={propensity_high}, low={propensity_low}"
        );
        // Exact linear relationship: rate_q16 * vacant_pairs.
        assert_eq!(propensity_low, 1000.0 * counters_low.vacant_pairs as f64);
    }

    #[test]
    fn pressure_couples_dissociative_adsorption_same_as_monomolecular() {
        let pressures = Pressures { values: [1.0, 7.0, 1.0, 1.0, 1.0] };
        let template = dissociative_ads_template(ADS_H, 1);
        assert_eq!(pressure_factor(&template, &pressures), 7.0);

        // A recombination-direction bimolecular template (occupied
        // reactant) must stay untouched by the same pressure vector.
        let recombination = bimolecular_template(ADS_H, ADS_H, 1);
        assert_eq!(pressure_factor(&recombination, &pressures), 1.0);
    }

    fn heteroatomic_dissociative_ads_template(species_a_bit: u8, species_b_bit: u8, rate: u32) -> ReactionRecord {
        ReactionRecord {
            rate_q16: rate,
            bin_id: 0,
            transition_a: species_a_bit as u16, // reactant=VACANT(0), product=species_a
            transition_b: species_b_bit as u16, // reactant=VACANT(0), product=species_b
            is_bimolecular: true,
        }
    }

    fn heteroatomic_recombination_template(species_a_bit: u8, species_b_bit: u8, rate: u32) -> ReactionRecord {
        ReactionRecord {
            rate_q16: rate,
            bin_id: 0,
            transition_a: (species_a_bit as u16) << 8, // reactant=species_a, product=VACANT
            transition_b: (species_b_bit as u16) << 8, // reactant=species_b, product=VACANT
            is_bimolecular: true,
        }
    }

    /// Phase 4 (water splitting, `2* + H2O(g) <-> H* + OH*`) is the first
    /// *heteroatomic* dissociative-adsorption reaction -- the two sites
    /// produce/consume *different* species, unlike O2/H2's homoatomic
    /// case (Phase 3). The forward direction must still draw from the
    /// same shared `vacant_pairs` pool (any adjacent vacant pair is a
    /// valid landing site, regardless of which two species end up there).
    #[test]
    fn live_count_heteroatomic_dissociative_adsorption_uses_vacant_pairs() {
        let width = 6;
        let height = 6;
        let seed = 5u64;
        let mut data = vec![ADS_O; width * height];
        data[10] = VACANT;
        data[11] = VACANT;
        let counters = OccupancyCounters::new(&data, width, seed);
        assert_eq!(counters.vacant_pairs, 1);

        let forward = heteroatomic_dissociative_ads_template(ADS_H, ADS_OH, 1000);
        assert_eq!(
            counters.total_propensity(&[forward], &Pressures::ones()),
            1000.0 * counters.vacant_pairs as f64
        );
    }

    /// The reverse (associative desorption, `H* + OH* -> 2*`) must draw
    /// from its own dedicated live pair-count (`h_oh_pairs`), not get
    /// mixed up with `co_ox_pairs`/`h2_pairs`/`vacant_pairs` -- this is
    /// exactly the kind of mistake the old if/else-chain `live_count`
    /// (fixed this phase into an exhaustive match) could have made for a
    /// pair type it didn't explicitly know about.
    #[test]
    fn live_count_h_oh_recombination_uses_its_own_pair_pool() {
        let width = 6;
        let height = 6;
        let seed = 6u64;
        let mut data = vec![VACANT; width * height];
        data[10] = ADS_H;
        data[11] = ADS_OH;
        let counters = OccupancyCounters::new(&data, width, seed);
        assert_eq!(counters.h_oh_pairs, 1);
        assert_eq!(counters.co_ox_pairs, 0);
        assert_eq!(counters.h2_pairs, 0);

        let reverse = heteroatomic_recombination_template(ADS_H, ADS_OH, 1000);
        assert_eq!(
            counters.total_propensity(&[reverse], &Pressures::ones()),
            1000.0 * counters.h_oh_pairs as f64
        );
    }

    /// A bimolecular reactant pair this build genuinely doesn't recognize
    /// (neither vacant-pair, O/CO, H/H, nor H/OH) must contribute zero
    /// live weight -- selectable-but-never-selected is the safe failure
    /// mode, not silently miscounted against an unrelated pool.
    #[test]
    fn live_count_unknown_bimolecular_pair_is_zero() {
        let width = 4;
        let data = vec![ADS_O; 16]; // no H/OH/CO anywhere
        let counters = OccupancyCounters::new(&data, width, 1);
        let bogus = heteroatomic_recombination_template(ADS_CO, ADS_OH, 1000);
        assert_eq!(counters.total_propensity(&[bogus], &Pressures::ones()), 0.0);
    }

    /// Heteroatomic dissociative adsorption (water splitting) genuinely
    /// consumes a gas-phase molecule (H2O) that `Pressures` doesn't track
    /// -- it must be treated as pressure-neutral (1.0) rather than
    /// incorrectly gated on H2's pressure just because site A's product
    /// happens to be H* (which does have a pressure slot). See
    /// `pressure_factor`'s doc comment.
    #[test]
    fn pressure_factor_is_neutral_for_heteroatomic_dissociative_adsorption() {
        let pressures = Pressures { values: [1.0, 99.0, 1.0, 1.0, 1.0] }; // H2 pressure cranked up
        let forward = heteroatomic_dissociative_ads_template(ADS_H, ADS_OH, 1);
        assert_eq!(
            pressure_factor(&forward, &pressures),
            1.0,
            "must not accidentally gate on H2 pressure just because site A produces H*"
        );

        // Homoatomic dissociative adsorption (both sites the same
        // species) is unaffected by this guard -- still genuinely
        // pressure-coupled, matching Phase 3's existing behavior.
        let homoatomic = dissociative_ads_template(ADS_H, 1);
        assert_eq!(pressure_factor(&homoatomic, &pressures), 99.0);

        // The reverse (associative desorption) direction is untouched
        // regardless -- no gas-phase reactant.
        let reverse = heteroatomic_recombination_template(ADS_H, ADS_OH, 1);
        assert_eq!(pressure_factor(&reverse, &pressures), 1.0);
    }

    /// End-to-end: the reverse (associative desorption) direction must
    /// only ever fire on a genuinely adjacent H*/OH* pair.
    #[test]
    fn select_event_h_oh_recombination_only_fires_on_a_genuinely_adjacent_pair() {
        let width = 6;
        let height = 6;
        let seed = 9u64;
        let mut data = vec![VACANT; width * height];
        // One real adjacent H*/OH* pair; a lone, non-adjacent H* that
        // must never be chosen as a partner site.
        data[10] = ADS_H;
        data[11] = ADS_OH;
        data[30] = ADS_H;
        let counters = OccupancyCounters::new(&data, width, seed);
        assert_eq!(counters.h_oh_pairs, 1);

        let templates = vec![heteroatomic_recombination_template(ADS_H, ADS_OH, u32::MAX)];
        let mut rng = rng();
        for _ in 0..200 {
            if let Some((_, site_a, site_b)) =
                select(&counters, &templates, &data, width, height, &mut rng, &Pressures::ones())
            {
                let site_b = site_b.expect("bimolecular event must return a second site");
                let pair = (data[site_a], data[site_b]);
                assert!(
                    pair == (ADS_H, ADS_OH) || pair == (ADS_OH, ADS_H),
                    "fired on non-matching pair {pair:?}"
                );
                let neighbors: Vec<usize> = crate::topology::all_neighbors(site_a, width, height)
                    .into_iter()
                    .flatten()
                    .collect();
                assert!(neighbors.contains(&site_b), "must be topology-adjacent");
            }
        }
    }
}
