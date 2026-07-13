//! O(1) next-reaction selection via Partial-Propensity Composition-Rejection
//! (the Slepoy/Thompson/Plimpton "SSA-CR" scheme), on fixed-point integers.
//!
//! A textbook Gillespie direct method draws one uniform random number over
//! the cumulative sum of all `M` reaction propensities, which is an O(M) (or
//! O(log M) with a Fenwick/segment tree) search. That search cost is what
//! makes naive SSA intractable once `M` reaches the reaction counts a
//! trillion-event OC20-scale lattice produces. Composition-rejection instead
//! groups reactions into a *fixed* number of magnitude-class bins and
//! answers "which reaction fires next?" with two integer draws whose cost
//! never depends on `M`:
//!
//! 1. **Composition** -- walk the `NUM_BINS` bin totals (a compile-time
//!    constant) to land on a bin `j`, proportional to that bin's share of
//!    total propensity.
//! 2. **Rejection** -- pick a uniformly random member of bin `j` and accept
//!    it with probability `rate_i / ceiling(j)`. Because every member of
//!    bin `j` has a rate within `[2^j, 2^(j+1))`, `ceiling(j) = 2^(j+1)`
//!    bounds the acceptance probability below by 1/2, so the expected
//!    number of rejection trials is <= 2 regardless of how many reactions
//!    share the bin.
//!
//! All of the per-candidate arithmetic (steps 1 and 2) is done in Q32.32
//! fixed-point (`FixedPoint`) so the hot sampling loop never leaves the
//! integer pipeline for the FPU div/sqrt microcode path.

use crate::layout::{ReactionLut, ReactionLutBlock};

/// Number of composition-rejection magnitude-class bins. Fixed at compile
/// time: this is exactly what makes bin selection O(1) rather than O(M) --
/// the composition walk below is always exactly `NUM_BINS` iterations no
/// matter how many live reactions the lattice has.
pub const NUM_BINS: usize = 32;

// ------------------------------------------------------------------------
// Fixed-point propensity arithmetic
// ------------------------------------------------------------------------

/// Q32.32 fixed-point value: bits `[32..64)` are the integer part, bits
/// `[0..32)` the fraction. Reaction propensities, bin sums, and the random
/// draws compared against them all live in this representation so the
/// per-event selection loop is pure integer add/compare/shift -- no
/// division, no FPU transcendental, and no rounding drift from repeatedly
/// re-normalizing floating-point probabilities across billions of events.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct FixedPoint(pub u64);

impl FixedPoint {
    pub const FRAC_BITS: u32 = 32;
    pub const ONE: FixedPoint = FixedPoint(1u64 << Self::FRAC_BITS);
    pub const ZERO: FixedPoint = FixedPoint(0);

    /// Widen a LUT rate stored as Q16.16 into this module's Q32.32 domain.
    /// The extra 16 fractional bits of headroom absorb rounding error from
    /// summing millions of per-reaction propensities into a single bin
    /// total without the sum's low bits collapsing to zero.
    #[inline(always)]
    pub const fn from_q16(raw: u32) -> Self {
        FixedPoint((raw as u64) << 16)
    }

    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        FixedPoint(self.0.saturating_add(other.0))
    }

    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        (self.0 as f64) / (Self::ONE.0 as f64)
    }
}

// ------------------------------------------------------------------------
// Deterministic, allocation-free PRNG
// ------------------------------------------------------------------------

/// SplitMix64-derived counter PRNG. Not cryptographic -- chosen purely for
/// speed and statistical quality per random draw, since the composition-
/// rejection loop may need several draws per fired event. Holds no heap
/// state, `Copy`-free but trivially `Clone`, and every method is a handful
/// of integer ops.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    #[inline(always)]
    pub const fn seeded(seed: u64) -> Self {
        // Avoid the fixed point at seed == 0, which would otherwise emit an
        // all-zero stream forever.
        Rng { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, bound)`. Uses the standard widening-multiply
    /// scaling trick (treat `next_u64()` as a fixed-point fraction of
    /// `2^64` and rescale) instead of `% bound`, so there's no division
    /// instruction and no bias blow-up for power-of-two-adjacent bounds.
    #[inline(always)]
    pub fn next_u32_below(&mut self, bound: u32) -> u32 {
        debug_assert!(bound > 0, "sampling range must be non-empty");
        (((self.next_u64() as u128) * (bound as u128)) >> 64) as u32
    }

    /// Uniform `FixedPoint` in `[0, max)`, via the same widening-multiply
    /// rescale as `next_u32_below` but over the full Q32.32 domain.
    #[inline(always)]
    pub fn next_fixed(&mut self, max: FixedPoint) -> FixedPoint {
        FixedPoint((((self.next_u64() as u128) * (max.0 as u128)) >> 64) as u64)
    }
}

// ------------------------------------------------------------------------
// Composition-rejection bin index
// ------------------------------------------------------------------------

/// One magnitude-class bin's metadata. Bin *membership* is not a separate
/// heap-allocated list: `reactions.lut` is built pre-sorted by `bin_id`, so
/// every bin is simply the contiguous global-reaction-id range
/// `[start, start + count)` -- `CompositionTable` only ever stores that
/// range plus the bin's summed propensity, both fixed-size scalars.
#[derive(Clone, Copy, Debug, Default)]
struct BinMeta {
    start: u32,
    count: u32,
    total_propensity: FixedPoint,
}

/// The O(1) reaction sampler. Holds exactly `NUM_BINS` `BinMeta` records
/// (`[BinMeta; NUM_BINS]`, stack-resident, no `Vec`) plus the grand total
/// propensity; sampling never allocates and never inspects more than
/// `NUM_BINS` bins or a small constant number of bin members.
pub struct CompositionTable {
    bins: [BinMeta; NUM_BINS],
    total_propensity: FixedPoint,
}

impl CompositionTable {
    /// Build the bin index with one linear O(M) pass over `lut` -- this is
    /// one-time setup cost paid once when the simulation loads its reaction
    /// set, not part of the per-event hot loop the O(1) claim is about.
    /// Relies on `reactions.lut` being pre-sorted by `bin_id` ascending
    /// (a build-time invariant of the tool that produces the file), which
    /// is what lets each bin's membership collapse to a `[start, count)`
    /// range instead of a scattered, separately-allocated index.
    pub fn build(lut: &ReactionLut) -> Self {
        let mut bins = [BinMeta::default(); NUM_BINS];
        let mut total = FixedPoint::ZERO;
        let reaction_count = lut.len() * ReactionLutBlock::LANES;

        for reaction_id in 0..reaction_count {
            let (rate_q16, bin_id, _transition) = lut.rate_of(reaction_id);
            let bin = &mut bins[bin_id as usize % NUM_BINS];
            if bin.count == 0 {
                bin.start = reaction_id as u32;
            }
            bin.count += 1;

            let rate = FixedPoint::from_q16(rate_q16);
            bin.total_propensity = bin.total_propensity.saturating_add(rate);
            total = total.saturating_add(rate);
        }

        Self {
            bins,
            total_propensity: total,
        }
    }

    #[inline(always)]
    pub fn total_propensity(&self) -> FixedPoint {
        self.total_propensity
    }

    /// This bin's rejection-envelope ceiling: every member has rate in
    /// `[2^bin_index, 2^(bin_index+1))` in the fixed-point rate domain, so
    /// `2^(bin_index+1)` (expressed relative to `FixedPoint::ONE`) is a
    /// valid upper bound for the acceptance test without inspecting any
    /// individual member. The shift amount is clamped to stay inside `u64`
    /// -- bins beyond that point saturate to `u64::MAX` and simply always
    /// accept, which only matters for magnitude classes far outside any
    /// physically meaningful OC20 rate constant.
    #[inline(always)]
    fn bin_ceiling(bin_index: usize) -> FixedPoint {
        let shift = (Self::FRAC_BITS_PLUS_ONE + bin_index as u32).min(63);
        FixedPoint(1u64 << shift)
    }
    const FRAC_BITS_PLUS_ONE: u32 = FixedPoint::FRAC_BITS + 1;

    /// Select the next reaction to fire in expected O(1) time, independent
    /// of the total reaction count. Returns `None` only when every bin is
    /// empty (no reactions currently active anywhere in the domain).
    pub fn sample_reaction(&self, rng: &mut Rng, lut: &ReactionLut) -> Option<u32> {
        if self.total_propensity == FixedPoint::ZERO {
            return None;
        }

        // --- Stage 1: composition ---
        // Fixed NUM_BINS iterations, never a function of the live reaction
        // count -- this is the O(1) property the module exists to provide.
        let draw = rng.next_fixed(self.total_propensity);
        let mut acc = FixedPoint::ZERO;
        let mut chosen = NUM_BINS - 1;
        for (j, bin) in self.bins.iter().enumerate() {
            acc = acc.saturating_add(bin.total_propensity);
            if draw.0 < acc.0 {
                chosen = j;
                break;
            }
        }

        let bin = &self.bins[chosen];
        if bin.count == 0 {
            // Landed in an empty bin due to fixed-point rounding at a bin
            // boundary; fall back to the highest non-empty bin below it.
            return self.sample_from_nearest_nonempty(chosen, rng, lut);
        }

        // --- Stage 2: rejection ---
        // Expected <= 2 trials: every member's rate is >= ceiling/2, so
        // each trial accepts with probability >= 1/2.
        let ceiling = Self::bin_ceiling(chosen);
        loop {
            let lane = rng.next_u32_below(bin.count);
            let reaction_id = bin.start + lane;
            let (rate_q16, _, _) = lut.rate_of(reaction_id as usize);
            let rate = FixedPoint::from_q16(rate_q16);

            let coin = rng.next_fixed(ceiling);
            if coin.0 < rate.0 {
                return Some(reaction_id);
            }
        }
    }

    fn sample_from_nearest_nonempty(
        &self,
        from: usize,
        rng: &mut Rng,
        lut: &ReactionLut,
    ) -> Option<u32> {
        for j in (0..=from).rev() {
            let bin = &self.bins[j];
            if bin.count == 0 {
                continue;
            }
            let ceiling = Self::bin_ceiling(j);
            loop {
                let lane = rng.next_u32_below(bin.count);
                let reaction_id = bin.start + lane;
                let (rate_q16, _, _) = lut.rate_of(reaction_id as usize);
                let rate = FixedPoint::from_q16(rate_q16);
                let coin = rng.next_fixed(ceiling);
                if coin.0 < rate.0 {
                    return Some(reaction_id);
                }
            }
        }
        None
    }
}

/// One local Gillespie SSA loop over a single spatial domain's reaction
/// set: samples a reaction with `CompositionTable`, then advances
/// simulated time by the exponential waiting-time draw `tau = -ln(u) /
/// total_propensity`. The waiting-time draw is the one place this module
/// intentionally uses a floating-point transcendental (`ln`) -- the
/// continuous-time SSA's exponential inter-event distribution has no
/// integer/fixed-point closed form, whereas *which* reaction fires (the
/// part that dominates per-event cost at trillion-event scale) stays
/// entirely on the fixed-point path above.
pub struct GillespieDomain {
    pub table: CompositionTable,
    pub rng: Rng,
    pub sim_time: f64,
}

impl GillespieDomain {
    pub fn new(lut: &ReactionLut, seed: u64) -> Self {
        Self {
            table: CompositionTable::build(lut),
            rng: Rng::seeded(seed),
            sim_time: 0.0,
        }
    }

    /// Advance one SSA step. Returns `Some((reaction_id, tau))` for the
    /// reaction that fired and the waiting time consumed, or `None` if the
    /// domain has gone fully quiescent (no active reactions).
    pub fn step(&mut self, lut: &ReactionLut) -> Option<(u32, f64)> {
        let reaction_id = self.table.sample_reaction(&mut self.rng, lut)?;

        let total = self.table.total_propensity().to_f64();
        let u = ((self.rng.next_u64() >> 11) as f64) * (1.1102230246251565e-16); // 2^-53
        let u = u.max(f64::MIN_POSITIVE); // guard ln(0)
        let tau = -u.ln() / total;

        self.sim_time += tau;
        Some((reaction_id, tau))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout;
    use crate::test_support::temp_path;

    fn lut_from(records: Vec<(u32, u8, u8)>, tag: &str) -> (ReactionLut, std::path::PathBuf) {
        let blocks = layout::pack_records_into_blocks(records);
        let path = temp_path(tag);
        layout::write_lut(&path, &blocks).unwrap();
        let lut = ReactionLut::open(&path).unwrap();
        (lut, path)
    }

    #[test]
    fn from_q16_widens_into_q32_32_domain() {
        assert_eq!(FixedPoint::from_q16(1).0, 1u64 << 16);
        assert_eq!(FixedPoint::from_q16(0).0, 0);
    }

    #[test]
    fn saturating_add_caps_at_u64_max() {
        let a = FixedPoint(u64::MAX - 5);
        let b = FixedPoint(10);
        assert_eq!(a.saturating_add(b).0, u64::MAX);
    }

    #[test]
    fn to_f64_matches_fraction_of_one() {
        assert_eq!(FixedPoint::ONE.to_f64(), 1.0);
        assert_eq!(FixedPoint::ZERO.to_f64(), 0.0);
        let half = FixedPoint(FixedPoint::ONE.0 / 2);
        assert!((half.to_f64() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn next_u32_below_stays_in_bounds() {
        let mut rng = Rng::seeded(42);
        for _ in 0..10_000 {
            assert!(rng.next_u32_below(17) < 17);
        }
    }

    #[test]
    fn next_fixed_stays_below_max() {
        let mut rng = Rng::seeded(7);
        let max = FixedPoint(1_000_000);
        for _ in 0..10_000 {
            assert!(rng.next_fixed(max).0 < max.0);
        }
    }

    #[test]
    fn seeded_rng_is_deterministic() {
        let mut a = Rng::seeded(123);
        let mut b = Rng::seeded(123);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn seed_zero_does_not_degenerate() {
        let mut rng = Rng::seeded(0);
        assert_ne!(rng.next_u64(), 0);
    }

    #[test]
    fn bin_ceiling_matches_upper_bound_of_its_magnitude_class() {
        assert_eq!(CompositionTable::bin_ceiling(0).0, 1u64 << 33);
        assert_eq!(CompositionTable::bin_ceiling(1).0, 1u64 << 34);
    }

    #[test]
    fn bin_ceiling_clamps_shift_to_63_bits() {
        // FRAC_BITS_PLUS_ONE (33) + (NUM_BINS - 1) (31) = 64, which would
        // overflow a u64 shift -- must clamp to 63.
        assert_eq!(CompositionTable::bin_ceiling(NUM_BINS - 1).0, 1u64 << 63);
    }

    #[test]
    fn sample_reaction_returns_none_when_lut_is_empty() {
        let (lut, path) = lut_from(Vec::new(), "empty_lut");
        assert_eq!(lut.len(), 0);

        let table = CompositionTable::build(&lut);
        let mut rng = Rng::seeded(1);
        assert_eq!(table.sample_reaction(&mut rng, &lut), None);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sample_reaction_always_returns_the_only_active_reaction() {
        let (lut, path) = lut_from(vec![(1000u32, 5u8, 0u8)], "single_reaction");

        let table = CompositionTable::build(&lut);
        let mut rng = Rng::seeded(99);
        for _ in 0..1000 {
            assert_eq!(table.sample_reaction(&mut rng, &lut), Some(0));
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sample_reaction_favors_the_higher_propensity_reaction() {
        // Reaction 1's rate is ~1000x reaction 0's, so it should dominate
        // stage-1 composition selection.
        let (lut, path) = lut_from(
            vec![(1u32, 0u8, 0u8), (1_000_000u32, 19u8, 0u8)],
            "two_reactions",
        );

        let table = CompositionTable::build(&lut);
        let mut rng = Rng::seeded(2024);
        let trials = 5000;
        let mut count_high = 0;
        for _ in 0..trials {
            match table.sample_reaction(&mut rng, &lut) {
                Some(0) => {}
                Some(1) => count_high += 1,
                other => panic!("unexpected reaction id {other:?}"),
            }
        }
        assert!(count_high as f64 / trials as f64 > 0.99);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gillespie_domain_step_advances_simulation_time() {
        let (lut, path) = lut_from(vec![(500u32, 4u8, 0u8)], "domain_step");

        let mut domain = GillespieDomain::new(&lut, 55);
        let before = domain.sim_time;
        let (reaction_id, tau) = domain.step(&lut).expect("domain has an active reaction");
        assert_eq!(reaction_id, 0);
        assert!(tau > 0.0);
        assert!(domain.sim_time > before);

        let _ = std::fs::remove_file(&path);
    }
}
