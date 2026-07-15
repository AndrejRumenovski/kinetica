# kinetica

Asynchronous, out-of-core lattice kinetic Monte Carlo (kMC) engine for
single-workstation, OC20-scale surface reaction simulation, written in
pure Rust.

The lattice lives as a memory-mapped file rather than in heap-resident
`Vec`s, is a real hexagonal fcc(111) close-packed surface geometry (see
"Lattice geometry and target surface: Pd(111)" below) rather than a
generic square grid, and the simulation is spatially decomposed into
independently-scheduled `rayon` work-stealing patches whose fired-reaction
trajectory is streamed out through a double-buffered `io_uring` writer so
no compute thread ever blocks on disk I/O.

Two reaction-selection engines coexist, auto-selected from a magic header
in `reactions.lut` itself (no CLI flag to keep in sync): a **static**
composition-rejection sampler (O(1) next-reaction selection regardless of
how many fixed-rate channels are active — `kinetica --generate-lut`'s
synthetic demo mode) and an **occupancy-gated** one (real, `oc20_ingest`-
built data — propensity scales with how many lattice sites *currently*
match a reaction's reactant state, so a reaction can only ever fire where
its reactant genuinely is). See "Occupancy-gated kMC" below for why the
second one exists and how it works.

## Building

The hot Gillespie loop and the neighborhood-scan kernels in `layout.rs`
are written to vectorize under AVX2/FMA, which the default `rustc` target
does not enable. Always build with:

```sh
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" cargo build --release
```

Because this bakes in host-specific instructions, resulting binaries are
not portable across heterogeneous CPU fleets — rebuild per target machine.

## Layout

| Path                    | Purpose                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `src/layout.rs`          | Bit-packed mmap'd lattice, cache-line-aligned `ReactionLutBlock` reaction table, LUT packing/writing, magic-header `LutKind` |
| `src/topology.rs`        | Neighbor topology (hexagonal, fcc(111) -- six neighbors per site) shared by the occupancy-gated engine and the bimolecular partner search |
| `src/gillespie.rs`       | O(1) partial-propensity composition-rejection (SSA-CR) reaction sampler, fixed-point propensity arithmetic -- the `Static`/`--generate-lut` engine |
| `src/occupancy.rs`       | Live per-patch occupancy counters + bounded-rejection site selection -- the `OccupancyGated`/real-data engine |
| `src/engine.rs`          | Spatial domain decomposition, rayon work-stealing patches, crossbeam boundary migration, double-buffered `io_uring` trajectory writer, dispatch between the two engines |
| `src/lib.rs`             | Library surface shared by `kinetica` and auxiliary tools              |
| `src/main.rs`            | `kinetica` CLI entrypoint                                              |
| `src/bin/oc20_ingest.rs` | Builds `reactions.lut` from real adsorption-energy data (OC20 or Catalysis-Hub) |
| `scripts/extract_energies.py` | Pulls adsorption-energy records from OC20 IS2RE LMDB shards |
| `scripts/extract_catalysis_hub.py` | Pulls the same record format from the Catalysis-Hub.org GraphQL API, plus real transition-state barriers where they exist |
| `scripts/oc20e_format.py`     | Shared binary format both extraction scripts write |

## Running the simulator

```sh
./target/release/kinetica \
    --lut-path reactions.lut \
    --lattice-path surface.lattice \
    --trajectory-path trajectory.bin
```

| Flag                 | Default            | Meaning                                  |
|-----------------------|---------------------|-------------------------------------------|
| `--lattice-path`      | `surface.lattice`  | Backing mmap file for the surface        |
| `--lattice-width`     | `4096`              | Lattice width in sites                   |
| `--lattice-height`    | `4096`              | Lattice height in sites                  |
| `--lut-path`          | `reactions.lut`     | Reaction rate-constant table             |
| `--trajectory-path`   | `trajectory.bin`    | Output fired-reaction trajectory log     |
| `--patches`           | available CPUs      | Spatial domains / rayon tasks            |
| `--steps`             | `1000000`           | Gillespie steps per patch                |
| `--generate-lut <N>`  | —                   | Synthesize `N` demo reactions into `--lut-path` instead of using a real one |
| `--pressure-o2 <F>`   | `1.0`                | Relative O2 partial pressure — gates O* adsorption |
| `--pressure-h2 <F>`   | `1.0`                | Relative H2 partial pressure — gates H* adsorption |
| `--pressure-co <F>`   | `1.0`                | Relative CO partial pressure — gates CO* adsorption |

The three `--pressure-*` flags are runtime simulator parameters, not baked
into `reactions.lut` — changing the feed-gas composition never requires
rebuilding the LUT. They only affect the occupancy-gated engine (real
data), and only an adsorption channel's propensity (`VACANT -> species`);
desorption and bimolecular reactions are untouched, since neither consumes
a gas-phase molecule. See "Gas-phase pressure coupling" below.

## Occupancy-gated kMC

Earlier versions of this engine treated every reaction as an independent,
always-available channel: `gillespie::CompositionTable` built its
propensity index once from `reactions.lut` alone, never touching the
lattice, and a fired reaction was applied to a *uniformly random* site
with no check that the site actually held the reactant. For
`--generate-lut`'s synthetic demo data that's a reasonable simplification
(the point is exercising the HPC architecture, not real chemistry). For
real, `oc20_ingest`-built data it was a genuine correctness gap: an
adsorption event applied via a bitwise OR to an already-occupied site
would silently set two adsorbate bits on the same site at once, and
propensities had no way to reflect the surface actually running out of a
species to desorb or filling up so there was nowhere left to adsorb.

`src/occupancy.rs` fixes this for real-data LUTs. Every lattice site's
"which quantile bucket does this site belong to, for a given species" is a
deterministic hash of its own index (`occupancy::site_bucket`) — no
per-site storage. Each patch keeps a live count, per (species, bucket), of
how many sites are currently vacant (what an adsorption template's
propensity scales with) or occupied by that species (what a desorption
template scales with), plus a live count of adjacent (O\*, CO\*) and
(H\*, H\*) site pairs for the two bimolecular reactions — all updated in
O(1) amortized time per fired event, never rescanned. Site selection uses
bounded rejection sampling (try a random candidate, verify it actually
matches, retry on a miss, fall back to a guaranteed deterministic scan
after enough misses): simpler than an exact O(1) free-list per bucket, and
still never fires on the wrong site, at a known, honest cost — a bucket
that's extremely sparse relative to the whole patch can need many
attempts. The natural next optimization, if that ever matters in practice,
is an explicit per-bucket free-list; not built yet.

**Both engines share the on-disk `ReactionLutBlock` format** but interpret
it differently, so `reactions.lut` now starts with an 8-byte magic header
(`KMCSTAT1` for `--generate-lut`'s static LUTs, `KMCOCC01` for
`oc20_ingest`'s occupancy-gated ones) that `kinetica`'s `run()` reads to
pick the right engine automatically — no flag to remember or get out of
sync with how the file was actually built. For an occupancy-gated LUT,
`bin_id` means "which quantile bucket" (0..4) rather than "composition-
rejection magnitude class"; see `oc20_ingest`'s docs for where that bucket
index comes from.

**This intentionally didn't try to fix everything at once.** Occupancy-gating
changes *how* a reaction is selected and applied, not *what* chemistry the
underlying rate constants represent. The next two sections close two more
of the gaps this left open: gas-phase partial pressure, and the lattice's
real geometry / which real surface its rate constants are actually
measured on.

## Gas-phase pressure coupling

Before this, every run's adsorption kinetics were identical regardless of
feed-gas composition: `reactions.lut`'s rate constants encode a fixed
propensity per adsorption channel, with no notion of how much of each gas
is actually present above the surface. Two runs meant to represent, say, a
CO-rich feed versus an O2-rich one produced exactly the same coverage
trajectory.

`--pressure-o2`/`--pressure-h2`/`--pressure-co` (default `1.0` each) are
runtime multipliers, not LUT-baked constants — `occupancy::Pressures`
scales an adsorption template's propensity by the matching species'
relative pressure at the point `total_propensity`/`select_event` compute
live weights, alongside the existing `rate_q16 * live_count` factors. Nothing
about the LUT itself changes, so switching feed-gas composition between
runs is just a CLI flag, not a rebuild. Only adsorption is affected —
identifiable as a template whose reactant is `VACANT` (`reactant_mask ==
0`) — since desorption and bimolecular reactions don't consume a
gas-phase molecule in the first place; scaling their propensity by a
partial pressure wouldn't correspond to anything physical.

Verified against the real Pd(111) `reactions.lut` and release binary:
identical starting lattices, identical seeds, differing only in
`--pressure-co` — baseline (`1.0`) settles at 45.0% CO coverage; `20.0`
settles at 72.1%, with O and H coverage correspondingly displaced (all
three compete for the same finite pool of vacant sites). Zero invalid
occupancy states in either run.

## Lattice geometry and target surface: Pd(111)

The lattice is a real fcc(111) close-packed surface — six equidistant
nearest neighbors per site, not the generic square/4-neighbor grid earlier
versions used. `src/topology.rs` centralizes what "adjacent site" means
(previously duplicated, inconsistently, across `occupancy.rs` and
`engine.rs`) and models the hex grid as an offset-coordinate ("odd-r")
reinterpretation of the same flat row-major mmap — no storage change, just
row-parity-dependent column offsets for the diagonal neighbors, so every
row zig-zags into true close-packed alignment. `topology::all_neighbors`
returns up to six neighbors; `topology::forward_neighbors` returns the
canonical half (three) a full-grid scan uses to count every unordered
adjacent pair exactly once.

`reactions.lut`'s real data is now filtered to a single real
crystallographic surface — **Pd(111)** — instead of pooling DFT samples
from every metal/facet a species happens to appear on. `oc20_ingest
--metal Pd --facet 111` (see below) restricts monomolecular O/H/CO
adsorption data to that one surface, with a per-species fallback to
"Pd, any facet" logged explicitly if a species' facet-filtered pool is too
sparse to bucket meaningfully (see `oc20_ingest --help`).

**Real bimolecular (CO-oxidation, H2-recombination) barriers are absent
from a strict Pd(111) build — a genuine, checked finding, not an
oversight.** Tracing the exact `surfaceComposition`/`facet` metadata behind
the two previously-cited real barriers turned up:

- The CO-oxidation barrier (`StreibelMicrokinetic2021`, ~0.98–1.21 eV) is
  real and on pure Pd, but at facet **(211)**, not (111) — plus a second
  record on an oxide-modified `Pd+1:3O` surface (excluded outright by the
  pure-single-element metal filter, since it isn't elemental Pd).
- The H2-recombination barrier (~0.35 eV) is on `PdH`-hydride surfaces
  (`PdH-hcp-4layer`/`PdH-hcp-6layer`) at non-clean facet labels
  (`101-0.75MLfccH`, etc.) — a hydride, not clean metallic Pd.

Neither survives a filter that means what it says ("real Pd(111) metal").
So this build's `reactions.lut` has real, Pd(111)-specific monomolecular
O/H/CO adsorption/desorption chemistry and **zero bimolecular reactions**
— rather than quietly keeping the old cross-facet/cross-composition
bimolecular records under a "Pd" label that would itself be exactly the
kind of pooling this change exists to eliminate. Restoring bimolecular
chemistry on a real, single, chemically-clean surface is future work (see
"Next step" in the project handoff), not something this pass papers over.

## Building `reactions.lut` from real data

Two independent real-data sources feed the same `oc20_ingest` pipeline;
see the CO-gap note below for why you likely want both.

### OC20

OC20's IS2RE task publishes only relaxed adsorption energies (initial
structure guess → DFT-relaxed structure and energy), not transition-state
barriers or rate constants. `oc20_ingest` bridges that gap with two
standard computational-catalysis approximations:

1. **Bronsted-Evans-Polanyi (BEP) relation** — estimate an activation
   energy from a reaction energy: `E_a = max(0, alpha * dE_rxn + beta)`.
2. **Harmonic transition-state theory / Arrhenius** — convert that barrier
   into a rate constant: `k = nu * exp(-E_a / (kB * T))`.

### Prerequisites

`scripts/extract_energies.py` needs only two pip packages — no `torch` or
`torch_geometric` install required, since it reads the LMDB container
directly and unpickles each record with a stub `Unpickler` that discards
every torch-specific field it doesn't need:

```sh
pip install --user lmdb numpy
```

### Pipeline

1. Download the OC20 IS2RE bundle and the adsorbate-mapping file into
   `data/oc20/` (both untracked/gitignored):

   ```sh
   mkdir -p data/oc20
   curl -o data/oc20/is2res_train_val_test_lmdbs.tar.gz \
       https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz
   curl -o data/oc20/oc20_data_mapping.pkl \
       https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl
   ```

   The full tar is 8.1GB compressed; extract only the split you need
   rather than all of it (LMDB shard sizes below are real, not sparse):

   | Split (`train/`) | Compressed extract | Real size on disk |
   |-------------------|--------------------|--------------------|
   | `10k`             | fast                | 1.3GB              |
   | `100k`            | a few minutes       | 13GB               |
   | `all`             | slow                | 62GB               |

   ```sh
   tar -xzf data/oc20/is2res_train_val_test_lmdbs.tar.gz \
       -C data/oc20 is2res_train_val_test_lmdbs/data/is2re/100k/train/
   ```

   > **Filesystem note:** reading a `100k`-or-larger shard with `lmdb`'s
   > mmap while it sits on an NTFS mount (`ntfs3`/FUSE) can crash with a
   > `Bus error` (`SIGBUS`) partway through the scan. The `10k` split was
   > fine in place; for `100k`/`all`, copy the shard to a native Linux
   > filesystem (ext4, etc.) first and run the extraction from there.

2. Extract `(species, energy, sid)` records from the LMDB shard:

   ```sh
   python3 scripts/extract_energies.py \
       --lmdb data/oc20/is2res_train_val_test_lmdbs/data/is2re/100k/train/data.lmdb \
       --mapping data/oc20/oc20_data_mapping.pkl \
       --out data/oc20/energies_all.bin
   ```

3. Convert those energies into a `reactions.lut`:

   ```sh
   cargo run --release --bin oc20_ingest -- \
       --input data/oc20/energies_all.bin \
       --out reactions.lut
   ```

   `--alpha`, `--beta`, `--nu`, and `--temperature` override the BEP/TST
   defaults — see `oc20_ingest --help`.

`oc20_ingest` logs a per-species record count so any coverage gap is
visible rather than silent. In particular, OC20's `train`/`val` splits
contain **no `*CO` samples at all** — `*CO` is one of the benchmark's
deliberately held-out "unseen adsorbate" out-of-domain test classes, and
the `test_*` splits ship with `y_relaxed`/`y_init` withheld (`None`) to
prevent leaderboard cheating. So a `reactions.lut` built from OC20 alone
will always have real O and H adsorption/desorption chemistry but zero CO
reactions. (Checked directly: OC20's `100k` train split already contains
~97.5% of every O/H sample that exists anywhere in the full 460k-sample
`all` split — re-extracting a bigger split won't find more O/H data
either.)

### Alternative/supplemental source: Catalysis-Hub.org (fills the CO gap, and has some real barriers)

[Catalysis-Hub.org](https://catalysis-hub.org) is a separate, curated
database of DFT chemisorption/reaction energies across many publications,
queryable live over a public GraphQL API — no multi-GB download needed.
Unlike OC20 it has real `*CO` adsorption data:

```sh
python3 scripts/extract_catalysis_hub.py --out data/oc20/energies_catalysis_hub.bin
cargo run --release --bin oc20_ingest -- \
    --input data/oc20/energies_catalysis_hub.bin \
    --out reactions.lut
```

**Restricting to one real surface (`--metal`/`--facet`).** Both the
extraction script and `oc20_ingest` accept `--metal`/`--facet` filters —
pushed down to the GraphQL API as server-side `surfaceComposition`/`facet`
filter args on the extraction side, applied post-hoc (with a per-species
fallback to metal-only if `--facet` leaves too few samples to bucket) on
the ingest side. This is what this repo's own `reactions.lut` is built
from — see "Lattice geometry and target surface: Pd(111)" above:

```sh
python3 scripts/extract_catalysis_hub.py \
    --out data/oc20/energies_pd111.bin \
    --metal Pd --facet 111
cargo run --release --bin oc20_ingest -- \
    --input data/oc20/energies_pd111.bin \
    --metal Pd --facet 111 \
    --out reactions.lut
```

`--metal` matches against a *pure single element* formula (via
`oc20e_format.parse_pure_metal`) — an alloy, intermetallic, oxide, or
hydride surface (e.g. `PdPt`, `Pd+1:3O`, `PdH`) is excluded, not folded in
under the pure metal's name, even though the database's own
`surfaceComposition` string sometimes does exactly that. `--facet` matches
a plain decimal Miller-index string (`"111"`); facet labels with suffixes
(`"110-lc-Ovac"`) or non-numeric content don't match anything and are
tagged "unknown" (`facet_code` returns 0) rather than silently
mis-parsed. Sample counts at this level of filtering are small (tens, not
thousands, per species) — `oc20_ingest`'s per-species fallback log line
makes it visible whenever a species had to broaden from "this facet" to
"this metal, any facet" to reach `BUCKETS_PER_SPECIES` (4) samples.

`scripts/extract_catalysis_hub.py` runs two passes. The bulk sweep
paginates the API for the elementary adsorption step
`star + <gas> -> <adsorbate>star` per species (`O2gas` at 0.5
stoichiometry, `H2gas` at 0.5, `COgas` at 1.0), keeping only exact matches
(no co-adsorbates or lumped multi-step reactions) — reaction energies
only, same as OC20, so `oc20_ingest` still applies BEP to these. Uses only
the Python standard library (`urllib`, `json`, `base64`) — no pip install
needed, unlike the OC20 path. A full run (no `--limit-per-species`) takes
a few minutes and yields on the order of 1-1.5k O, 8-12k H, and 5-6k CO
real reactions (varies run to run — this is a live, growing database).

**A small second pass finds genuine transition-state barriers.** Most
entries in this database, like OC20, only have relaxation/adsorption
energies — the schema's `activationEnergy` field is `null` for the
overwhelming majority of its 158k+ reactions. But a handful of
publications (`FalsigOn2014`, `WangUniversal2011`, `CatappTrends2008`,
`JiangTrends2009`, and others) *do* report real NEB/dimer-method barriers
for O₂, H₂, and CO dissociative/molecular adsorption across several metal
surfaces — around 40 records as of this writing. `oc20_ingest` uses these
directly for the forward (adsorption) direction instead of the BEP
estimate, deriving the reverse (desorption) barrier from the same
thermodynamic-consistency relation (`Ea_rev = Ea_fwd - dE_rxn`) it always
uses. It logs how many records per species carried a real barrier so this
is visible, e.g.:

```
oc20_ingest: species O: 1368 adsorption-energy records  (15 with a real DFT-computed activation energy, not BEP)
oc20_ingest: species O: collapsed into 4 quantile bucket(s) (sizes: 342, 342, 342, 342)
```

That second line is `bucket_by_quantile` at work: `oc20_ingest` doesn't
build one `ReactionRecord` per DFT sample (hundreds per species) — it
sorts a species' samples by reaction energy, splits them into
`BUCKETS_PER_SPECIES` (4) roughly-equal groups, and collapses each group
into one representative adsorption/desorption template built from the
group's mean energy (mean real Ea too, if any group member has one). This
is what makes occupancy-gating (above) tractable: a handful of templates
per species, each genuinely tied to a live count of matching lattice
sites, rather than hundreds of statically-always-available channels with
no notion of the lattice's current state. It also keeps real
heterogeneity — a genuinely fast-reacting quartile of surfaces versus a
genuinely slow-reacting one — rather than washing everything out into one
averaged number.

**There's also real barrier data for actual bimolecular surface reactions
like CO oxidation itself, `O* + CO* -> CO2 + 2*`** — see "Bimolecular
reactions" below for how this is now wired all the way through.

### Bimolecular reactions

Each `ReactionLutBlock` lane carries a second `(reactant_mask << 4) |
product_mask)` transition (`transition_b`) plus an `is_bimolecular` flag,
alongside the original single-site `transition_a`. A monomolecular
(adsorption/desorption) reaction only ever touches `transition_a`'s site,
exactly as before. How the two sites get *chosen* differs by engine (see
"Occupancy-gated kMC" above):

- **Static** (`--generate-lut` demo data): site A is a uniformly random
  patch site; site B is one of its same-patch hex-topology neighbors (see
  `topology::all_neighbors`), picked without checking either site's actual
  occupancy -- fine for exercising the architecture, not meant to be
  chemically exact.
- **Occupancy-gated** (real, `oc20_ingest`-built data): site A is
  rejection-sampled until it genuinely holds the reaction's first
  reactant species; site B is deterministically checked among site A's
  (up to six) neighbors for the second reactant species, retrying the
  whole pair on a miss. A bimolecular reaction can only ever fire on a
  site pair that actually has both reactants present.

Both constrain site B to the *same patch* as site A (rather than possibly
crossing into a neighboring `rayon` task's row band), so both sites update
as a single atomic step in that patch's own trajectory with no cross-
thread synchronization needed. If site A has no same-patch neighbor at all
(a degenerate 1×1 patch), the event is skipped rather than forced onto an
invalid site.

`kinetica --generate-lut` synthesizes roughly 1 in 8 demo reactions as
bimolecular so the static path is exercised even without real data on
hand.

**Dissociative adsorption (O2, H2).** O2 and H2 adsorb *dissociatively* —
the real elementary step is `2* + O2(g) -> 2 O*` / `2* + H2(g) -> 2 H*`,
consuming two adjacent sites at once, not one. Earlier versions modeled
this as two independent monomolecular events (`VACANT -> O*` twice),
energetically correct per atom (the extracted DFT energy already uses the
right 0.5-stoichiometry-per-atom convention) but kinetically wrong — two
single-site channels have no notion that they're really one two-site
event, so there's no site-pair-blocking coverage dependence: a lattice
running out of *adjacent* vacant pairs (as opposed to just vacant sites)
kept adsorbing at the same rate. `oc20_ingest` now builds these species'
adsorption as genuine `is_bimolecular` records instead — same reaction
data and BEP/Arrhenius model as before, just correctly gated on
`occupancy::OccupancyCounters::vacant_pairs` (a live count of adjacent
*vacant* pairs, shared by both species' templates the same way
`vacant_count` is already shared across monomolecular adsorption
templates). CO adsorbs molecularly (one site) and is unaffected —
`oc20_ingest`'s `DISSOCIATIVE_SPECIES` names exactly the two species this
applies to. Desorption is untouched for both: this only corrects the
adsorption direction. Pressure-coupled the same as monomolecular
adsorption (see "Gas-phase pressure coupling" above) — verified against
the real Pd(111) `reactions.lut`: `--pressure-h2 15.0` raises H coverage
from 10.8% to 29.2%, zero invalid occupancy states.

**`oc20_ingest` can also build real bimolecular reactions from
Catalysis-Hub data — though not, currently, for the Pd(111)-filtered build
this repo ships (see "Lattice geometry and target surface: Pd(111)" above
for why: the real barriers below turned out to live on Pd(211)/oxide/
hydride surfaces once their exact facet/composition was checked, not on
clean Pd(111)).** The mechanism itself is real and tested — it's the
*intersection* with a strict single-surface filter that's currently empty,
not a gap in the engine or the ingest pipeline.
`scripts/extract_catalysis_hub.py`'s real-barrier sweep picks out genuine
two-site recombination barriers (`BIMOLECULAR_PATTERNS`) and writes them
to a second, parallel binary file (`OC20BI02` format — see
`scripts/oc20e_format.py`) via `--bimolecular-out`. Pass that file to
`oc20_ingest --bimolecular-input` (using an unfiltered or differently-
filtered extraction — e.g. no `--metal`/`--facet`, or `--metal Pd --facet
211` for the real CO-oxidation barrier specifically):

```sh
python3 scripts/extract_catalysis_hub.py \
    --out data/oc20/energies_catalysis_hub.bin \
    --bimolecular-out data/oc20/energies_catalysis_hub_bimolecular.bin
cargo run --release --bin oc20_ingest -- \
    --input data/oc20/energies_catalysis_hub.bin \
    --bimolecular-input data/oc20/energies_catalysis_hub_bimolecular.bin \
    --out reactions.lut
```

Unlike the monomolecular adsorption/desorption pair, a bimolecular record
only ever builds a *single* forward `ReactionRecord` — there's no BEP
fallback for a two-species step (a bimolecular record is only emitted
when Catalysis-Hub reports a real activation energy), and no reverse
reaction is derived: the gas product leaving the surface isn't the
reverse of a single elementary step back onto two sites, so there's no
thermodynamically meaningful `Ea_rev` the way there is for
adsorption/desorption. These real barriers are rare and vary run to run
just like the rest of this live, growing database (see above) — a
record you know exists (e.g. by sid) may not show up on a given run's
cursor pagination, so if you're chasing a specific one, try again —
currently two patterns are matched:

- **`O* + CO* -> CO2 + 2*`** (e.g. `StreibelMicrokinetic2021`,
  "Microkinetic Modeling of Propene Combustion" — real NEB barriers of
  ~0.98-1.21 eV on Pd).
- **`2 H* -> H2 + 2*`** (e.g. "Dynamics and Hysteresis of Hydrogen
  Interaction..." — ~0.35 eV). This one is *homoatomic* (both sites are
  the same species), and `oc20_ingest` treats that specially: it
  **replaces** H's monomolecular desorption reaction rather than adding
  a third rate channel alongside it. The existing monomolecular H
  adsorption/desorption pair already approximates H2 dissociative
  adsorption/associative desorption as a single-site event (using half
  the H2 dissociation energy per H atom — see `SPECIES_PATTERNS`); a
  real two-site measurement of the same physical recombination event
  isn't a *second* reaction, it's a more accurate model of the same one.
  Building both would just split one real process's propensity across
  two channels at different levels of approximation. H's monomolecular
  *adsorption* is untouched — these barriers say nothing about the
  adsorption direction, only the (already-known-to-be-two-site)
  recombinative desorption.

`oc20_ingest` logs which species had their desorption replaced this way,
and folds every rate (mono- and bimolecular alike) into the same Q16.16
rescaling pass, so the single fastest reaction overall still anchors the
scale factor.

**What was checked and doesn't apply here:** a full scan of every real
DFT-barrier reaction in the database (1000+ records) involving two of
our three tracked species turned up one more candidate, `H* + CO* ->
CHO*` (~0.97 eV) — but its product is an adsorbed formyl species
(`CHOstar`), not a gas + two vacant sites, so it can't be represented
without a fourth adsorbate bit in `layout.rs`'s occupancy mask. No
genuine two-site O2 recombination record with a real barrier exists in
the data either (only the same single-site 0.5-stoichiometry pattern
already covered by the monomolecular path). So within the current
three-species (O/H/CO) model, CO oxidation and H2 recombination are the
only two real bimolecular reactions the live data actually supports (on
*some* metal/facet — see the Pd(111)-specific finding above for why
neither currently makes it into this repo's own single-surface build).

`data/` (dataset downloads/extractions) and `PROMPT.md` are intentionally
untracked — see `.gitignore`. `scripts/extract_energies.py`,
`scripts/extract_catalysis_hub.py`, and `scripts/oc20e_format.py` (the
shared binary format both scripts write) are all tracked; only the
downloads, extractions, and generated `.bin` files under `data/` are not.
