# kinetica

Asynchronous, out-of-core lattice kinetic Monte Carlo (kMC) engine for
single-workstation, OC20-scale surface reaction simulation, written in
pure Rust.

The lattice lives as a memory-mapped file rather than in heap-resident
`Vec`s, next-reaction selection is O(1) regardless of how many reaction
channels are active (composition-rejection sampling over fixed-point
propensities), and the simulation is spatially decomposed into
independently-scheduled `rayon` work-stealing patches whose fired-reaction
trajectory is streamed out through a double-buffered `io_uring` writer so
no compute thread ever blocks on disk I/O.

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
| `src/layout.rs`          | Bit-packed mmap'd lattice, cache-line-aligned `ReactionLutBlock` reaction table, LUT packing/writing |
| `src/gillespie.rs`       | O(1) partial-propensity composition-rejection (SSA-CR) reaction sampler, fixed-point propensity arithmetic |
| `src/engine.rs`          | Spatial domain decomposition, rayon work-stealing patches, crossbeam boundary migration, double-buffered `io_uring` trajectory writer |
| `src/lib.rs`             | Library surface shared by `kinetica` and auxiliary tools              |
| `src/main.rs`            | `kinetica` CLI entrypoint                                              |
| `src/bin/oc20_ingest.rs` | Builds `reactions.lut` from real OC20 IS2RE adsorption-energy data     |

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

## Building `reactions.lut` from real OC20 data

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
prevent leaderboard cheating. So a `reactions.lut` built this way will
always have real O and H adsorption/desorption chemistry but zero CO
reactions — there is currently no real-energy source for CO anywhere in
this dataset bundle.

`data/` (dataset downloads/extractions) and `PROMPT.md` are intentionally
untracked — see `.gitignore`. `scripts/extract_energies.py` is tracked;
only the OC20 downloads/extractions and generated `.bin` files under
`data/` are not.
