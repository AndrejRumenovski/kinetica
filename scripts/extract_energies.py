"""One-time extraction: OC20 IS2RE LMDB shards -> a flat binary of
(species, energy_mev, sid) records for kinetica's oc20_ingest tool.

Note: OC20's `train`/`val` splits do not include `*CO` at all -- it is
held out entirely as an "unseen adsorbate" out-of-domain test class, and
the `test_*` splits ship with `y_relaxed`/`y_init` withheld (both `None`)
to prevent leaderboard cheating. Extracting from `train` will therefore
always yield 0 CO records; `oc20_ingest` logs this per-species so it's
visible rather than a silent gap.

Deliberately minimal: reads the LMDB container directly and unpickles each
torch_geometric Data object with a stub Unpickler that never needs torch or
torch_geometric installed (see `AnyStub` below) -- we only care about two
plain-Python scalar attributes (`sid`, `y_relaxed`), every torch Tensor
attribute on the object is discarded immediately.

Output format: see oc20e_format.py. OC20 never provides real
transition-state barriers, so every record here has `has_real_ea=False`.
"""

import argparse
import io
import pickle
import sys

import lmdb

from oc20e_format import facet_code, metal_index, miller_facet_string, parse_pure_metal, write_records

# OC20's global adsorbate-index table (mapping_adsorbates_2020may12.txt)
# assigns these three indices to exactly the three adsorbates kinetica's
# bitflags (layout.rs) model: *O, *H, *CO.
ADS_ID_TO_SPECIES = {0: 0, 1: 1, 5: 2}  # O -> 0, H -> 1, CO -> 2


class AnyStub:
    """Swallows any torch/torch_geometric class or reduce-function this
    process doesn't have installed, keeping only what plain pickle opcodes
    (dicts, floats, ints, strings) already reconstructed natively."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return AnyStub()

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple):
            for part in state:
                if isinstance(part, dict):
                    self.__dict__.update(part)

    def __reduce__(self):
        return (AnyStub, ())


class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("torch"):
            return AnyStub
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return None


def load_sid_map(mapping_path):
    """`{sid: (species, metal_idx, facet_val)}`.

    `metal`/`facet` are best-effort: OC20's bulks are frequently
    multi-element alloys/intermetallics (`bulk_symbols` like `"RuTa"`),
    not a single pure metal, and `miller_index` can have a component with
    magnitude >= 10 that this pipeline's simple single-digit facet
    encoding can't represent -- both cases fall back to "unknown" (0)
    rather than being dropped, so this source's coverage stats stay
    accurate even where the facet-level detail isn't available.
    """
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    sid_to_meta = {}
    for key, meta in mapping.items():
        species = ADS_ID_TO_SPECIES.get(meta.get("ads_id"))
        if species is None:
            continue
        # keys look like "random<sid>"
        sid = int(key[len("random"):])
        metal_symbol = parse_pure_metal(meta.get("bulk_symbols"))
        facet_str = miller_facet_string(meta.get("miller_index"))
        sid_to_meta[sid] = (
            species,
            metal_index(metal_symbol) if metal_symbol else 0,
            facet_code(facet_str) if facet_str else 0,
        )
    return sid_to_meta


def extract(lmdb_path, sid_to_meta, out):
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, max_readers=1)
    records = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for _key, value in cursor:
            obj = StubUnpickler(io.BytesIO(value)).load()
            sid = obj.__dict__.get("sid")
            meta = sid_to_meta.get(sid)
            if meta is None:
                continue
            species, metal_idx, facet_val = meta
            energy_ev = obj.__dict__.get("y_relaxed")
            if energy_ev is None:
                continue
            energy_mev = int(round(energy_ev * 1000.0))
            records.append((species, energy_mev, sid, False, 0, metal_idx, facet_val))
    env.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb", required=True, help="path to a data.lmdb shard")
    parser.add_argument("--mapping", required=True, help="path to oc20_data_mapping.pkl")
    parser.add_argument("--out", required=True, help="output flat binary path")
    args = parser.parse_args()

    sid_to_meta = load_sid_map(args.mapping)
    print(f"loaded {len(sid_to_meta)} target sids from mapping", file=sys.stderr)

    records = extract(args.lmdb, sid_to_meta, args.out)
    print(f"matched {len(records)} records in {args.lmdb}", file=sys.stderr)

    write_records(records, args.out)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
