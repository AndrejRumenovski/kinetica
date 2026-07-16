"""Shared binary format for kinetica's `oc20_ingest` tool, written by both
`extract_energies.py` (OC20) and `extract_catalysis_hub.py`
(Catalysis-Hub.org) so `oc20_ingest` consumes either source unchanged.

Format (little-endian):
    magic:        8 bytes  b"OC20E003"
    record_count: u32
    records[count]:
        species:      u8   (0 = O, 1 = H, 2 = CO, 3 = OH -- OH is sourced
                            entirely from the bimolecular water-splitting
                            reaction below, never from a record in this
                            format, but shares the same index space)
        energy_mev:   i32  (reaction/adsorption energy, milli-eV)
        sid:          u32  (source system/reaction id, for traceability only)
        has_real_ea:  u8   (1 if `real_ea_mev` is a real DFT-computed
                            activation energy rather than a BEP estimate
                            oc20_ingest should derive; 0 otherwise)
        real_ea_mev:  i32  (meaningful only when has_real_ea == 1)
        metal:        u8   (index into METALS below; 0 = unknown/not one
                            of the metals this pipeline tracks)
        facet:        u16  (Miller-index digits as a decimal number, e.g.
                            111 for (111); 0 = unknown/unparseable facet)

v3 (OC20E003) adds `metal`/`facet` on top of v2 (OC20E002), so
`oc20_ingest --metal --facet` can filter real DFT samples down to one
real crystallographic surface instead of pooling every metal/facet a
species happens to appear on on top of one idealized generic lattice --
see kinetica's chemistry-review artifact and `src/topology.rs`'s switch
to a real fcc(111) hex geometry, which this data now targets. v2 added
`has_real_ea`/`real_ea_mev` on top of v1 (OC20E001), which was just
(species, energy_mev, sid).

A second, parallel format (`OC20BI03`) carries two-species *bimolecular*
records (e.g. Langmuir-Hinshelwood surface reactions like
`O* + CO* -> CO2 + 2*`, or dissociative adsorption like
`2* + H2O(g) -> H* + OH*`) -- structurally different from the
single-species records above (two species indices instead of one, and no
BEP fallback: a bimolecular record is only ever emitted when a real
DFT-computed activation energy exists, since there is no bimolecular BEP
relation in this tool). Kept as a separate file/format rather than folded
into OC20E003 so the common single-species path never has to reason about
an optional second species field it doesn't use.

Format (little-endian):
    magic:            8 bytes  b"OC20BI03"
    record_count:     u32
    records[count]:
        species_a:        u8   (same species indices as above)
        species_b:        u8
        energy_mev:       i32  (forward reaction energy, milli-eV --
                                meaningful, and used, when
                                is_dissociative == 1: it derives the
                                reverse rate the same
                                thermodynamic-consistency way the
                                single-species adsorption/desorption pair
                                does)
        sid:              u32  (source system/reaction id, traceability only)
        ea_mev:           i32  (real DFT-computed forward activation
                                energy, milli-eV -- always meaningful;
                                there is no has_real_ea flag because this
                                format only ever carries real barriers)
        metal:            u8   (same METALS index convention as above)
        facet:            u16  (same Miller-index convention as above)
        is_dissociative:  u8   (0 = recombination direction: both sites
                                start occupied, clear to vacant, release a
                                gas product -- forward-only, e.g. CO
                                oxidation, H2 recombination. 1 =
                                dissociative-adsorption direction: both
                                sites start vacant, fill from a gas
                                reactant -- built both directions, e.g.
                                water splitting)

v3 (OC20BI03) adds `is_dissociative` on top of v2 (OC20BI02), so the same
format can carry a real barrier measured in *either* direction rather than
always assuming recombination -- see `oc20_ingest.rs`'s `BiEnergyRecord`
doc comment for why the two directions need different treatment. v2 added
`metal`/`facet` on top of v1 (OC20BI01).
"""

import re
import struct

# Metal index table shared with `oc20_ingest.rs`'s own `METALS` constant --
# index 0 is reserved for "not one of the metals this pipeline tracks",
# still stored (never silently dropped at extraction time), just excluded
# by any `--metal` filter downstream. Extend this list, and the matching
# Rust one, together -- the numeric index is what's on disk, not the
# string, so the two must stay in lockstep.
METALS = [
    "unknown",
    "Pd",
    "Pt",
    "Cu",
    "Ni",
    "Rh",
    "Ru",
    "Ag",
    "Au",
    "Fe",
    "Co",
    "Ir",
    "Os",
]


def metal_index(symbol):
    """Index of `symbol` in `METALS`, or 0 ("unknown") if it isn't one of
    the metals this pipeline tracks."""
    return METALS.index(symbol) if symbol in METALS else 0


_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_pure_metal(formula):
    """The single element symbol in a chemical formula string (e.g.
    `"Pd4"` -> `"Pd"`), or `None` if `formula` names more than one
    distinct element (an alloy/intermetallic/compound bulk) -- this
    pipeline only tracks facet-specific data for one pure metal at a
    time, so a mixed bulk can't be attributed to a single `--metal`
    filter value."""
    elements = {sym for sym, _count in _ELEMENT_RE.findall(formula or "") if sym}
    if len(elements) == 1:
        return next(iter(elements))
    return None


def facet_code(facet_str):
    """A `facet` string (e.g. `"111"`) to the `u16` encoding this format
    stores, or 0 ("unknown") if it isn't a plain decimal Miller-index
    string (some database entries use suffixed/non-numeric facet labels
    like `"110-lc-Ovac"`, which this simple encoding can't represent)."""
    if facet_str and facet_str.isdigit():
        try:
            return int(facet_str) & 0xFFFF
        except ValueError:
            return 0
    return 0


def miller_facet_string(miller_index):
    """A 3-tuple Miller index (e.g. `(1, 1, 1)`) to the same decimal-digit
    facet-string convention `facet_code` expects (e.g. `"111"`), or `None`
    if any index has magnitude >= 10 (can't be represented as a single
    digit by this simple encoding -- rare in practice)."""
    if miller_index is None or any(abs(m) >= 10 for m in miller_index):
        return None
    digits = sorted((abs(m) for m in miller_index), reverse=True)
    return "".join(str(d) for d in digits)


MAGIC = b"OC20E003"
RECORD_STRUCT = "<BiIBiBH"  # species, energy_mev, sid, has_real_ea, real_ea_mev, metal, facet
RECORD_SIZE = struct.calcsize(RECORD_STRUCT)

MAGIC_BIMOLECULAR = b"OC20BI03"
# species_a, species_b, energy_mev, sid, ea_mev, metal, facet, is_dissociative
RECORD_STRUCT_BIMOLECULAR = "<BBiIiBHB"
RECORD_SIZE_BIMOLECULAR = struct.calcsize(RECORD_STRUCT_BIMOLECULAR)


def write_records(records, out_path):
    """`records`: iterable of (species, energy_mev, sid, has_real_ea,
    real_ea_mev, metal, facet)."""
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(records)))
        for species, energy_mev, sid, has_real_ea, real_ea_mev, metal, facet in records:
            f.write(
                struct.pack(
                    RECORD_STRUCT,
                    species,
                    energy_mev,
                    sid & 0xFFFFFFFF,
                    1 if has_real_ea else 0,
                    real_ea_mev,
                    metal,
                    facet,
                )
            )


def read_records(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError(f"not an {MAGIC!r} file")
    count = struct.unpack_from("<I", data, 8)[0]
    records = []
    offset = 12
    for _ in range(count):
        records.append(struct.unpack_from(RECORD_STRUCT, data, offset))
        offset += RECORD_SIZE
    return records


def write_bimolecular_records(records, out_path):
    """`records`: iterable of (species_a, species_b, energy_mev, sid,
    ea_mev, metal, facet, is_dissociative)."""
    with open(out_path, "wb") as f:
        f.write(MAGIC_BIMOLECULAR)
        f.write(struct.pack("<I", len(records)))
        for species_a, species_b, energy_mev, sid, ea_mev, metal, facet, is_dissociative in records:
            f.write(
                struct.pack(
                    RECORD_STRUCT_BIMOLECULAR,
                    species_a,
                    species_b,
                    energy_mev,
                    sid & 0xFFFFFFFF,
                    ea_mev,
                    metal,
                    facet,
                    1 if is_dissociative else 0,
                )
            )


def read_bimolecular_records(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC_BIMOLECULAR:
        raise ValueError(f"not an {MAGIC_BIMOLECULAR!r} file")
    count = struct.unpack_from("<I", data, 8)[0]
    records = []
    offset = 12
    for _ in range(count):
        records.append(struct.unpack_from(RECORD_STRUCT_BIMOLECULAR, data, offset))
        offset += RECORD_SIZE_BIMOLECULAR
    return records
