"""Shared binary format for kinetica's `oc20_ingest` tool, written by both
`extract_energies.py` (OC20) and `extract_catalysis_hub.py`
(Catalysis-Hub.org) so `oc20_ingest` consumes either source unchanged.

Format (little-endian):
    magic:        8 bytes  b"OC20E002"
    record_count: u32
    records[count]:
        species:      u8   (0 = O, 1 = H, 2 = CO)
        energy_mev:   i32  (reaction/adsorption energy, milli-eV)
        sid:          u32  (source system/reaction id, for traceability only)
        has_real_ea:  u8   (1 if `real_ea_mev` is a real DFT-computed
                            activation energy rather than a BEP estimate
                            oc20_ingest should derive; 0 otherwise)
        real_ea_mev:  i32  (meaningful only when has_real_ea == 1)

v2 (OC20E002) adds `has_real_ea`/`real_ea_mev` on top of v1 (OC20E001),
which was just (species, energy_mev, sid). Real activation energies are
rare (most surface-chemistry databases, OC20 included, only publish
relaxation/adsorption energies, not transition-state barriers) but do
exist for a handful of elementary steps -- see
`extract_catalysis_hub.py`'s `fetch_real_barrier_records`.

A second, parallel format (`OC20BI01`) carries two-species *bimolecular*
records (e.g. Langmuir-Hinshelwood surface reactions like
`O* + CO* -> CO2 + 2*`) -- structurally different from the single-species
records above (two species indices instead of one, and no BEP fallback:
a bimolecular record is only ever emitted when a real DFT-computed
activation energy exists, since there is no bimolecular BEP relation in
this tool). Kept as a separate file/format rather than folded into
OC20E002 so the common single-species path never has to reason about an
optional second species field it doesn't use.

Format (little-endian):
    magic:        8 bytes  b"OC20BI01"
    record_count: u32
    records[count]:
        species_a:    u8   (0 = O, 1 = H, 2 = CO -- same indices as above)
        species_b:    u8
        energy_mev:   i32  (reaction energy, milli-eV; kept for
                            diagnostics/thermodynamic bookkeeping, not
                            currently used to derive a reverse reaction)
        sid:          u32  (source system/reaction id, for traceability only)
        ea_mev:       i32  (real DFT-computed forward activation energy,
                            milli-eV -- always meaningful; there is no
                            has_real_ea flag because this format only ever
                            carries real barriers)
"""

import struct

MAGIC = b"OC20E002"
RECORD_STRUCT = "<BiIBi"  # species, energy_mev, sid, has_real_ea, real_ea_mev
RECORD_SIZE = struct.calcsize(RECORD_STRUCT)

MAGIC_BIMOLECULAR = b"OC20BI01"
RECORD_STRUCT_BIMOLECULAR = "<BBiIi"  # species_a, species_b, energy_mev, sid, ea_mev
RECORD_SIZE_BIMOLECULAR = struct.calcsize(RECORD_STRUCT_BIMOLECULAR)


def write_records(records, out_path):
    """`records`: iterable of (species, energy_mev, sid, has_real_ea, real_ea_mev)."""
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(records)))
        for species, energy_mev, sid, has_real_ea, real_ea_mev in records:
            f.write(
                struct.pack(
                    RECORD_STRUCT,
                    species,
                    energy_mev,
                    sid & 0xFFFFFFFF,
                    1 if has_real_ea else 0,
                    real_ea_mev,
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
    """`records`: iterable of (species_a, species_b, energy_mev, sid, ea_mev)."""
    with open(out_path, "wb") as f:
        f.write(MAGIC_BIMOLECULAR)
        f.write(struct.pack("<I", len(records)))
        for species_a, species_b, energy_mev, sid, ea_mev in records:
            f.write(
                struct.pack(
                    RECORD_STRUCT_BIMOLECULAR,
                    species_a,
                    species_b,
                    energy_mev,
                    sid & 0xFFFFFFFF,
                    ea_mev,
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
