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
"""

import struct

MAGIC = b"OC20E002"
RECORD_STRUCT = "<BiIBi"  # species, energy_mev, sid, has_real_ea, real_ea_mev
RECORD_SIZE = struct.calcsize(RECORD_STRUCT)


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
