"""Hand-rolled parser for kinetica's sectioned config file format -- the
Python counterpart to `src/config.rs`'s `SimConfig::parse`, so
`extract_energies.py`/`extract_catalysis_hub.py` can drive their
per-species pattern dicts from the same `--config <PATH>` file
`oc20_ingest` reads, instead of each hardcoding its own parallel copy (the
"keep these five files in sync by hand-comment convention" problem
`src/config.rs`'s own doc comment describes).

Mirrors the Rust parser field-for-field, section-for-section,
error-style-for-error-style -- see `src/config.rs`'s module doc comment
for the full file format description and an example. Zero new
dependencies (stdlib only): `namedtuple` instead of Rust's `struct`,
`ValueError` instead of `Result<_, String>`, same reasoning as this
project's existing Python scripts (which only reach for `lmdb`/`urllib`
where genuinely needed) and the Rust side's own no-`serde` convention.

Only the fields the Python extraction scripts actually consume matter
here (gas, stoich, product, oc20_ads_id, bimolecular gas) -- `metal`/
`facet`/BEP defaults are parsed too (this is one shared file, read by
both languages), but only `oc20_ingest.rs` (Rust) turns them into rate
constants; nothing on the Python side needs to.
"""

from collections import namedtuple

# One `[species]` row: a single adsorbate's identity and how
# `oc20_ingest`/the Python extractors should treat it. `role` is one of
# "molecular" | "dissociative" | "product_only" (see `src/config.rs`'s
# `SpeciesRole` for what each means).
SpeciesEntry = namedtuple(
    "SpeciesEntry", ["name", "bit", "gas", "stoich", "product", "role", "oc20_ads_id"]
)

# One `[bimolecular]` row: a two-site reaction between two already-declared
# `[species]` entries. `direction` is one of "recombination" |
# "dissociative" (see `src/config.rs`'s `BimolecularDirection`).
BimolecularEntry = namedtuple(
    "BimolecularEntry", ["key", "species_a", "species_b", "direction", "gas"]
)

# A fully parsed config file. `species`/`bimolecular` are in declaration
# order -- a species' *row order* is its index into
# `layout::SPECIES_BITS`-equivalent arrays, same as the Rust side.
SimConfig = namedtuple(
    "SimConfig",
    [
        "metal",
        "facet",
        "alpha",
        "beta_ev",
        "nu",
        "temperature_k",
        "species",
        "bimolecular",
    ],
)

# The one-hot-byte-in-a-u16-mask packing `layout::SPECIES_BITS`'s doc
# comment describes caps any species list at 8 -- mirrored here so a
# malformed config fails at parse time with a clear message instead of
# only failing later when `oc20_ingest.rs` rejects the same file.
MAX_SPECIES = 8


def species_index(config, name):
    """Row-order index of the species named `name` in `config.species`, or
    `None` if no such species is declared. The Python counterpart to
    looking a species name up in `layout::SpeciesTable` on the Rust side,
    except here it's against the *config* (build-time), not a *built LUT*
    (runtime)."""
    for i, entry in enumerate(config.species):
        if entry.name == name:
            return i
    return None


def _strip_comment(line):
    return line.split("#", 1)[0]


def _parse_opt(field):
    """`-` means "no value" for an optional field; anything else is that
    field's value verbatim."""
    return None if field == "-" else field


def _parse_bit(field, line_no, species_name):
    """Accepts a plain decimal byte or a `0x`-prefixed hex byte, and
    rejects anything that isn't a one-hot bit -- the same requirement
    `layout::SpeciesTable::new` enforces on the Rust side."""
    try:
        value = int(field, 16) if field.startswith("0x") else int(field, 10)
    except ValueError:
        value = None
    if value is None or not (0 <= value <= 0xFF):
        raise ValueError(
            f"line {line_no}: species `{species_name}`'s bit `{field}` isn't a valid byte "
            "(decimal or 0x-prefixed hex)"
        )
    if bin(value).count("1") != 1:
        raise ValueError(
            f"line {line_no}: species `{species_name}`'s bit {value:#04x} is not one-hot"
        )
    return value


def _parse_species_row(name, value, line_no):
    fields = [f.strip() for f in value.split(",")]
    if len(fields) != 6:
        raise ValueError(
            f"line {line_no}: species `{name}` expects 6 comma-separated fields "
            f"(bit, gas, stoich, product, role, oc20_ads_id), got {len(fields)}"
        )
    bit_field, gas_field, stoich_field, product_field, role_field, ads_id_field = fields

    bit = _parse_bit(bit_field, line_no, name)
    gas = _parse_opt(gas_field)

    stoich_str = _parse_opt(stoich_field)
    stoich = None
    if stoich_str is not None:
        try:
            stoich = float(stoich_str)
        except ValueError:
            raise ValueError(
                f"line {line_no}: species `{name}`'s stoich expects a number, got `{stoich_str}`"
            )

    product = _parse_opt(product_field)

    if role_field not in ("molecular", "dissociative", "product_only"):
        raise ValueError(
            f"line {line_no}: species `{name}`'s role `{role_field}` isn't one of "
            "molecular/dissociative/product_only"
        )

    ads_id_str = _parse_opt(ads_id_field)
    oc20_ads_id = None
    if ads_id_str is not None:
        try:
            oc20_ads_id = int(ads_id_str)
        except ValueError:
            raise ValueError(
                f"line {line_no}: species `{name}`'s oc20_ads_id expects an integer, "
                f"got `{ads_id_str}`"
            )

    return SpeciesEntry(
        name=name,
        bit=bit,
        gas=gas,
        stoich=stoich,
        product=product,
        role=role_field,
        oc20_ads_id=oc20_ads_id,
    )


def _parse_bimolecular_row(key, value, line_no):
    fields = [f.strip() for f in value.split(",")]
    if len(fields) != 4:
        raise ValueError(
            f"line {line_no}: bimolecular entry `{key}` expects 4 comma-separated fields "
            f"(species_a, species_b, direction, gas), got {len(fields)}"
        )
    species_a, species_b, direction, gas_field = fields
    if direction not in ("recombination", "dissociative"):
        raise ValueError(
            f"line {line_no}: bimolecular entry `{key}`'s direction `{direction}` isn't one "
            "of recombination/dissociative"
        )
    return BimolecularEntry(
        key=key,
        species_a=species_a,
        species_b=species_b,
        direction=direction,
        gas=_parse_opt(gas_field),
    )


def _validate_species(species):
    if len(species) > MAX_SPECIES:
        raise ValueError(
            f"{len(species)} species exceeds the architectural ceiling of {MAX_SPECIES} "
            "(see layout::SPECIES_BITS's doc comment for why)"
        )
    seen_names = set()
    seen_bits = set()
    for entry in species:
        if entry.name in seen_names:
            raise ValueError(f"species name `{entry.name}` is declared more than once")
        if entry.bit in seen_bits:
            raise ValueError(f"species bit {entry.bit:#04x} is declared more than once")
        seen_names.add(entry.name)
        seen_bits.add(entry.bit)


def _validate_bimolecular(bimolecular, species):
    names = {entry.name for entry in species}
    for entry in bimolecular:
        for name in (entry.species_a, entry.species_b):
            if name not in names:
                raise ValueError(
                    f"bimolecular entry `{entry.key}` references undeclared species `{name}`"
                )


def parse_config(text):
    """Parse `text` (a config file's full contents) into a `SimConfig`, or
    raise `ValueError` naming the offending line -- see this module's own
    doc comment for the file format (mirrors `src/config.rs`'s
    `SimConfig::parse` exactly)."""
    metal = None
    facet = None
    alpha = 0.87
    beta_ev = 0.0
    nu = 1.0e13
    temperature_k = 298.15
    species = []
    bimolecular = []
    section = None

    for zero_based_line, raw_line in enumerate(text.splitlines()):
        line_no = zero_based_line + 1
        line = _strip_comment(raw_line).strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            if name not in ("system", "bep", "species", "bimolecular"):
                raise ValueError(f"line {line_no}: unknown section `[{name}]`")
            section = name
            continue

        if "=" not in line:
            raise ValueError(f"line {line_no}: expected `key = value`, got `{line}`")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if section is None:
            raise ValueError(f"line {line_no}: `{key}` appears before any `[section]` header")
        elif section == "system":
            if key == "metal":
                metal = value
            elif key == "facet":
                try:
                    facet = int(value)
                except ValueError:
                    raise ValueError(f"line {line_no}: `facet` expects an integer, got `{value}`")
            else:
                raise ValueError(f"line {line_no}: unknown [system] key `{key}`")
        elif section == "bep":
            try:
                parsed = float(value)
            except ValueError:
                raise ValueError(f"line {line_no}: `{key}` expects a number, got `{value}`")
            if key == "alpha":
                alpha = parsed
            elif key == "beta":
                beta_ev = parsed
            elif key == "nu":
                nu = parsed
            elif key == "temperature":
                temperature_k = parsed
            else:
                raise ValueError(f"line {line_no}: unknown [bep] key `{key}`")
        elif section == "species":
            species.append(_parse_species_row(key, value, line_no))
        elif section == "bimolecular":
            bimolecular.append(_parse_bimolecular_row(key, value, line_no))

    _validate_species(species)
    _validate_bimolecular(bimolecular, species)

    return SimConfig(
        metal=metal,
        facet=facet,
        alpha=alpha,
        beta_ev=beta_ev,
        nu=nu,
        temperature_k=temperature_k,
        species=species,
        bimolecular=bimolecular,
    )


def load_config(path):
    """Read and parse the config file at `path`."""
    with open(path) as f:
        return parse_config(f.read())
