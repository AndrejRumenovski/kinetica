"""Extraction: Catalysis-Hub.org GraphQL API -> the same flat binary
format `scripts/extract_energies.py` produces (see oc20e_format.py), so
`oc20_ingest` consumes either source unchanged.

Catalysis-Hub (https://catalysis-hub.org) is a curated database of DFT
chemisorption/reaction energies across many publications -- unlike OC20,
it has real `*CO` adsorption data (OC20 holds `*CO` out of train/val
entirely; see scripts/extract_energies.py's docstring).

Every species/reaction pattern this script looks for is generated from
`--config`'s `[species]`/`[bimolecular]` rows (see `kinetica_config.py`
and `build_species_patterns`/`build_real_barrier_patterns`/
`build_bimolecular_patterns` below) rather than hardcoded here, so
targeting a different species set is a config-file edit, not a source
change -- mirrors `oc20_ingest.rs --config`'s own move away from
compile-time species constants.

Three passes:

1. `fetch_species_records` -- the bulk sweep. Clean, single-site
   adsorption reactions `star + <gas> -> <adsorbate>star` for each
   config-declared species, keeping only *exact* reactant/product matches
   (no co-adsorbates, no multi-step lumped reactions). This is the large
   majority of records (tens of thousands) but reaction energies only --
   `oc20_ingest` derives an activation energy from these via BEP.

2. `fetch_real_barrier_records` -- a small, separate sweep for reactions
   with a real (non-null) `activationEnergy`: genuine DFT-computed
   transition-state barriers, not a BEP estimate. These are rare (most
   entries in this database, like OC20, only have relaxation/adsorption
   energies) and use looser matching than pass 1 (some publications here
   omit the explicit `star` reactant or use a different stoichiometry
   convention for O2/H2 dissociative adsorption), so it isn't safe to
   reuse `is_clean_adsorption`'s strict check for them. Records from this
   pass are merged into pass 1's by `sid`: if the same underlying
   reaction was already found by the bulk sweep, its `has_real_ea`/
   `real_ea_mev` fields are upgraded in place; otherwise it's added as a
   new record (this happens for the O2/H2 entries, which use a
   stoichiometry the bulk sweep's strict filter rejects).

   This same sweep also picks out two further, disjoint categories: real
   *bimolecular* (two-site) barriers, in either direction -- a
   "recombination" `[bimolecular]` entry (both sites occupied -> gas + 2
   vacant), e.g. `O* + CO* -> CO2 + 2*` (`StreibelMicrokinetic2021`,
   ~0.98-1.21 eV on Pd) or `2 H* -> H2 + 2*` (e.g. "Dynamics and
   Hysteresis of Hydrogen Interaction...", ~0.35 eV) in the real
   `configs/pd111.conf`; and a "dissociative" entry (both sites vacant ->
   gas dissociates onto them), e.g. `2* + H2O(g) -> H* + OH*` (water
   splitting, ~1.01 eV) in that same config. Neither can be folded into
   the single-species `OC20E003` format (they consume/produce two
   adsorbed species across two different sites in one event), so they're
   collected separately and written out via `write_bimolecular_records`
   into the parallel `OC20BI03` format, tagged with which direction they
   were measured in -- see `oc20e_format.py`.
"""

import argparse
import base64
import json
import sys
import urllib.request

from kinetica_config import load_config, species_index
from oc20e_format import (
    facet_code,
    metal_index,
    parse_pure_metal,
    write_bimolecular_records,
    write_records,
)

API_URL = "https://api.catalysis-hub.org/graphql"

# A handful of older publications in this database key oxygen's gas
# reference as "Ogas" instead of "O2gas" for the real-barrier pass -- a
# Catalysis-Hub data-source quirk, not a chemistry choice, so it stays a
# script-local override keyed by species name rather than living in the
# shared config schema `kinetica_config.py` parses.
EXTRA_REAL_BARRIER_GAS_ALIASES = {"O": {"Ogas"}}


def build_species_patterns(config):
    """`(species index, gas reactant key, gas stoichiometry, adsorbed
    product key)` for every species `--config` declares with a gas
    source and a molecular/dissociative role -- generated from
    `config.species` instead of the hardcoded list this used to be. A
    `product_only` species (no gas of its own, e.g. OH, which only ever
    forms via a `[bimolecular]` reaction) is skipped here; see
    `build_bimolecular_patterns` for that path instead."""
    return [
        (i, entry.gas, entry.stoich, entry.product)
        for i, entry in enumerate(config.species)
        if entry.gas is not None and entry.role != "product_only"
    ]


def build_real_barrier_patterns(config):
    """Looser per-species key sets for the real-barrier pass: just "does
    this reaction's product side consist of exactly one unit of our
    adsorbate, and does its reactant side consist only of keys we
    recognize as a gas reference or vacant site for that species" -- no
    stoichiometry check, since this small curated subset isn't internally
    consistent about it. Same species selection as
    `build_species_patterns`, plus `EXTRA_REAL_BARRIER_GAS_ALIASES`."""
    return [
        (
            i,
            {"star", entry.gas} | EXTRA_REAL_BARRIER_GAS_ALIASES.get(entry.name, set()),
            entry.product,
        )
        for i, entry in enumerate(config.species)
        if entry.gas is not None and entry.role != "product_only"
    ]


def build_bimolecular_patterns(config, direction):
    """Bimolecular (two-site) patterns for `--config`'s `[bimolecular]`
    entries matching `direction` ("recombination" or "dissociative") --
    generated from `config.bimolecular` instead of the hardcoded
    `RECOMBINATION_PATTERNS`/`DISSOCIATIVE_PATTERNS` lists these used to
    be.

    **Recombination** (e.g. `O* + CO* -> CO2 + 2*`): every reactant
    species clears to vacant, and the products are (a gas molecule) +
    (freed sites) -- the direction `oc20_ingest.rs` builds forward-only,
    no thermodynamic reverse (see `OC20BI03`'s `is_dissociative = 0`).
    Returns `(species_a, species_b, reactant_stoich, product_key)`.
    Homoatomic entries (`species_a == species_b`, e.g. `2 H* -> H2 + 2*`)
    get `{product: 2.0}` as their reactant stoichiometry instead of two
    separate 1.0 entries.

    **Dissociative** (e.g. `2* + H2O(g) -> H* + OH*`): the reverse
    shape -- two vacant sites plus a gas reactant become two occupied
    sites, the one direction where building a real thermodynamic reverse
    (associative desorption) makes sense, since it's genuinely the same
    elementary step run backward (see `OC20BI03`'s `is_dissociative = 1`).
    Returns `(species_a, species_b, gas_stoich, product_stoich)`.

    Reactant matching in both directions allows an extra zero-stoichiometry
    "star" key alongside the named species/gas -- this database's records
    for these exact reactions list it explicitly (e.g. `{"star": 0,
    "Ostar": 1, "COstar": 1}`) rather than omitting it; see
    `match_recombination_pattern`/`match_dissociative_pattern`.
    """
    patterns = []
    for entry in config.bimolecular:
        if entry.direction != direction:
            continue
        ia = species_index(config, entry.species_a)
        ib = species_index(config, entry.species_b)
        sa = config.species[ia]
        sb = config.species[ib]
        homoatomic = ia == ib
        if direction == "recombination":
            reactant_stoich = (
                {sa.product: 2.0} if homoatomic else {sa.product: 1.0, sb.product: 1.0}
            )
            patterns.append((ia, ib, reactant_stoich, entry.gas))
        else:
            gas_stoich = {entry.gas: 1.0}
            product_stoich = (
                {sa.product: 2.0} if homoatomic else {sa.product: 1.0, sb.product: 1.0}
            )
            patterns.append((ia, ib, gas_stoich, product_stoich))
    return patterns


def match_recombination_pattern(reactants, products, recombination_patterns):
    """Return `(species_a, species_b)` for the first
    `recombination_patterns` entry `reactants`/`products` matches, or
    `None`. Product matching only requires the named gas product be
    present at ~1.0 stoichiometry; the freed-site count on the product
    side isn't checked, since some publications omit it."""
    for species_a, species_b, reactant_stoich, product_key in recombination_patterns:
        if not set(reactants.keys()) <= (set(reactant_stoich.keys()) | {"star"}):
            continue
        if any(abs(reactants.get(k, 0) - v) > 1e-6 for k, v in reactant_stoich.items()):
            continue
        if abs(products.get(product_key, 0) - 1.0) > 1e-6:
            continue
        return species_a, species_b
    return None


def match_dissociative_pattern(reactants, products, dissociative_patterns):
    """Return `(species_a, species_b)` for the first
    `dissociative_patterns` entry `reactants`/`products` matches, or
    `None`."""
    for species_a, species_b, gas_stoich, product_stoich in dissociative_patterns:
        if not set(reactants.keys()) <= (set(gas_stoich.keys()) | {"star"}):
            continue
        if any(abs(reactants.get(k, 0) - v) > 1e-6 for k, v in gas_stoich.items()):
            continue
        if set(products.keys()) != set(product_stoich.keys()):
            continue
        if any(abs(products.get(k, 0) - v) > 1e-6 for k, v in product_stoich.items()):
            continue
        return species_a, species_b
    return None

PAGE_SIZE = 100


def graphql_query(query):
    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read())
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    return payload["data"]


def sid_from_id(opaque_id):
    try:
        decoded = base64.b64decode(opaque_id).decode("utf-8")  # "Reaction:448815"
        return int(decoded.rsplit(":", 1)[-1])
    except Exception:
        return 0


def is_clean_adsorption(reactants, products, gas_key, gas_stoich, product_key):
    """True iff `reactants`/`products` are *exactly* the elementary
    adsorption step `star + gas_stoich*gas_key -> 1*product_key`, with no
    extra co-reactants/co-products (e.g. no lumped multi-adsorbate or
    hydrogenation steps riding along)."""
    if set(reactants.keys()) != {"star", gas_key}:
        return False
    if abs(reactants.get("star", 0) - 1.0) > 1e-6:
        return False
    if abs(reactants.get(gas_key, 0) - gas_stoich) > 1e-6:
        return False
    if set(products.keys()) != {product_key}:
        return False
    if abs(products.get(product_key, 0) - 1.0) > 1e-6:
        return False
    return True


def fetch_species_records(species, gas_key, gas_stoich, product_key, limit=None, metal=None, facet=None):
    """Bulk sweep: returns {sid: (species, energy_mev, sid, False, 0, metal_idx, facet_code)}.

    `metal`/`facet`, when given, are pushed down as server-side GraphQL
    filter args (`surfaceComposition`/`facet`) so this doesn't have to
    page through every reaction of every metal just to throw most of them
    away client-side.
    """
    records = {}
    after = None
    metal_clause = f', surfaceComposition: "{metal}"' if metal else ""
    facet_clause = f', facet: "{facet}"' if facet else ""
    metal_idx = metal_index(metal) if metal else 0
    facet_val = facet_code(facet) if facet else 0
    while True:
        after_clause = f', after: "{after}"' if after else ""
        limit_clause = f", first: {min(PAGE_SIZE, limit - len(records)) if limit else PAGE_SIZE}"
        query = f"""{{
          reactions(reactants: "{gas_key}"{metal_clause}{facet_clause}{limit_clause}{after_clause}) {{
            totalCount
            pageInfo {{ hasNextPage endCursor }}
            edges {{ node {{ id reactants products reactionEnergy surfaceComposition facet }} }}
          }}
        }}"""
        data = graphql_query(query)["reactions"]

        for edge in data["edges"]:
            node = edge["node"]
            reactants = json.loads(node["reactants"])
            products = json.loads(node["products"])
            if not is_clean_adsorption(reactants, products, gas_key, gas_stoich, product_key):
                continue
            energy_ev = node["reactionEnergy"]
            if energy_ev is None:
                continue
            sid = sid_from_id(node["id"])
            # The server-side filter already constrains these when set, but
            # re-derive from the node's own fields rather than trusting the
            # filter args blindly -- keeps the record's stored metal/facet
            # accurate even if this function is ever called without one.
            rec_metal = metal_idx if metal else metal_index(parse_pure_metal(node["surfaceComposition"]))
            rec_facet = facet_val if facet else facet_code(node["facet"])
            records[sid] = (species, int(round(energy_ev * 1000.0)), sid, False, 0, rec_metal, rec_facet)

        print(
            f"  ...{gas_key}: scanned to {len(records)} matches "
            f"(of {data['totalCount']} candidates)",
            file=sys.stderr,
        )

        if limit and len(records) >= limit:
            break
        if not data["pageInfo"]["hasNextPage"]:
            break
        after = data["pageInfo"]["endCursor"]

    return records


def fetch_real_barrier_records(
    real_barrier_patterns,
    recombination_patterns,
    dissociative_patterns,
    metal=None,
    facet=None,
):
    """Real-barrier sweep: returns `(mono_records, bimolecular_records)`.

    `mono_records`: {sid: (species, energy_mev, sid, True, ea_mev, metal,
    facet)} for every reaction with a non-null `activationEnergy` matching
    one of `real_barrier_patterns`.

    `bimolecular_records`: {sid: (species_a, species_b, energy_mev, sid,
    ea_mev, metal, facet, is_dissociative)} for every reaction matching
    one of `recombination_patterns` (is_dissociative=False) or
    `dissociative_patterns` (is_dissociative=True). Disjoint from
    `mono_records` and from each other -- a node only ever matches one of
    the three categories, since the reactant-key sets involved don't
    overlap.

    `metal`/`facet`, when given, are pushed down as server-side filter
    args, same as `fetch_species_records` -- this sweep already scans a
    much smaller candidate set (only records with a real
    `activationEnergy`), but there's no reason to pull barriers for metals
    this run doesn't care about either.
    """
    mono_records = {}
    bimolecular_records = {}
    seen_ids = set()
    after = None
    metal_clause = f', surfaceComposition: "{metal}"' if metal else ""
    facet_clause = f', facet: "{facet}"' if facet else ""
    while True:
        after_clause = f', after: "{after}"' if after else ""
        query = f"""{{
          reactions(first: {PAGE_SIZE}, activationEnergy: -100, op: ">"{metal_clause}{facet_clause}{after_clause}) {{
            pageInfo {{ hasNextPage endCursor }}
            edges {{ node {{ id reactants products activationEnergy reactionEnergy surfaceComposition facet }} }}
          }}
        }}"""
        data = graphql_query(query)["reactions"]

        for edge in data["edges"]:
            node = edge["node"]
            if node["id"] in seen_ids:
                continue
            seen_ids.add(node["id"])

            reactants = json.loads(node["reactants"])
            products = json.loads(node["products"])
            energy_ev = node["reactionEnergy"]
            if energy_ev is None:
                continue

            rec_metal = metal_index(parse_pure_metal(node["surfaceComposition"]))
            rec_facet = facet_code(node["facet"])

            matched = False
            for species, gas_keys, product_key in real_barrier_patterns:
                if set(products.keys()) != {product_key}:
                    continue
                if abs(products.get(product_key, 0) - 1.0) > 1e-6:
                    continue
                if not set(reactants.keys()) <= gas_keys:
                    continue
                sid = sid_from_id(node["id"])
                mono_records[sid] = (
                    species,
                    int(round(energy_ev * 1000.0)),
                    sid,
                    True,
                    int(round(node["activationEnergy"] * 1000.0)),
                    rec_metal,
                    rec_facet,
                )
                matched = True
                break

            if matched:
                continue

            recombination_match = match_recombination_pattern(
                reactants, products, recombination_patterns
            )
            if recombination_match is not None:
                species_a, species_b = recombination_match
                sid = sid_from_id(node["id"])
                bimolecular_records[sid] = (
                    species_a,
                    species_b,
                    int(round(energy_ev * 1000.0)),
                    sid,
                    int(round(node["activationEnergy"] * 1000.0)),
                    rec_metal,
                    rec_facet,
                    False,
                )
                continue

            dissociative_match = match_dissociative_pattern(
                reactants, products, dissociative_patterns
            )
            if dissociative_match is not None:
                species_a, species_b = dissociative_match
                sid = sid_from_id(node["id"])
                bimolecular_records[sid] = (
                    species_a,
                    species_b,
                    int(round(energy_ev * 1000.0)),
                    sid,
                    int(round(node["activationEnergy"] * 1000.0)),
                    rec_metal,
                    rec_facet,
                    True,
                )

        if not data["pageInfo"]["hasNextPage"]:
            break
        after = data["pageInfo"]["endCursor"]

    return mono_records, bimolecular_records


def default_bimolecular_out(out_path):
    if out_path.endswith(".bin"):
        return out_path[: -len(".bin")] + "_bimolecular.bin"
    return out_path + "_bimolecular.bin"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output flat binary path")
    parser.add_argument(
        "--bimolecular-out",
        default=None,
        help="output path for real bimolecular (two-site) records, both "
        "recombination and dissociative-adsorption direction "
        "[default: <out> with a _bimolecular suffix]",
    )
    parser.add_argument(
        "--limit-per-species",
        type=int,
        default=None,
        help="cap records fetched per species in the bulk sweep (default: no cap)",
    )
    parser.add_argument(
        "--skip-real-barriers",
        action="store_true",
        help="skip the (fast) real-activation-energy sweep (also skips the "
        "bimolecular sweep, since it rides along the same pass)",
    )
    parser.add_argument(
        "--metal",
        default=None,
        help="restrict to this pure metal's surfaceComposition (e.g. Pd) -- "
        "pushed down as a server-side filter; see oc20e_format.METALS for "
        "the tracked list. Independent of --config's [system] metal, which "
        "only drives BEP/species defaults on the oc20_ingest (Rust) side -- "
        "pass this explicitly to restrict the extraction query itself",
    )
    parser.add_argument(
        "--facet",
        default=None,
        help="restrict to this Miller-index facet (e.g. 111) -- pushed down "
        "as a server-side filter; independent of --config, same reasoning "
        "as --metal above",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to the sectioned config file oc20_ingest --config also reads "
        "(see kinetica_config.py); its [species]/[bimolecular] rows drive which "
        "adsorption/recombination/dissociative-adsorption patterns this "
        "extraction looks for",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    species_patterns = build_species_patterns(config)
    real_barrier_patterns = build_real_barrier_patterns(config)
    recombination_patterns = build_bimolecular_patterns(config, "recombination")
    dissociative_patterns = build_bimolecular_patterns(config, "dissociative")

    by_sid = {}
    for species, gas_key, gas_stoich, product_key in species_patterns:
        print(f"fetching clean {product_key} adsorption reactions...", file=sys.stderr)
        records = fetch_species_records(
            species,
            gas_key,
            gas_stoich,
            product_key,
            args.limit_per_species,
            metal=args.metal,
            facet=args.facet,
        )
        print(f"  -> {len(records)} clean {product_key} records", file=sys.stderr)
        by_sid.update(records)

    bimolecular_records = {}
    if not args.skip_real_barriers:
        print("fetching real (non-BEP) activation-energy reactions...", file=sys.stderr)
        real_barrier_records, bimolecular_records = fetch_real_barrier_records(
            real_barrier_patterns,
            recombination_patterns,
            dissociative_patterns,
            metal=args.metal,
            facet=args.facet,
        )
        new_count = sum(1 for sid in real_barrier_records if sid not in by_sid)
        upgraded_count = len(real_barrier_records) - new_count
        by_sid.update(real_barrier_records)
        print(
            f"  -> {len(real_barrier_records)} real-barrier records "
            f"({upgraded_count} upgraded existing, {new_count} newly added)",
            file=sys.stderr,
        )
        dissociative_count = sum(1 for rec in bimolecular_records.values() if rec[7])
        print(
            f"  -> {len(bimolecular_records)} real bimolecular records "
            f"({len(bimolecular_records) - dissociative_count} recombination, "
            f"{dissociative_count} dissociative adsorption)",
            file=sys.stderr,
        )

    all_records = list(by_sid.values())
    write_records(all_records, args.out)
    print(f"wrote {len(all_records)} total records to {args.out}", file=sys.stderr)

    if bimolecular_records:
        bimolecular_out = args.bimolecular_out or default_bimolecular_out(args.out)
        write_bimolecular_records(list(bimolecular_records.values()), bimolecular_out)
        print(
            f"wrote {len(bimolecular_records)} bimolecular records to {bimolecular_out}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
