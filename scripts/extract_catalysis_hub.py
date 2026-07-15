"""Extraction: Catalysis-Hub.org GraphQL API -> the same flat binary
format `scripts/extract_energies.py` produces (see oc20e_format.py), so
`oc20_ingest` consumes either source unchanged.

Catalysis-Hub (https://catalysis-hub.org) is a curated database of DFT
chemisorption/reaction energies across many publications -- unlike OC20,
it has real `*CO` adsorption data (OC20 holds `*CO` out of train/val
entirely; see scripts/extract_energies.py's docstring).

Three passes:

1. `fetch_species_records` -- the bulk sweep. Clean, single-site
   adsorption reactions `star + <gas> -> <adsorbate>star` for each of our
   three species, keeping only *exact* reactant/product matches (no
   co-adsorbates, no multi-step lumped reactions). This is the large
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

   This same sweep also picks out a third, disjoint category: real
   *bimolecular* CO-oxidation barriers, `O* + CO* -> CO2 + 2*`
   (publications like `StreibelMicrokinetic2021` report genuine
   NEB/dimer-method barriers of ~0.98-1.21 eV on Pd for exactly this
   step). These can't be folded into the single-species `OC20E002`
   format (they consume two adsorbed species from two different sites in
   one event), so they're collected separately and written out via
   `write_bimolecular_records` into the parallel `OC20BI01` format --
   see `oc20e_format.py`.
"""

import argparse
import base64
import json
import sys
import urllib.request

from oc20e_format import write_bimolecular_records, write_records

API_URL = "https://api.catalysis-hub.org/graphql"

# (species index, gas reactant key, gas stoichiometry, adsorbed product key)
# Species indices match extract_energies.py / oc20_ingest.rs: 0=O, 1=H, 2=CO.
SPECIES_PATTERNS = [
    (0, "O2gas", 0.5, "Ostar"),
    (1, "H2gas", 0.5, "Hstar"),
    (2, "COgas", 1.0, "COstar"),
]

# Looser per-species key sets for the real-barrier pass: just "does this
# reaction's product side consist of exactly one unit of our adsorbate,
# and does its reactant side consist only of keys we recognize as a gas
# reference or vacant site for that species" -- no stoichiometry check,
# since this small curated subset isn't internally consistent about it.
REAL_BARRIER_PATTERNS = [
    (0, {"star", "O2gas", "Ogas"}, "Ostar"),
    (1, {"star", "H2gas"}, "Hstar"),
    (2, {"star", "COgas"}, "COstar"),
]

# Bimolecular CO-oxidation: O* + CO* -> CO2 + 2*. Species indices match
# SPECIES_PATTERNS above (0 = O, 1 = H, 2 = CO). Reactant side must have
# exactly one O* and one CO* (no co-adsorbates) -- a zero-stoichiometry
# "star" entry is allowed alongside them, since this database's records
# for this exact reaction (e.g. `StreibelMicrokinetic2021`, "Microkinetic
# Modeling of Propene Combustion") list reactants as
# `{"star": 0, "Ostar": 1, "COstar": 1}` rather than omitting "star"
# entirely. Product side just needs to contain CO2gas -- some
# publications here omit the freed `2 star` product explicitly, so that
# part isn't checked.
BIMOLECULAR_SPECIES_A = 0  # O
BIMOLECULAR_SPECIES_B = 2  # CO
BIMOLECULAR_REACTANT_KEYS = {"star", "Ostar", "COstar"}
BIMOLECULAR_PRODUCT_KEY = "CO2gas"

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


def fetch_species_records(species, gas_key, gas_stoich, product_key, limit=None):
    """Bulk sweep: returns {sid: (species, energy_mev, sid, False, 0)}."""
    records = {}
    after = None
    while True:
        after_clause = f', after: "{after}"' if after else ""
        limit_clause = f", first: {min(PAGE_SIZE, limit - len(records)) if limit else PAGE_SIZE}"
        query = f"""{{
          reactions(reactants: "{gas_key}"{limit_clause}{after_clause}) {{
            totalCount
            pageInfo {{ hasNextPage endCursor }}
            edges {{ node {{ id reactants products reactionEnergy }} }}
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
            records[sid] = (species, int(round(energy_ev * 1000.0)), sid, False, 0)

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


def fetch_real_barrier_records():
    """Real-barrier sweep: returns `(mono_records, bimolecular_records)`.

    `mono_records`: {sid: (species, energy_mev, sid, True, ea_mev)} for
    every reaction with a non-null `activationEnergy` matching one of
    `REAL_BARRIER_PATTERNS`.

    `bimolecular_records`: {sid: (species_a, species_b, energy_mev, sid,
    ea_mev)} for every reaction matching the CO-oxidation pattern
    (`BIMOLECULAR_REACTANT_KEYS` -> product containing
    `BIMOLECULAR_PRODUCT_KEY`). Disjoint from `mono_records` -- a node
    only ever matches one of the two categories, since the reactant-key
    sets involved don't overlap.
    """
    mono_records = {}
    bimolecular_records = {}
    seen_ids = set()
    after = None
    while True:
        after_clause = f', after: "{after}"' if after else ""
        query = f"""{{
          reactions(first: {PAGE_SIZE}, activationEnergy: -100, op: ">"{after_clause}) {{
            pageInfo {{ hasNextPage endCursor }}
            edges {{ node {{ id reactants products activationEnergy reactionEnergy }} }}
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

            matched = False
            for species, gas_keys, product_key in REAL_BARRIER_PATTERNS:
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
                )
                matched = True
                break

            if matched:
                continue

            if (
                set(reactants.keys()) <= BIMOLECULAR_REACTANT_KEYS
                and abs(reactants.get("Ostar", 0) - 1.0) <= 1e-6
                and abs(reactants.get("COstar", 0) - 1.0) <= 1e-6
                and abs(products.get(BIMOLECULAR_PRODUCT_KEY, 0) - 1.0) <= 1e-6
            ):
                sid = sid_from_id(node["id"])
                bimolecular_records[sid] = (
                    BIMOLECULAR_SPECIES_A,
                    BIMOLECULAR_SPECIES_B,
                    int(round(energy_ev * 1000.0)),
                    sid,
                    int(round(node["activationEnergy"] * 1000.0)),
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
        help="output path for real bimolecular (CO-oxidation) records "
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
        "bimolecular CO-oxidation sweep, since it rides along the same pass)",
    )
    args = parser.parse_args()

    by_sid = {}
    for species, gas_key, gas_stoich, product_key in SPECIES_PATTERNS:
        print(f"fetching clean {product_key} adsorption reactions...", file=sys.stderr)
        records = fetch_species_records(
            species, gas_key, gas_stoich, product_key, args.limit_per_species
        )
        print(f"  -> {len(records)} clean {product_key} records", file=sys.stderr)
        by_sid.update(records)

    bimolecular_records = {}
    if not args.skip_real_barriers:
        print("fetching real (non-BEP) activation-energy reactions...", file=sys.stderr)
        real_barrier_records, bimolecular_records = fetch_real_barrier_records()
        new_count = sum(1 for sid in real_barrier_records if sid not in by_sid)
        upgraded_count = len(real_barrier_records) - new_count
        by_sid.update(real_barrier_records)
        print(
            f"  -> {len(real_barrier_records)} real-barrier records "
            f"({upgraded_count} upgraded existing, {new_count} newly added)",
            file=sys.stderr,
        )
        print(
            f"  -> {len(bimolecular_records)} real bimolecular CO-oxidation records "
            "(O* + CO* -> CO2 + 2*)",
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
