"""Extraction: Catalysis-Hub.org GraphQL API -> the same flat OC20E001
binary format `scripts/extract_energies.py` produces, so `oc20_ingest`
consumes either source unchanged.

Catalysis-Hub (https://catalysis-hub.org) is a curated database of DFT
chemisorption/reaction energies across many publications -- unlike OC20,
it has real `*CO` adsorption data (OC20 holds `*CO` out of train/val
entirely; see scripts/extract_energies.py's docstring). It does *not*
generally have real transition-state activation energies either (the
`activationEnergy` field exists in its schema but is null for effectively
every entry queried), so this still feeds the same BEP + Arrhenius rate
model `oc20_ingest` already applies -- it just adds a real CO data source
that OC20 alone cannot provide.

This queries the public GraphQL endpoint for clean, single-site adsorption
reactions of the form `star + <gas> -> <adsorbate>star` for each of our
three species, paginating with Relay-style cursors, and keeps only
reactions whose reactant/product dictionaries are an *exact* match for
that pattern (no co-adsorbates, no multi-step lumped reactions).

Output format: identical to extract_energies.py's OC20E001 flat binary --
see that script's docstring for the exact byte layout. `sid` here is the
numeric suffix of Catalysis-Hub's own opaque reaction id (e.g.
"UmVhY3Rpb246NDQ4ODE1" base64-decodes to "Reaction:448815" -> sid 448815),
kept only for traceability.
"""

import argparse
import base64
import json
import struct
import sys
import urllib.request

MAGIC = b"OC20E001"
API_URL = "https://api.catalysis-hub.org/graphql"

# (species index, gas reactant key, gas stoichiometry, adsorbed product key)
# Species indices match extract_energies.py / oc20_ingest.rs: 0=O, 1=H, 2=CO.
SPECIES_PATTERNS = [
    (0, "O2gas", 0.5, "Ostar"),
    (1, "H2gas", 0.5, "Hstar"),
    (2, "COgas", 1.0, "COstar"),
]

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
    records = []
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
            records.append((species, int(round(energy_ev * 1000.0)), sid_from_id(node["id"])))

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


def write_records(records, out_path):
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(records)))
        for species, energy_mev, sid in records:
            f.write(struct.pack("<BiI", species, energy_mev, sid & 0xFFFFFFFF))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output flat binary path")
    parser.add_argument(
        "--limit-per-species",
        type=int,
        default=None,
        help="cap records fetched per species (default: no cap)",
    )
    args = parser.parse_args()

    all_records = []
    for species, gas_key, gas_stoich, product_key in SPECIES_PATTERNS:
        print(f"fetching clean {product_key} adsorption reactions...", file=sys.stderr)
        records = fetch_species_records(
            species, gas_key, gas_stoich, product_key, args.limit_per_species
        )
        print(f"  -> {len(records)} clean {product_key} records", file=sys.stderr)
        all_records.extend(records)

    write_records(all_records, args.out)
    print(f"wrote {len(all_records)} total records to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
