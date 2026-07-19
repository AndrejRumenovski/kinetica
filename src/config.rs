//! Hand-rolled parser for the sectioned text config file that drives
//! which metal/facet/species set `oc20_ingest --config <PATH>` builds a
//! `reactions.lut` from -- a single shared source of truth, replacing
//! the hand-comment convention ("species index N means adsorbate X, keep
//! these five files in sync") this project relied on before. The
//! matching Python extraction scripts don't read this file yet (they
//! still carry their own hardcoded species patterns) -- that's the next
//! item of work, not yet done.
//!
//! Zero new dependencies (no `serde`/`toml`): consistent with this
//! crate's deliberately minimal, exact-pinned dependency list and the
//! hand-rolled CLI parsing already used elsewhere (see
//! `bin/oc20_ingest.rs`'s own `Config::parse`).
//!
//! `SimConfig::parse` and its LUT-header counterpart
//! (`layout::SpeciesTable`) were built and fuzzed in isolation one phase
//! before `oc20_ingest` grew the `--config` flag that depends on them --
//! see `configs/pd111.conf` for the config this repo's own
//! `reactions.lut` is built from.
//!
//! # File format
//!
//! ```text
//! [system]
//! metal = Pd
//! facet = 111
//!
//! [bep]
//! alpha = 0.87
//! beta = 0.0
//! nu = 1e13
//! temperature = 298.15
//!
//! [species]
//! O   = 0x01, O2gas,  0.5, Ostar,   dissociative, 0
//! H   = 0x04, H2gas,  0.5, Hstar,   dissociative, 1
//! CO  = 0x02, COgas,  1.0, COstar,  molecular,    5
//! OH  = 0x08, -,      -,   OHstar,  product_only, -
//! H2O = 0x10, H2Ogas, 1.0, H2Ostar, molecular,    -
//!
//! [bimolecular]
//! co_ox   = O, CO, recombination, CO2gas
//! h2_rec  = H, H,  recombination, H2gas
//! h2o_dis = H, OH, dissociative,  H2Ogas
//! ```
//!
//! `#` starts a trailing comment (stripped before parsing); `-` means
//! "no value" for an optional field. A species' index into
//! `layout::SPECIES_BITS`-equivalent arrays is its `[species]` *row
//! order*, deliberately decoupled from its one-hot `bit` value (matching
//! `layout::SPECIES_BITS`'s own non-sorted bit assignment).

/// One `[species]` row: a single adsorbate's identity and how
/// `oc20_ingest`/the Python extractors should treat it.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeciesEntry {
    /// Display name (e.g. `"O"`, `"H2O"`) -- also the label
    /// `layout::SpeciesTable` eventually stamps into the built LUT's
    /// header.
    pub name: String,
    /// One-hot occupancy bit this species occupies (see
    /// `layout::SPECIES_BITS`'s doc comment for the 8-species ceiling
    /// this is validated against).
    pub bit: u8,
    /// Gas-phase molecule this species adsorbs from (e.g. `"O2gas"`), if
    /// any -- consumed only by the Python extraction scripts.
    pub gas: Option<String>,
    /// Per-atom/per-molecule stoichiometry against `gas` (e.g. `0.5` for
    /// a diatomic gas dissociating two atoms per molecule), if
    /// applicable.
    pub stoich: Option<f64>,
    /// Catalysis-Hub/OC20 product key (e.g. `"Ostar"`), if any --
    /// consumed only by the Python extraction scripts.
    pub product: Option<String>,
    /// How this species adsorbs -- see `SpeciesRole`.
    pub role: SpeciesRole,
    /// OC20's own global adsorbate-index table id for this species, if
    /// it's ever sourced from OC20 rather than only Catalysis-Hub.
    pub oc20_ads_id: Option<u32>,
}

/// How a species' monomolecular adsorption/desorption is modeled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeciesRole {
    /// Adsorbs as a single molecule onto one site (e.g. CO, H2O).
    Molecular,
    /// Dissociates from a diatomic gas onto two adjacent sites (e.g. O
    /// from O2, H from H2) -- generalizes `oc20_ingest`'s old hardcoded
    /// `DISSOCIATIVE_SPECIES` list.
    Dissociative,
    /// Never has its own monomolecular adsorption template -- only
    /// formed/consumed as a product of a `[bimolecular]` reaction (e.g.
    /// OH, which only ever forms via water splitting).
    ProductOnly,
}

/// One `[bimolecular]` row: a two-site reaction between two
/// already-declared `[species]` entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BimolecularEntry {
    /// This row's label (e.g. `"co_ox"`) -- purely a human-readable key,
    /// not consumed structurally.
    pub key: String,
    /// First reactant species name (must match a `[species]` entry).
    pub species_a: String,
    /// Second reactant species name (must match a `[species]` entry).
    pub species_b: String,
    /// Which direction this reaction runs -- see `BimolecularDirection`.
    pub direction: BimolecularDirection,
    /// Gas-phase molecule this reaction produces/consumes (e.g.
    /// `"CO2gas"`, `"H2Ogas"`), if any -- consumed only by the Python
    /// extraction scripts.
    pub gas: Option<String>,
}

/// Which direction a `[bimolecular]` entry's real barrier runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BimolecularDirection {
    /// Two occupied, adjacent sites recombine into a gas-phase product
    /// (e.g. CO oxidation, H2 recombination) -- forward-only, since
    /// there's no thermodynamically meaningful reverse for a gas product
    /// leaving the surface.
    Recombination,
    /// Two vacant, adjacent sites dissociatively adsorb a heteroatomic
    /// gas (e.g. water splitting, `2* + H2O(g) -> H* + OH*`) -- built
    /// both directions from one real forward barrier.
    Dissociative,
}

/// A fully parsed config file: everything needed to drive both
/// `oc20_ingest` (Rust) and the Python extraction scripts from one
/// shared source of truth.
#[derive(Clone, Debug, PartialEq)]
pub struct SimConfig {
    /// `[system] metal`, if set -- resolved against `oc20_ingest`'s own
    /// `METALS` table at use time; this module doesn't know that list,
    /// so it's carried through unvalidated as a plain string.
    pub metal: Option<String>,
    /// `[system] facet`, if set.
    pub facet: Option<u16>,
    /// `[bep] alpha` -- BEP slope, default `0.87` (matches
    /// `oc20_ingest`'s own CLI-flag default).
    pub alpha: f64,
    /// `[bep] beta` -- BEP intercept (eV), default `0.0`.
    pub beta_ev: f64,
    /// `[bep] nu` -- Arrhenius attempt frequency (s^-1), default `1e13`.
    pub nu: f64,
    /// `[bep] temperature` -- Kelvin, default `298.15`.
    pub temperature_k: f64,
    /// `[species]` entries, in declaration order -- that order becomes
    /// each species' index into `layout::SPECIES_BITS`-equivalent
    /// arrays.
    pub species: Vec<SpeciesEntry>,
    /// `[bimolecular]` entries, in declaration order.
    pub bimolecular: Vec<BimolecularEntry>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    System,
    Bep,
    Species,
    Bimolecular,
}

impl SimConfig {
    /// Parse `text` (a config file's full contents) into a `SimConfig`,
    /// or a human-readable error naming the offending line. Hand-rolled
    /// rather than pulled from a crate -- see this module's own doc
    /// comment for why.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut metal = None;
        let mut facet = None;
        let mut alpha = 0.87;
        let mut beta_ev = 0.0;
        let mut nu = 1.0e13;
        let mut temperature_k = 298.15;
        let mut species = Vec::new();
        let mut bimolecular = Vec::new();
        let mut section = Section::None;

        for (zero_based_line, raw_line) in text.lines().enumerate() {
            let line_no = zero_based_line + 1;
            let line = strip_comment(raw_line).trim();
            if line.is_empty() {
                continue;
            }

            if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                section = match name {
                    "system" => Section::System,
                    "bep" => Section::Bep,
                    "species" => Section::Species,
                    "bimolecular" => Section::Bimolecular,
                    other => return Err(format!("line {line_no}: unknown section `[{other}]`")),
                };
                continue;
            }

            let Some((key, value)) = line.split_once('=') else {
                return Err(format!(
                    "line {line_no}: expected `key = value`, got `{line}`"
                ));
            };
            let key = key.trim();
            let value = value.trim();

            match section {
                Section::None => {
                    return Err(format!(
                        "line {line_no}: `{key}` appears before any `[section]` header"
                    ))
                }
                Section::System => match key {
                    "metal" => metal = Some(value.to_string()),
                    "facet" => {
                        facet = Some(value.parse().map_err(|_| {
                            format!("line {line_no}: `facet` expects an integer, got `{value}`")
                        })?)
                    }
                    other => return Err(format!("line {line_no}: unknown [system] key `{other}`")),
                },
                Section::Bep => {
                    let parsed = parse_f64(value, line_no, key)?;
                    match key {
                        "alpha" => alpha = parsed,
                        "beta" => beta_ev = parsed,
                        "nu" => nu = parsed,
                        "temperature" => temperature_k = parsed,
                        other => {
                            return Err(format!("line {line_no}: unknown [bep] key `{other}`"))
                        }
                    }
                }
                Section::Species => species.push(parse_species_row(key, value, line_no)?),
                Section::Bimolecular => {
                    bimolecular.push(parse_bimolecular_row(key, value, line_no)?)
                }
            }
        }

        validate_species(&species)?;
        validate_bimolecular(&bimolecular, &species)?;

        Ok(SimConfig {
            metal,
            facet,
            alpha,
            beta_ev,
            nu,
            temperature_k,
            species,
            bimolecular,
        })
    }
}

fn strip_comment(line: &str) -> &str {
    line.split('#').next().unwrap_or("")
}

fn parse_f64(value: &str, line_no: usize, key: &str) -> Result<f64, String> {
    value
        .parse()
        .map_err(|_| format!("line {line_no}: `{key}` expects a number, got `{value}`"))
}

/// `-` means "no value" for an optional field; anything else is that
/// field's value verbatim.
fn parse_opt(field: &str) -> Option<String> {
    if field == "-" {
        None
    } else {
        Some(field.to_string())
    }
}

fn parse_species_row(name: &str, value: &str, line_no: usize) -> Result<SpeciesEntry, String> {
    let fields: Vec<&str> = value.split(',').map(str::trim).collect();
    let [bit, gas, stoich, product, role, oc20_ads_id] = fields.as_slice() else {
        return Err(format!(
            "line {line_no}: species `{name}` expects 6 comma-separated fields \
             (bit, gas, stoich, product, role, oc20_ads_id), got {}",
            fields.len()
        ));
    };

    let bit = parse_bit(bit, line_no, name)?;
    let gas = parse_opt(gas);
    let stoich = match parse_opt(stoich) {
        Some(s) => Some(s.parse().map_err(|_| {
            format!("line {line_no}: species `{name}`'s stoich expects a number, got `{s}`")
        })?),
        None => None,
    };
    let product = parse_opt(product);
    let role = match *role {
        "molecular" => SpeciesRole::Molecular,
        "dissociative" => SpeciesRole::Dissociative,
        "product_only" => SpeciesRole::ProductOnly,
        other => {
            return Err(format!(
                "line {line_no}: species `{name}`'s role `{other}` isn't one of \
                 molecular/dissociative/product_only"
            ))
        }
    };
    let oc20_ads_id = match parse_opt(oc20_ads_id) {
        Some(s) => Some(s.parse().map_err(|_| {
            format!("line {line_no}: species `{name}`'s oc20_ads_id expects an integer, got `{s}`")
        })?),
        None => None,
    };

    Ok(SpeciesEntry {
        name: name.to_string(),
        bit,
        gas,
        stoich,
        product,
        role,
        oc20_ads_id,
    })
}

/// Accepts a plain decimal byte or a `0x`-prefixed hex byte (the config
/// examples throughout this crate use `0x01`-style bits), and rejects
/// anything that isn't a one-hot bit -- the same requirement
/// `layout::SpeciesTable::new` enforces on the Rust side, checked here
/// too so a malformed config fails with a line number instead of a
/// generic error from deeper in the pipeline.
fn parse_bit(field: &str, line_no: usize, species_name: &str) -> Result<u8, String> {
    let value = match field.strip_prefix("0x") {
        Some(hex) => u8::from_str_radix(hex, 16),
        None => field.parse(),
    }
    .map_err(|_| {
        format!(
            "line {line_no}: species `{species_name}`'s bit `{field}` isn't a valid byte \
             (decimal or 0x-prefixed hex)"
        )
    })?;
    if value.count_ones() != 1 {
        return Err(format!(
            "line {line_no}: species `{species_name}`'s bit {value:#04x} is not one-hot"
        ));
    }
    Ok(value)
}

fn parse_bimolecular_row(
    key: &str,
    value: &str,
    line_no: usize,
) -> Result<BimolecularEntry, String> {
    let fields: Vec<&str> = value.split(',').map(str::trim).collect();
    let [species_a, species_b, direction, gas] = fields.as_slice() else {
        return Err(format!(
            "line {line_no}: bimolecular entry `{key}` expects 4 comma-separated fields \
             (species_a, species_b, direction, gas), got {}",
            fields.len()
        ));
    };
    let direction = match *direction {
        "recombination" => BimolecularDirection::Recombination,
        "dissociative" => BimolecularDirection::Dissociative,
        other => {
            return Err(format!(
                "line {line_no}: bimolecular entry `{key}`'s direction `{other}` isn't one of \
                 recombination/dissociative"
            ))
        }
    };
    Ok(BimolecularEntry {
        key: key.to_string(),
        species_a: species_a.to_string(),
        species_b: species_b.to_string(),
        direction,
        gas: parse_opt(gas),
    })
}

fn validate_species(species: &[SpeciesEntry]) -> Result<(), String> {
    if species.len() > crate::layout::MAX_SPECIES {
        return Err(format!(
            "{} species exceeds the architectural ceiling of {} (see \
             layout::SPECIES_BITS's doc comment for why)",
            species.len(),
            crate::layout::MAX_SPECIES
        ));
    }
    for (i, entry) in species.iter().enumerate() {
        if species[..i].iter().any(|e| e.name == entry.name) {
            return Err(format!(
                "species name `{}` is declared more than once",
                entry.name
            ));
        }
        if species[..i].iter().any(|e| e.bit == entry.bit) {
            return Err(format!(
                "species bit {:#04x} is declared more than once",
                entry.bit
            ));
        }
    }
    Ok(())
}

fn validate_bimolecular(
    bimolecular: &[BimolecularEntry],
    species: &[SpeciesEntry],
) -> Result<(), String> {
    for entry in bimolecular {
        for name in [&entry.species_a, &entry.species_b] {
            if !species.iter().any(|s| &s.name == name) {
                return Err(format!(
                    "bimolecular entry `{}` references undeclared species `{name}`",
                    entry.key
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PD111_EXAMPLE: &str = "\
[system]
metal = Pd
facet = 111

[bep]
alpha = 0.87
beta = 0.0
nu = 1e13
temperature = 298.15

[species]
O   = 0x01, O2gas,  0.5, Ostar,   dissociative, 0
H   = 0x04, H2gas,  0.5, Hstar,   dissociative, 1
CO  = 0x02, COgas,  1.0, COstar,  molecular,    5
OH  = 0x08, -,      -,   OHstar,  product_only, -
H2O = 0x10, H2Ogas, 1.0, H2Ostar, molecular,    -

[bimolecular]
co_ox   = O, CO, recombination, CO2gas
h2_rec  = H, H,  recombination, H2gas
h2o_dis = H, OH, dissociative,  H2Ogas
";

    #[test]
    fn parses_the_full_pd111_example() {
        let config = SimConfig::parse(PD111_EXAMPLE).unwrap();

        assert_eq!(config.metal.as_deref(), Some("Pd"));
        assert_eq!(config.facet, Some(111));
        assert_eq!(config.alpha, 0.87);
        assert_eq!(config.beta_ev, 0.0);
        assert_eq!(config.nu, 1e13);
        assert_eq!(config.temperature_k, 298.15);

        assert_eq!(config.species.len(), 5);
        assert_eq!(config.species[0].name, "O");
        assert_eq!(config.species[0].bit, 0x01);
        assert_eq!(config.species[0].gas.as_deref(), Some("O2gas"));
        assert_eq!(config.species[0].stoich, Some(0.5));
        assert_eq!(config.species[0].product.as_deref(), Some("Ostar"));
        assert_eq!(config.species[0].role, SpeciesRole::Dissociative);
        assert_eq!(config.species[0].oc20_ads_id, Some(0));

        let oh = &config.species[3];
        assert_eq!(oh.name, "OH");
        assert_eq!(oh.gas, None);
        assert_eq!(oh.stoich, None);
        assert_eq!(oh.role, SpeciesRole::ProductOnly);
        assert_eq!(oh.oc20_ads_id, None);

        assert_eq!(config.bimolecular.len(), 3);
        assert_eq!(config.bimolecular[0].key, "co_ox");
        assert_eq!(config.bimolecular[0].species_a, "O");
        assert_eq!(config.bimolecular[0].species_b, "CO");
        assert_eq!(
            config.bimolecular[0].direction,
            BimolecularDirection::Recombination
        );
        assert_eq!(config.bimolecular[0].gas.as_deref(), Some("CO2gas"));

        assert_eq!(
            config.bimolecular[2].direction,
            BimolecularDirection::Dissociative
        );
    }

    #[test]
    fn ignores_blank_lines_and_trailing_comments() {
        let text = "\
[system]
metal = Pd  # the target metal

# a comment line on its own

facet = 111
";
        let config = SimConfig::parse(text).unwrap();
        assert_eq!(config.metal.as_deref(), Some("Pd"));
        assert_eq!(config.facet, Some(111));
    }

    #[test]
    fn rejects_unknown_section() {
        assert!(SimConfig::parse("[bogus]\nfoo = bar\n").is_err());
    }

    #[test]
    fn rejects_key_before_any_section() {
        assert!(SimConfig::parse("metal = Pd\n").is_err());
    }

    #[test]
    fn rejects_line_without_equals() {
        assert!(SimConfig::parse("[system]\nmetal Pd\n").is_err());
    }

    #[test]
    fn rejects_unknown_system_key() {
        assert!(SimConfig::parse("[system]\nbogus = Pd\n").is_err());
    }

    #[test]
    fn rejects_non_integer_facet() {
        assert!(SimConfig::parse("[system]\nfacet = not_a_number\n").is_err());
    }

    #[test]
    fn rejects_non_numeric_bep_value() {
        assert!(SimConfig::parse("[bep]\nalpha = not_a_number\n").is_err());
    }

    #[test]
    fn rejects_species_row_with_wrong_field_count() {
        assert!(SimConfig::parse("[species]\nO = 0x01, O2gas\n").is_err());
    }

    #[test]
    fn rejects_species_bit_that_is_not_one_hot() {
        let text = "[species]\nO = 0x03, O2gas, 0.5, Ostar, dissociative, 0\n";
        assert!(SimConfig::parse(text).is_err());
    }

    #[test]
    fn rejects_species_bit_that_is_not_a_valid_byte() {
        let text = "[species]\nO = not_a_byte, O2gas, 0.5, Ostar, dissociative, 0\n";
        assert!(SimConfig::parse(text).is_err());
    }

    #[test]
    fn rejects_unknown_species_role() {
        let text = "[species]\nO = 0x01, O2gas, 0.5, Ostar, bogus_role, 0\n";
        assert!(SimConfig::parse(text).is_err());
    }

    #[test]
    fn rejects_duplicate_species_name() {
        let text = "\
[species]
O = 0x01, O2gas, 0.5, Ostar, dissociative, 0
O = 0x02, O2gas, 0.5, Ostar, dissociative, 0
";
        assert!(SimConfig::parse(text).is_err());
    }

    #[test]
    fn rejects_duplicate_species_bit() {
        let text = "\
[species]
O = 0x01, O2gas, 0.5, Ostar, dissociative, 0
H = 0x01, H2gas, 0.5, Hstar, dissociative, 1
";
        assert!(SimConfig::parse(text).is_err());
    }

    #[test]
    fn rejects_more_than_max_species_entries() {
        let mut text = String::from("[species]\n");
        for i in 0..(crate::layout::MAX_SPECIES + 1) {
            text.push_str(&format!(
                "S{i} = 0x{:02x}, gas{i}, 1.0, P{i}, molecular, -\n",
                1u16 << (i % 8)
            ));
        }
        assert!(SimConfig::parse(&text).is_err());
    }

    #[test]
    fn rejects_bimolecular_row_with_wrong_field_count() {
        assert!(SimConfig::parse("[bimolecular]\nco_ox = O, CO\n").is_err());
    }

    #[test]
    fn rejects_unknown_bimolecular_direction() {
        let text = "[bimolecular]\nco_ox = O, CO, sideways, CO2gas\n";
        assert!(SimConfig::parse(text).is_err());
    }

    #[test]
    fn rejects_bimolecular_entry_referencing_undeclared_species() {
        let text = "\
[species]
O = 0x01, O2gas, 0.5, Ostar, dissociative, 0

[bimolecular]
co_ox = O, CO, recombination, CO2gas
";
        assert!(SimConfig::parse(text).is_err());
    }
}
