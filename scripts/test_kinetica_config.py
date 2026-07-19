"""Tests for `kinetica_config.py`, mirroring `src/config.rs`'s own test
module case-for-case (same example config text, same error cases) so the
two parsers are verified against the same behavioral contract rather than
independently invented ones. Stdlib `unittest` only -- no new dependency,
same reasoning as `kinetica_config.py` itself.

Run with: python3 -m unittest scripts.test_kinetica_config -v
(from the repo root) or python3 scripts/test_kinetica_config.py directly.
"""

import unittest

from kinetica_config import MAX_SPECIES, parse_config, species_index

PD111_EXAMPLE = """\
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
"""


class ParsesTheFullPd111Example(unittest.TestCase):
    def setUp(self):
        self.config = parse_config(PD111_EXAMPLE)

    def test_system_and_bep(self):
        self.assertEqual(self.config.metal, "Pd")
        self.assertEqual(self.config.facet, 111)
        self.assertEqual(self.config.alpha, 0.87)
        self.assertEqual(self.config.beta_ev, 0.0)
        self.assertEqual(self.config.nu, 1e13)
        self.assertEqual(self.config.temperature_k, 298.15)

    def test_first_species_row(self):
        self.assertEqual(len(self.config.species), 5)
        o = self.config.species[0]
        self.assertEqual(o.name, "O")
        self.assertEqual(o.bit, 0x01)
        self.assertEqual(o.gas, "O2gas")
        self.assertEqual(o.stoich, 0.5)
        self.assertEqual(o.product, "Ostar")
        self.assertEqual(o.role, "dissociative")
        self.assertEqual(o.oc20_ads_id, 0)

    def test_product_only_species_row(self):
        oh = self.config.species[3]
        self.assertEqual(oh.name, "OH")
        self.assertIsNone(oh.gas)
        self.assertIsNone(oh.stoich)
        self.assertEqual(oh.role, "product_only")
        self.assertIsNone(oh.oc20_ads_id)

    def test_bimolecular_rows(self):
        self.assertEqual(len(self.config.bimolecular), 3)
        co_ox = self.config.bimolecular[0]
        self.assertEqual(co_ox.key, "co_ox")
        self.assertEqual(co_ox.species_a, "O")
        self.assertEqual(co_ox.species_b, "CO")
        self.assertEqual(co_ox.direction, "recombination")
        self.assertEqual(co_ox.gas, "CO2gas")
        self.assertEqual(self.config.bimolecular[2].direction, "dissociative")

    def test_species_index_lookup(self):
        self.assertEqual(species_index(self.config, "CO"), 2)
        self.assertIsNone(species_index(self.config, "Xenon"))


class IgnoresBlankLinesAndTrailingComments(unittest.TestCase):
    def test(self):
        text = "[system]\nmetal = Pd  # the target metal\n\n# a comment line on its own\n\nfacet = 111\n"
        config = parse_config(text)
        self.assertEqual(config.metal, "Pd")
        self.assertEqual(config.facet, 111)


class RejectsMalformedInput(unittest.TestCase):
    def assert_rejects(self, text):
        with self.assertRaises(ValueError):
            parse_config(text)

    def test_rejects_unknown_section(self):
        self.assert_rejects("[bogus]\nfoo = bar\n")

    def test_rejects_key_before_any_section(self):
        self.assert_rejects("metal = Pd\n")

    def test_rejects_line_without_equals(self):
        self.assert_rejects("[system]\nmetal Pd\n")

    def test_rejects_unknown_system_key(self):
        self.assert_rejects("[system]\nbogus = Pd\n")

    def test_rejects_non_integer_facet(self):
        self.assert_rejects("[system]\nfacet = not_a_number\n")

    def test_rejects_non_numeric_bep_value(self):
        self.assert_rejects("[bep]\nalpha = not_a_number\n")

    def test_rejects_species_row_with_wrong_field_count(self):
        self.assert_rejects("[species]\nO = 0x01, O2gas\n")

    def test_rejects_species_bit_that_is_not_one_hot(self):
        self.assert_rejects("[species]\nO = 0x03, O2gas, 0.5, Ostar, dissociative, 0\n")

    def test_rejects_species_bit_that_is_not_a_valid_byte(self):
        self.assert_rejects("[species]\nO = not_a_byte, O2gas, 0.5, Ostar, dissociative, 0\n")

    def test_rejects_unknown_species_role(self):
        self.assert_rejects("[species]\nO = 0x01, O2gas, 0.5, Ostar, bogus_role, 0\n")

    def test_rejects_duplicate_species_name(self):
        text = (
            "[species]\n"
            "O = 0x01, O2gas, 0.5, Ostar, dissociative, 0\n"
            "O = 0x02, O2gas, 0.5, Ostar, dissociative, 0\n"
        )
        self.assert_rejects(text)

    def test_rejects_duplicate_species_bit(self):
        text = (
            "[species]\n"
            "O = 0x01, O2gas, 0.5, Ostar, dissociative, 0\n"
            "H = 0x01, H2gas, 0.5, Hstar, dissociative, 1\n"
        )
        self.assert_rejects(text)

    def test_rejects_more_than_max_species_entries(self):
        lines = ["[species]"]
        for i in range(MAX_SPECIES + 1):
            lines.append(f"S{i} = {1 << (i % 8):#04x}, gas{i}, 1.0, P{i}, molecular, -")
        self.assert_rejects("\n".join(lines) + "\n")

    def test_rejects_bimolecular_row_with_wrong_field_count(self):
        self.assert_rejects("[bimolecular]\nco_ox = O, CO\n")

    def test_rejects_unknown_bimolecular_direction(self):
        self.assert_rejects("[bimolecular]\nco_ox = O, CO, sideways, CO2gas\n")

    def test_rejects_bimolecular_entry_referencing_undeclared_species(self):
        text = (
            "[species]\n"
            "O = 0x01, O2gas, 0.5, Ostar, dissociative, 0\n"
            "\n"
            "[bimolecular]\n"
            "co_ox = O, CO, recombination, CO2gas\n"
        )
        self.assert_rejects(text)


if __name__ == "__main__":
    unittest.main()
