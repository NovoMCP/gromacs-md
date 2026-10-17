"""Tests for Pfam role lookup."""

import pytest

try:
    from intake.pfam import lookup_role, table_size
except ImportError:
    pytest.skip("intake dependencies not installed", allow_module_level=True)


def test_lookup_structural_zinc_finger():
    # zf-C2H2 is a structural zinc finger
    role = lookup_role("zf-C2H2")
    assert role == "structural"


def test_lookup_catalytic_carbonic_anhydrase():
    role = lookup_role("Carb_anhydrase")
    assert role == "catalytic"


def test_lookup_electron_cytochrome():
    role = lookup_role("Cytochrom_C")
    assert role == "electron"


def test_lookup_transport_ferritin():
    role = lookup_role("Ferritin")
    assert role == "transport"


def test_lookup_unknown_family():
    role = lookup_role("NonexistentFamily_XYZ123")
    assert role is None


def test_lookup_case_insensitive():
    role1 = lookup_role("ZF-C2H2")
    role2 = lookup_role("zf-c2h2")
    assert role1 == role2


def test_table_loaded():
    assert table_size() > 0


def test_lookup_trypsin_calcium_is_structural():
    # trypsin's Ca loop is far from the catalytic triad; not a QM/MM case
    assert lookup_role("Trypsin") == "structural"


def test_lookup_family_subtype_by_prefix():
    assert lookup_role("Peptidase_M10A") == "catalytic"
    assert lookup_role("EF_hand_3") == "structural"


@pytest.mark.parametrize("family", ["GAF", "PAS", "RasGEF", "ArfGap", "Gag_p24", "C2"])
def test_short_keys_do_not_match_unrelated_families(family):
    # "ga", "ras", "arf" are table keys; "pas" and "c2" sit inside longer ones
    assert lookup_role(family) is None
