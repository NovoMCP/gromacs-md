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
