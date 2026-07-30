"""Tests for the donor-atom heuristic classifier."""

import pytest

try:
    from intake.heuristics import classify_by_donors
except ImportError:
    pytest.skip("intake dependencies not installed", allow_module_level=True)


def test_zinc_3his_is_catalytic():
    # 3xHis → catalytic zinc site (e.g., carbonic anhydrase)
    ligating = ["HIS:94:NE2", "HIS:96:NE2", "HIS:119:NE2"]
    role, reason = classify_by_donors("ZN", ligating)
    assert role == "catalytic"


def test_zinc_cys4_is_structural():
    # 4xCys → structural zinc finger
    ligating = ["CYS:10:SG", "CYS:13:SG", "CYS:30:SG", "CYS:33:SG"]
    role, reason = classify_by_donors("ZN", ligating)
    assert role == "structural"


def test_zinc_cys2his2_is_structural():
    # C2H2 zinc finger
    ligating = ["CYS:10:SG", "CYS:13:SG", "HIS:30:NE2", "HIS:33:NE2"]
    role, reason = classify_by_donors("ZN", ligating)
    assert role == "structural"


def test_magnesium_is_catalytic():
    # Mg is always catalytic (kinases, GTPases)
    ligating = ["ASP:57:OD1", "THR:35:OG1", "HOH:401:O"]
    role, reason = classify_by_donors("MG", ligating)
    assert role == "catalytic"


def test_calcium_efhand_is_structural():
    # Ca with 3+ Asp/Glu → EF-hand structural
    ligating = ["ASP:20:OD1", "ASP:22:OD1", "GLU:31:OE1", "ASN:24:OD1"]
    role, reason = classify_by_donors("CA", ligating)
    assert role == "structural"


def test_iron_is_unknown():
    # Fe defaults to unknown (needs specialized handling)
    ligating = ["HIS:87:NE2", "HIS:204:NE2"]
    role, reason = classify_by_donors("FE", ligating)
    assert role == "unknown"
