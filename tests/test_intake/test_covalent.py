"""Tests for covalent-ligand refusal and the membrane-branch gate."""

import pytest

try:
    from intake import classifier, opm
    from intake.covalent import find_covalent_ligand_links
except ImportError:
    pytest.skip("intake dependencies not installed", allow_module_level=True)

# LINK records as they appear in 6LU7, 4EY7, 3HS4 and 2RH1
N3_ON_CYS145 = "LINK         SG  CYS A 145                 C20 PJE C   5     1555   1555  1.79  "
NAG_ON_ASN = "LINK         ND2 ASN A 350                 C1  NAG A 601     1555   1555  1.44  "
ZN_ON_HIS = "LINK         NE2 HIS A  94                ZN    ZN A 301     1555   1555  2.05  "
PALMITOYL = "LINK         SG  CYS A 341                 C1  PLM A 415     1555   1555  1.62  "


def test_a_ligand_bonded_to_a_cysteine_is_found():
    assert find_covalent_ligand_links(N3_ON_CYS145) == ["PJE C5 C20 - CYS A145 SG (1.79 A)"]


@pytest.mark.parametrize("line", [NAG_ON_ASN, ZN_ON_HIS, PALMITOYL])
def test_glycans_metals_and_lipidation_are_not_covalent_ligands(line):
    assert find_covalent_ligand_links(line) == []


PDB = """\
ATOM      1  N   CYS A 145      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  CYS A 145      11.400  10.000  10.000  1.00  0.00           C
ATOM      3  C   CYS A 145      12.000  11.400  10.000  1.00  0.00           C
ATOM      4  O   CYS A 145      13.200  11.500  10.000  1.00  0.00           O
ATOM      5  CB  CYS A 145      11.900   9.200  11.200  1.00  0.00           C
ATOM      6  SG  CYS A 145      11.300   7.500  11.200  1.00  0.00           S
HETATM    7  C20 PJE C   5      11.900   6.800  12.700  1.00  0.00           C
END
"""


@pytest.mark.asyncio
async def test_classifier_refuses_a_covalent_ligand():
    d = await classifier.classify_structure(pdb_content=N3_ON_CYS145 + "\n" + PDB)
    assert d.route == "refused" and d.suggested_branch == "covalent_ligand"
    assert "CYS A145" in d.reasons[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,route,branch", [
    (False, "refused", "charmm36m_membrane"),
    (True, "run_membrane", None),
])
async def test_membrane_proteins_are_refused_unless_the_branch_is_enabled(monkeypatch, enabled, route, branch):
    async def in_opm(pdb_id, http_client=None, redis_client=None):
        return True, "opm_api", None

    monkeypatch.setattr(opm, "check_membrane", in_opm)
    monkeypatch.setattr(classifier, "MEMBRANE_BRANCH_ENABLED", enabled)
    d = await classifier.classify_structure(pdb_content=PDB, pdb_id="2RH1")
    assert (d.route, d.suggested_branch) == (route, branch)
    assert d.reasons[0].startswith("membrane protein")
