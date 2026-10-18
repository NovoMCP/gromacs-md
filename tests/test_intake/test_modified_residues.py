"""Tests for modified residues in the protein chain (F10)."""

import pytest

from gromacs_md.pdb_handling import clean_pdb_protein_only, convert_selenomethionine
from intake.modified_residues import find_modified_residues, unsupported_modified_residues

try:
    from intake import classifier
except ImportError:  # pragma: no cover
    classifier = None

# MODRES records as in 1ATP (PKA) and a selenomethionine structure
MODRES_1ATP = (
    "MODRES 1ATP TPO E  197  THR  PHOSPHOTHREONINE                                   \n"
    "MODRES 1ATP SEP E  338  SER  PHOSPHOSERINE                                      \n"
)
MODRES_MSE = "MODRES 1XYZ MSE A    1  MET  SELENOMETHIONINE                                   \n"


def _atom(rec, serial, name, resname, chain, resnum, element):
    return (f"{rec:<6}{serial:>5} {name:<4} {resname} {chain}{resnum:>4}    "
            f"  10.000  10.000  10.000  1.00  0.00          {element:>2}\n")


# TRP 196 - TPO 197 - LEU 198 in chain E, then a ligand after the chain
CHAIN_WITH_TPO = (
    _atom("ATOM", 1, "CA", "TRP", "E", 196, "C")
    + _atom("HETATM", 2, "CA", "TPO", "E", 197, "C")
    + _atom("HETATM", 3, "P", "TPO", "E", 197, "P")
    + _atom("ATOM", 4, "CA", "LEU", "E", 198, "C")
    + _atom("HETATM", 5, "PG", "ATP", "E", 400, "P")
    + "TER\n"
    + _atom("HETATM", 6, "O", "HOH", "E", 501, "O")
    + "END\n"
)


def test_modres_records_are_read():
    found = find_modified_residues(MODRES_1ATP)
    assert [r.label() for r in found] == ["TPO E197 phosphothreonine", "SEP E338 phosphoserine"]
    assert found[0].parent == "THR"


def test_selenomethionine_is_supported():
    assert unsupported_modified_residues(MODRES_MSE) == []
    assert len(unsupported_modified_residues(MODRES_1ATP + MODRES_MSE)) == 2


def test_cleaning_refuses_a_residue_inside_the_chain():
    with pytest.raises(ValueError, match="TPO E197"):
        clean_pdb_protein_only(CHAIN_WITH_TPO)


def test_ligands_and_waters_after_the_chain_are_still_stripped():
    pdb = CHAIN_WITH_TPO.replace("TPO", "THR").replace("HETATM    2", "ATOM      2") \
                        .replace("HETATM    3", "ATOM      3")
    clean = clean_pdb_protein_only(pdb)
    assert "ATP" not in clean and "HOH" not in clean and "THR" in clean


def test_selenomethionine_becomes_methionine():
    pdb = (MODRES_MSE
           + _atom("HETATM", 1, "CA", "MSE", "A", 1, "C")
           + _atom("HETATM", 2, "SE", "MSE", "A", 1, "SE")
           + _atom("ATOM", 3, "CA", "ALA", "A", 2, "C"))
    out, n = convert_selenomethionine(pdb)
    assert n == 1 and "MSE" not in out and "MODRES" not in out
    se_line = out.splitlines()[1]
    assert se_line.startswith("ATOM  ") and se_line[12:16] == " SD " and se_line[17:20] == "MET"
    assert se_line[76:78] == " S"
    assert clean_pdb_protein_only(out).count("MET") == 2


@pytest.mark.asyncio
@pytest.mark.skipif(classifier is None, reason="intake dependencies not installed")
async def test_classifier_refuses_phosphorylated_residues():
    pdb = MODRES_1ATP + CHAIN_WITH_TPO
    d = await classifier.classify_structure(pdb_content=pdb)
    assert d.route == "refused" and d.suggested_branch == "modified_residues"
    assert "TPO E197 phosphothreonine" in d.reasons[0]
    assert "refused with any ligand" in d.reasons[0]
    assert d.profile.ptm_residues == ["TPO E197 phosphothreonine", "SEP E338 phosphoserine"]


TPO_LINKS = (
    "LINK         C   TRP E 196                 N   TPO E 197     1555   1555  1.32  \n"
    "LINK         C   TPO E 197                 N   LEU E 198     1555   1555  1.30  \n"
)


def test_peptide_links_of_modified_residues_are_not_covalent_ligands():
    from intake.covalent import find_covalent_ligand_links
    assert find_covalent_ligand_links(MODRES_1ATP + TPO_LINKS) == []
    assert len(find_covalent_ligand_links(TPO_LINKS)) == 2  # without MODRES they look like ligands


@pytest.mark.asyncio
@pytest.mark.skipif(classifier is None, reason="intake dependencies not installed")
async def test_1atp_is_refused_for_its_modified_residues_not_as_covalent():
    d = await classifier.classify_structure(pdb_content=MODRES_1ATP + TPO_LINKS + CHAIN_WITH_TPO)
    assert d.suggested_branch == "modified_residues"
