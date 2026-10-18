"""Nucleotide-bound Mg2+ kept as a cofactor: site detection (intake), the
nucleotide's pH 7 form, and what the tleap build gets. Fixture: the 6OIM
GDP/Mg2+ site (KRAS G12C), with Ser17 and four shell waters."""

from pathlib import Path

import pytest

from intake.cofactors import kept_cofactors, magnesium_sites

FIX = Path(__file__).parent / "fixtures" / "cofactors"
SITE = (FIX / "6oim_site.pdb").read_text()


def _hetatm(serial, name, res, chain, seq, xyz, element):
    x, y, z = xyz
    return (f"HETATM{serial:5d} {name:<4s} {res:>3s} {chain}{seq:>4s}    {x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {element:>2s}\n")


def test_6oim_magnesium_is_kept_with_gdp_and_its_waters():
    (site,) = magnesium_sites(SITE)
    assert site.supported, site.refusal
    assert site.nucleotide == ("GDP", "A", "302")
    assert len(site.waters) == 4
    assert any(d.startswith("SER:A17:OG") for d in site.donors)
    kept = kept_cofactors(SITE)
    assert kept["nucleotides"] == [("GDP", "A", "302")]


def test_magnesium_without_a_nucleotide_is_not_kept():
    pdb = _hetatm(1, "MG", "MG", "A", "1", (0, 0, 0), "MG") + \
          _hetatm(2, "O", "HOH", "A", "2", (2.1, 0, 0), "O")
    (site,) = magnesium_sites(pdb)
    assert not site.supported and "no nucleotide" in site.refusal
    assert kept_cofactors(pdb)["magnesium"] == []


def test_magnesium_with_a_nitrogen_donor_is_not_kept():
    pdb = SITE + "ATOM   9999  NE2 HIS A  99      -2.512   2.588   2.300  1.00  0.00           N\n"
    (site,) = magnesium_sites(pdb)
    assert not site.supported and "NE2" in site.refusal


def test_no_magnesium_means_nothing_kept():
    pdb = "".join(l + "\n" for l in SITE.splitlines() if " MG " not in l)
    assert magnesium_sites(pdb) == []


# ---------------------------------------------------------------------------
# Builder side
# ---------------------------------------------------------------------------
Chem = pytest.importorskip("rdkit.Chem")
from gromacs_md import cofactor_build as cb  # noqa: E402


def test_cofactor_site_reads_crystal_coordinates():
    site = cb.cofactor_site(SITE)
    (gdp,) = site["nucleotides"]
    assert gdp["code"] == "GDP" and len(gdp["atoms"]) == 28
    assert len(site["waters"]) == 4
    assert site["magnesium"][0]["nucleotide"] == "GDP A302"


def test_gdp_is_built_as_the_trianion():
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    site = cb.cofactor_site(SITE)
    mol = cb.nucleotide_mol((FIX / "GDP.cif").read_text(), site["nucleotides"][0]["atoms"])
    assert Chem.GetFormalCharge(mol) == -3
    assert CalcMolFormula(mol) == "C10H12N5O11P2-3"
    # crystal heavy-atom positions are kept
    conf = mol.GetConformer()
    first = mol.GetAtomWithIdx(0)
    assert first.GetSymbol() == "P"
    assert tuple(round(v, 3) for v in conf.GetAtomPosition(0)) == \
        tuple(round(v, 3) for v in site["nucleotides"][0]["atoms"]["PB"])


def test_ions_pdb_has_magnesium_and_oxygen_only_waters():
    text = cb.ions_pdb(cb.cofactor_site(SITE))
    atoms = [l for l in text.splitlines() if l.startswith("HETATM")]
    assert [l[17:20].strip() for l in atoms] == ["MG"] + ["WAT"] * 4
    assert all(l[12:16].strip() in ("MG", "O") for l in atoms)


def test_tleap_script_loads_cofactors_after_the_protein():
    from gromacs_md import amber_build as ab

    cof = cb.Cofactors([ab.LigandParams(Path("GDP.mol2"), Path("GDP.frcmod"))], ["GDP", "MG"], "")
    script = ab._tleap_script(Path("prot.pdb"), [], None, cofactors=cof)
    assert "cof0 = loadmol2 GDP.mol2" in script
    assert "loadamberparams GDP.frcmod" in script
    assert f"cofions = loadpdb {ab.COFACTOR_IONS_PDB}" in script
    assert "sys = combine { prot cof0 cofions }" in script


def test_force_field_names_the_cofactor_parameters():
    from gromacs_md import amber_build as ab

    cof = cb.Cofactors([], ["GDP", "MG"], "")
    assert ab.force_field() == ab.FORCE_FIELD
    assert "Mg2+ Li/Merz 12-6" in ab.force_field(cof)
    assert ab.force_field(cof).endswith("built with tleap")


def test_native_site_skips_kept_cofactors():
    from gromacs_md.placement import native_ligand_site

    assert native_ligand_site(SITE)["resname"] == "GDP"
    assert native_ligand_site(SITE, exclude={"GDP"}) is None


def test_cofactors_count_as_receptor_in_build_groups(tmp_path):
    from types import SimpleNamespace as NS
    from gromacs_md import amber_build as ab

    counter = iter(range(100))

    def res(name, n):
        return NS(name=name, atoms=[NS(idx=next(counter), name=f"X{i}") for i in range(n)])

    structure = NS(residues=[res("ALA", 3), res("GDP", 2), res("MG", 1), res("LIG", 2), res("SOL", 3)])
    groups = ab.write_build_groups(structure, tmp_path / "groups.json", ["LIG"], 1,
                                   receptor_resnames=["GDP", "MG"])
    assert groups["protein"] == [[1, 6]]
    assert groups["ligand"] == [[7, 8]]
