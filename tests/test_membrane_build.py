"""Tests for membrane orientation, packing inputs, staged equilibration and metrics."""

import math
from pathlib import Path

import pytest

from gromacs_md import analysis, membrane_build as mb, topology

FIXTURES = Path(__file__).parent / "fixtures"


def _rotate_z(pdb_text, angle, shift):
    c, s = math.cos(angle), math.sin(angle)
    out = []
    for l in pdb_text.splitlines():
        if l.startswith(("ATOM", "HETATM")):
            x, y, z = float(l[30:38]), float(l[38:46]), float(l[46:54])
            x, y = c * x - s * y + shift[0], s * x + c * y + shift[1]
            l = f"{l[:30]}{x:8.3f}{y:8.3f}{z + shift[2]:8.3f}{l[54:]}"
        out.append(l)
    return "\n".join(out) + "\n"


def test_superpose_moves_everything_into_the_reference_frame():
    pytest.importorskip("numpy")
    crambin = (FIXTURES / "1crn_protein.pdb").read_text()
    reference = _rotate_z(crambin, 0.7, (5.0, -3.0, 12.0))
    moved, rmsd = mb.superpose_onto(crambin, reference)
    assert rmsd < 0.01
    first = lambda t: [l for l in t.splitlines() if l.startswith("ATOM")][10]
    assert first(moved)[30:54] == first(reference)[30:54]


def test_superpose_refuses_a_different_structure():
    pytest.importorskip("numpy")
    crambin = (FIXTURES / "1crn_protein.pdb").read_text()
    scrambled = "\n".join(l[:30] + f"{i % 7 * 3.1:8.3f}{i % 5 * 2.7:8.3f}{i % 3 * 4.2:8.3f}" + l[54:]
                          if l.startswith("ATOM") else l for i, l in enumerate(crambin.splitlines()))
    with pytest.raises(mb.MembraneBuildError, match="doesn't match"):
        mb.superpose_onto(crambin, scrambled)


def test_opm_half_thickness():
    assert mb.opm_half_thickness("REMARK      1/2 of bilayer thickness:   15.9   \n") == 15.9
    assert mb.opm_half_thickness("HEADER\n") is None


def test_complex_pdb_names_the_ligand_after_its_mol2_residue(tmp_path):
    from gromacs_md.amber_build import LigandParams, PreparedProtein
    mol2 = tmp_path / "l.mol2"
    mol2.write_text("@<TRIPOS>ATOM\n      1 C1   1.0 2.0 3.0 ca 1 CAU 0.0\n"
                    "      2 CL1  4.0 5.0 6.0 cl 1 CAU 0.0\n      3 CL2  4.0 5.0 6.0 c3 1 CAU 0.0\n@<TRIPOS>BOND\n")
    prot = PreparedProtein("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\nTER\nEND\n",
                           [], [("A", 1)])
    text = mb.complex_pdb(prot, LigandParams(mol2, tmp_path / "l.frcmod"))
    het = [l for l in text.splitlines() if l.startswith("HETATM")]
    assert [l[17:20] for l in het] == ["CAU"] * 3
    assert [l[76:78].strip() for l in het] == ["C", "Cl", "C"]  # CL2 is a carbon


def test_membrane_stages_release_restraints_and_end_semi_isotropic(tmp_path):
    names, fcs = [], []
    for k, stage in enumerate(topology.MEMBRANE_STAGES):
        text = topology.write_membrane_stage_mdp(tmp_path, "x", stage, 303.15, 1.0, first=(k == 0)).read_text()
        names.append(stage[0])
        fcs.append(stage[4:])
        assert ("gen_vel                 = yes" in text) == (k == 0)
        assert ("semiisotropic" in text) == stage[1]
    # every force constant only goes down, and the last stage frees the side chains and lipids
    for a, b in zip(fcs, fcs[1:]):
        assert all(y <= x for x, y in zip(a, b))
    assert fcs[-1][1:] == (0, 0)
    total_ps = sum(s[3] for s in topology.MEMBRANE_STAGES)
    assert total_ps == 1875


def test_production_mdp_is_semi_isotropic_for_membranes(tmp_path):
    topology.write_md_mdp(tmp_path, "m", 1.0, 303.15, 1.0, membrane=True)
    text = (tmp_path / "m_md.mdp").read_text()
    assert "pcoupltype              = semiisotropic" in text and "ref_p                   = 1.0 1.0" in text
    assert "DispCorr                = EnerPres" in text
    topology.write_md_mdp(tmp_path, "s", 1.0, 300.0, 1.0)
    soluble = (tmp_path / "s_md.mdp").read_text()
    assert "pcoupltype              = isotropic" in soluble and "ref_p                   = 1.0\n" in soluble


GRO = """\
bilayer
    7
    1MET      N    1   2.000   2.000   3.000
    2CAU     C1    2   2.000   2.000   4.000
    3PA     C12    3   2.000   2.100   4.250
    4PC     P31    4   5.000   5.000   5.500
    5OL     C12    5   5.000   5.000   5.000
    7PC       P    6   1.000   1.000   1.500
    9PC       P    7   1.000   1.000   1.600
   10.00000  10.00000   7.00000
"""


def test_membrane_metrics_thickness_leaflets_and_pocket_lipids(tmp_path, monkeypatch):
    (tmp_path / "x_md.gro").write_text(GRO)
    (tmp_path / "x_build_groups.json").write_text('{"protein": [[1, 1]], "ligand": [[2, 2]]}')

    def no_gmx(*a, **k):
        raise OSError("gmx not here")
    monkeypatch.setattr(analysis.subprocess, "run", no_gmx)
    m = analysis.membrane_metrics(tmp_path, "x")
    assert m["lipids"] == 3 and m["leaflets"] == {"upper": 1, "lower": 2}
    assert m["thickness_pp_angstrom"] == pytest.approx((5.5 - 1.55) * 10, abs=0.01)
    # the PA tail 2.5 Å from the ligand counts its lipid (head PC 4) once
    assert m["lipids_contacting_ligand"] == 1
    assert "error" in m["area_per_lipid_angstrom2"]


def test_packed_box_is_the_union_of_regions_plus_tolerance():
    inp = """tolerance 2.0
structure POPC.pdb
  inside box -52.98 -52.98 -23.0 52.98 52.98 0.0
end structure
structure WAT.pdb
  inside box -52.98 -52.98 -80.22 52.98 52.98 -23.0
end structure
structure WAT.pdb
  inside box -52.98 -52.98 23.0 52.98 52.98 49.76
end structure
"""
    assert mb.packed_box(inp) == pytest.approx((107.96, 107.96, 131.98))
    with pytest.raises(mb.MembraneBuildError):
        mb.packed_box("tolerance 2.0\n")


def test_membrane_estimate_scales_from_the_protein_rate_until_measured():
    from gromacs_md.queue import membrane_runtime_minutes
    # 181L-class protein at 5.2 min/ns -> 15.6 min/ns for a bilayer system
    assert membrane_runtime_minutes(1.0, None, 5.2) == math.ceil(15 + 3.25 * 15.6 + 5)
    assert membrane_runtime_minutes(10.0, 20.0, 5.2) == math.ceil(15 + 12.25 * 20.0 + 5)


def test_construct_notes_name_fusions_and_dropped_hetatms():
    from gromacs_md.pdb_handling import construct_notes
    header = ("COMPND    MOL_ID: 1;\n"
              "COMPND   2 MOLECULE: BETA-2-ADRENERGIC RECEPTOR/T4-LYSOZYME CHIMERA;\n"
              "COMPND   3 CHAIN: A;\n"
              "HETATM    1  C1  CLR A 412       0.000   0.000   0.000  1.00  0.00           C\n"
              "HETATM    2  C1  CAU A 500       0.000   0.000   0.000  1.00  0.00           C\n"
              "HETATM    3  S   SO4 A 600       0.000   0.000   0.000  1.00  0.00           S\n")
    notes = construct_notes(header, ligand_smiles="CC")
    assert "fusion (BETA-2-ADRENERGIC RECEPTOR/T4-LYSOZYME CHIMERA)" in notes[0]
    assert notes[1].startswith("crystal HETATM groups not simulated: CAU, CLR (the submitted ligand")
    plain = "COMPND   2 MOLECULE: ALPHA/BETA HYDROLASE;\n"
    assert construct_notes(plain) == []


def test_results_path_is_reported_on_the_host(tmp_path, monkeypatch):
    from gromacs_md import s3_storage
    monkeypatch.setattr(s3_storage, "RESULTS_HOST_DIR", "/home/user/station/md-results")
    (tmp_path / "ws").mkdir()
    out = s3_storage.save_results_locally(tmp_path / "ws", "gro_x_1", {"output_files": []},
                                          results_dir=str(tmp_path / "res"))
    assert out == "/home/user/station/md-results/gro_x_1"
    assert (tmp_path / "res" / "gro_x_1" / "result.json").exists()
