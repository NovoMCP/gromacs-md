"""Tests for protein-ligand H-bond persistence from `gmx hbond -pf` (F4)."""

from gromacs_md.analysis import _gro_atom_labels, _hbond_frames, _summarize_hbonds

# Output layout of GROMACS 2026 `gmx hbond -r Protein -t Ligand -pf -o`
NDX = """\
[ Protein ]
   1    2    3
[ donors_hydrogens_Protein ]
   1    2
[ hbonds_Protein-Ligand_frame_0 ]
      1      2     10
     20     21     11
     20     22     11
[ hbonds_Protein-Ligand_frame_1 ]
      1      2     10
[ hbonds_Protein-Ligand_frame_2 ]
      1      2     10
     12     13      5
"""

GRO = """\
t
   22
   25ASP      N    1   0.000   0.000   0.000
   25ASP      H    2   0.000   0.000   0.000
   25ASP     CA    3   0.000   0.000   0.000
   26GLY      O    5   0.000   0.000   0.000
  901MK1     O1   10   0.000   0.000   0.000
  901MK1     N2   11   0.000   0.000   0.000
  901MK1     N3   12   0.000   0.000   0.000
  901MK1     H3   13   0.000   0.000   0.000
"""


def test_frames_merge_bonds_through_two_hydrogens():
    frames = _hbond_frames(NDX)
    assert frames == [{(1, 10), (20, 11)}, {(1, 10)}, {(1, 10), (12, 5)}]


def test_summary_ranks_bonds_and_grades_on_the_most_persistent(tmp_path):
    labels = {1: "ASP25:N", 5: "GLY26:O", 10: "MK1901:O1", 11: "MK1901:N2", 12: "MK1901:N3", 20: "ILE50:N"}
    # 4 frames in hbnum.xvg; the 4th had no H-bonds and no section
    out = _summarize_hbonds(_hbond_frames(NDX), 4, labels, ligand_atoms={10, 11, 12, 13})
    assert out["n_frames"] == 4 and out["max_count"] == 2
    assert out["any_contact_persistence"] == 0.75
    first = out["bonds"][0]
    assert first == {"protein_atom": "ASP25:N", "ligand_atom": "MK1901:O1",
                     "donor": "protein", "persistence": 0.75}
    ligand_donor = next(b for b in out["bonds"] if b["donor"] == "ligand")
    assert ligand_donor["protein_atom"] == "GLY26:O" and ligand_donor["persistence"] == 0.25
    assert out["quality"] == "Moderate"  # 0.75 is not above 0.75


def test_gro_labels(tmp_path):
    gro = tmp_path / "x.gro"
    gro.write_text(GRO + "   1.0 1.0 1.0\n")
    labels = _gro_atom_labels(gro)
    assert labels[1] == "ASP25:N" and labels[5] == "MK1901:O1"


def test_hbonds_inside_one_residue_are_not_contacts():
    # a covalent adduct's backbone N to its own ligand carbonyl
    labels = {1: "CVX481:N", 2: "CVX481:OAC", 3: "MET477:N", 4: "CVX481:N1"}
    frames = [{(1, 2), (3, 4)}, {(1, 2), (3, 4)}]
    s = _summarize_hbonds(frames, 2, labels, ligand_atoms={2, 4})
    assert [b["protein_atom"] for b in s["bonds"]] == ["MET477:N"]
    assert s["mean_count"] == 1.0
