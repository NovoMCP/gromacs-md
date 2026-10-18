"""Tests for the tleap builder's protein prep, ions, restraints and parity parsing.

tleap and sander aren't installed in CI; the builds they run are checked on
the GB10 (181L + benzene: every energy term within 0.2 kcal/mol, PME
electrostatics within 1e-4).
"""

import json
from pathlib import Path

import pytest

from gromacs_md import amber_build as ab
from gromacs_md import analysis

FIXTURES = Path(__file__).parent / "fixtures"

SANDER_OUT = """\
 NSTEP =        0   TIME(PS) =       0.000  TEMP(K) =     0.00  PRESS =     0.0
 Etot   =   -132306.7812  EKtot   =         0.0000  EPtot      =   -132306.7812
 BOND   =       616.4406  ANGLE   =       421.6789  DIHED      =      1887.2832
 1-4 NB =       939.6616  1-4 EEL =      5657.9419  VDWAALS    =     16134.8609
 EELEC  =   -157964.6483  EHBOND  =         0.0000  RESTRAINT  =         0.0000
 Ewald error estimate:   0.4703E-03
"""

GMX_LOG = """\
   Energies (kJ/mol)
           Bond          Angle    Proper Dih. Per. Imp. Dih.          LJ-14
    2.57916e+03    1.76431e+03    7.83895e+03    5.74434e+01    3.93154e+03
     Coulomb-14        LJ (SR)  Disper. corr.   Coulomb (SR)   Coul. recip.
    1.42042e+04    6.99816e+04   -2.47320e+03   -7.01718e+05    4.07236e+04
      Potential    Kinetic En.   Total Energy    Temperature Pres. DC (bar)
   -5.63111e+05    5.22079e+03   -5.57890e+05    8.78125e+00    0.00000e+00

"""


def test_sander_and_gmx_energies_parse_to_the_same_terms():
    amber = ab._parse_sander(SANDER_OUT)
    gmx = ab._parse_gmx_log(GMX_LOG)
    assert set(amber) == set(gmx) == set(ab.PARITY_TOLERANCE)
    assert amber["bond"] == pytest.approx(616.4406)
    assert amber["vdw_14"] == pytest.approx(939.6616)
    # kJ -> kcal; proper + improper dihedrals; LJ (SR) + dispersion correction
    assert gmx["bond"] == pytest.approx(616.434, abs=0.01)
    assert gmx["dihedral"] == pytest.approx((7838.95 + 57.4434) / 4.184, abs=0.01)
    assert gmx["vdw"] == pytest.approx((69981.6 - 2473.20) / 4.184, abs=0.1)
    assert gmx["elec"] == pytest.approx((-701718 + 40723.6) / 4.184, abs=1)


def test_ion_counts_neutralize_then_add_salt():
    # 181L in its tleap box: 583524 Å3 at +8 -> 53 NaCl pairs, 8 more Cl-
    assert ab.ion_counts(583523.86, 8.0) == (53, 61)
    assert ab.ion_counts(583523.86, -3.0) == (56, 53)
    assert ab.ion_counts(1000.0, 0.0) == (0, 0)


def test_volume_and_charge_read_from_the_tleap_log():
    log = ("  Volume: 583523.860 A^3 \n  Added 15095 residues.\n"
           "Total unperturbed charge:   8.000000\nTotal perturbed charge:     8.000000\n")
    assert ab._parse_volume_and_charge(log) == (583523.86, 8.0)
    with pytest.raises(ab.AmberBuildError):
        ab._parse_volume_and_charge("Total unperturbed charge: 0.0\n")


def test_leap_errors_catch_unknown_atoms_and_fatal_lines():
    log = ("Loading PDB file: ./p.pdb\n"
           "  Created a new atom named: HD1 within residue: .R<HIE 31>\n"
           "FATAL:  Atom .R<HIE 31>.A<HD1 21> does not have a type.\n"
           "Exiting LEaP: Errors = 1; Warnings = 0; Notes = 0.\n")
    errors = ab._leap_errors(log)
    assert len(errors) == 2
    assert not ab._leap_errors("Exiting LEaP: Errors = 0; Warnings = 1; Notes = 0.\n")


def test_tleap_script_bonds_disulfides_and_only_adds_ions_it_needs(tmp_path):
    lig = ab.LigandParams(tmp_path / "l.mol2", tmp_path / "l.frcmod")
    probe = ab._tleap_script(tmp_path / "p.pdb", [(3, 40)], lig)
    assert "bond prot.3.SG prot.40.SG" in probe
    assert "loadamberparams l.frcmod" in probe and "combine { prot lig }" in probe
    assert "saveamberparm" not in probe and "addIons" not in probe
    build = ab._tleap_script(tmp_path / "p.pdb", [], None, ions=(0, 8), out_prefix="x_amber")
    # a count of 0 would mean "neutralize" to addIonsRand, so it is left out
    assert "addIonsRand sys Cl- 8\n" in build
    assert "saveamberparm sys x_amber.prmtop x_amber.inpcrd" in build
    assert "addIonsRand" not in ab._tleap_script(None, [], lig, ions=(0, 0), out_prefix="x")


TOP = """\
[ moleculetype ]
; Name            nrexcl
system1          3

[ atoms ]
;   nr       type  resnr residue  atom   cgnr    charge       mass
     1         N3      1    MET      N      1 0.15920000  14.010000   ; qtot 0.159
     2          H      1    MET     H1      2 0.19840000   1.008000   ; qtot 0.358
     3         CX      1    MET     CA      3 0.02210000  12.010000   ; qtot 0.380

[ bonds ]
     1      2     1   0.10100 363171.200000

[ moleculetype ]
; Name            nrexcl
SOL          3

[ atoms ]
    1         OW      1    SOL     OW      1 -0.83400000  16.000000
    2         HW      1    SOL    HW1      2 0.41700000   1.008000
    3         HW      1    SOL    HW2      3 0.41700000   1.008000

[ moleculetype ]
; Name            nrexcl
BNZ          3

[ atoms ]
    1         ca      1    BNZ      C      1 -0.13000000  12.010000
    2         ha      1    BNZ      H      2 0.13000000   1.008000

[ system ]
x

[ molecules ]
system1  1
BNZ      1
SOL      100
"""


def test_position_restraints_on_solute_heavy_atoms_only():
    out = ab.add_position_restraints(TOP)
    blocks = out.split("[ moleculetype ]")
    protein, water, ligand = blocks[1], blocks[2], blocks[3]
    assert protein.count("#ifdef POSRES") == 1
    restrained = [l.split()[0] for l in protein.split("[ position_restraints ]")[1].splitlines()[2:]
                  if l.strip() and not l.startswith("#")]
    assert restrained == ["1", "3"]  # N and CA, not H1
    assert "POSRES" not in water
    assert ligand.count("#ifdef POSRES") == 1
    # each block sits inside its own moleculetype, before the next one starts
    assert out.index("#endif") < out.index("SOL          3")
    assert "[ system ]" in out and out.rstrip().endswith("SOL      100")


def test_ranges_compress_atom_numbers():
    assert ab._ranges([1, 2, 3, 7, 8, 10]) == [[1, 3], [7, 8], [10, 10]]
    assert ab._ranges([]) == []


def test_prepare_protein_crambin_disulfides_and_numbering():
    pytest.importorskip("openmm")
    prot = ab.prepare_protein((FIXTURES / "1crn_protein.pdb").read_text())
    assert prot.disulfides == [(3, 40), (4, 32), (16, 26)]
    assert len(prot.residue_ids) == 46 and prot.residue_ids[0] == ("A", 1)
    names = {line[17:20] for line in prot.pdb_text.splitlines() if line.startswith("ATOM")}
    assert "CYX" in names and "CYS" not in names
    assert not any(line[76:78].strip() == "H" for line in prot.pdb_text.splitlines()
                   if line.startswith("ATOM"))
    assert prot.pdb_text.count("TER") == 1 and not prot.notes


def test_prepare_protein_splits_segments_at_a_chain_break():
    pytest.importorskip("openmm")
    lines = (FIXTURES / "1crn_protein.pdb").read_text().splitlines(keepends=True)
    gapped = "".join(l for l in lines if not (l.startswith("ATOM") and l[22:26].strip() in ("20", "21")))
    prot = ab.prepare_protein(gapped)
    assert prot.pdb_text.count("TER") == 2
    assert any("chain breaks before A22" in n for n in prot.notes)
    assert ("A", 22) in prot.residue_ids and ("A", 20) not in prot.residue_ids


def test_build_info_defaults_to_pdb2gmx(tmp_path):
    assert ab.load_build_info(tmp_path, "x")["builder"] == "pdb2gmx"
    ab.build_info_path(tmp_path, "x").write_text(json.dumps({"builder": "tleap", "force_field": "ff"}))
    assert ab.load_build_info(tmp_path, "x")["builder"] == "tleap"


GRO = """\
t
    6
    1MET      N    1   0.000   0.000   0.000
    1MET     CA    2   0.100   0.000   0.000
    2PA      C1    3   0.200   0.000   0.000
    3CVX     SG    4   0.300   0.000   0.000
    3CVX    C10    5   0.400   0.000   0.000
    4SOL     OW    6   0.500   0.000   0.000
   3.0 3.0 3.0
"""


def _ndx_groups(ndx_path):
    groups, name = {}, None
    for line in ndx_path.read_text().splitlines():
        if line.startswith("["):
            name = line.strip("[] ")
            groups[name] = []
        elif line.strip():
            groups[name] += [int(v) for v in line.split()]
    return groups


def test_index_uses_the_build_groups_over_residue_names(tmp_path):
    (tmp_path / "x_md.gro").write_text(GRO)
    # by residue name, the lipid (PA) and the adduct residue (CVX) are "ligand"
    ndx, _, _ = analysis._make_mmgbsa_index(tmp_path, "x", "x_md.gro")
    assert _ndx_groups(ndx)["Ligand"] == [3, 4, 5]
    (tmp_path / "x_build_groups.json").write_text(json.dumps(
        {"protein": [[1, 2], [4, 4]], "ligand": [[5, 5]], "ligand_resnames": []}))
    ndx, _, _ = analysis._make_mmgbsa_index(tmp_path, "x", "x_md.gro")
    groups = _ndx_groups(ndx)
    assert groups["Protein"] == [1, 2, 4]
    assert groups["Ligand"] == [5]  # the lipid (3) and water (6) are neither


def test_decomp_cutoff_uses_the_build_groups(tmp_path):
    gro = tmp_path / "x_md.gro"
    gro.write_text(GRO)
    groups = ({1, 2}, {5})
    # nearest protein atom (CA, 0.1 nm) is 3 Å from the ligand atom (0.4 nm)
    assert analysis._decomp_cutoff_angstrom(gro, groups=groups) == 5.0
    assert analysis._decomp_cutoff_angstrom(gro, groups=(set(), {5})) is None


def test_topology_residue_numbers_take_the_pdb_numbers():
    top = ab._renumber_atoms_sections(TOP, [[394], [1], [395]], [["MET"], ["SOL"], ["BNZ"]])
    rows = [l.split() for l in top.splitlines() if l.strip() and l.split()[0].isdigit() and len(l.split()) >= 8]
    assert [r[2] for r in rows if r[3] == "MET"] == ["394", "394", "394"]
    assert [r[2] for r in rows if r[3] == "BNZ"] == ["395", "395"]
    assert [r[2] for r in rows if r[3] == "SOL"] == ["1", "1", "1"]
    with pytest.raises(ab.AmberBuildError, match="out of step"):
        ab._renumber_atoms_sections(TOP, [[394], [1], [395]], [["ALA"], ["SOL"], ["BNZ"]])
