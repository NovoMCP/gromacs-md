"""Tests for the restrained NVT/NPT equilibration."""

from gromacs_md import topology


def _mdp(path):
    opts = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split(";")[0].split("=", 1)
            opts[k.strip().replace("-", "_")] = v.strip()
    return opts


def test_nvt_restrains_the_solute_and_starts_at_temperature(tmp_path):
    topology.write_nvt_mdp(tmp_path, "x", 310.0)
    nvt = _mdp(tmp_path / "x_nvt.mdp")
    assert nvt["define"] == "-DPOSRES"
    assert nvt["gen_vel"] == "yes" and nvt["gen_temp"] == "310.0"


def test_npt_restrains_and_continues_from_nvt(tmp_path):
    topology.write_npt_mdp(tmp_path, "x", 300.0, 1.0)
    npt = _mdp(tmp_path / "x_npt.mdp")
    assert npt["define"] == "-DPOSRES"
    assert npt["refcoord_scaling"] == "com"
    assert npt["continuation"] == "yes" and "gen_vel" not in npt
    assert npt["pcoupl"] == "C-rescale"


def test_production_is_unrestrained(tmp_path):
    topology.write_md_mdp(tmp_path, "x", 1.0, 300.0, 1.0)
    assert "define" not in _mdp(tmp_path / "x_md.mdp")


# ACPYPE-style itp: benzene-like fragment, C/H atoms
LIGAND_ITP = """\
[ atomtypes ]
;name   bond_type     mass     charge   ptype   sigma         epsilon       Amb
 ca       ca          0.00000  0.00000   A     3.31521e-01   4.13379e-01 ; 1.86  0.0988
 ha       ha          0.00000  0.00000   A     2.62548e-01   6.73624e-02 ; 1.47  0.0161

[ moleculetype ]
;name            nrexcl
 x                3

[ atoms ]
;   nr  type  resi  res  atom  cgnr     charge      mass
     1   ca     1   UNL    C1    1    -0.130000     12.01000 ; qtot -0.130
     2   ha     1   UNL    H1    2     0.130000      1.00800 ; qtot 0.000
     3   ca     1   UNL    C2    3    -0.130000     12.01000 ; qtot -0.130
     4   ha     1   UNL    H2    4     0.130000      1.00800 ; qtot 0.000

[ bonds ]
;   ai     aj funct   r             k
     1      2   1    1.0870e-01    2.8811e+05 ;     C1 - H1
"""


def _complex(tmp_path):
    (tmp_path / "x_protein.gro").write_text("p\n    1\n    1ALA     CA    1   0.000   0.000   0.000\n   1.0 1.0 1.0\n")
    (tmp_path / "x_ligand.gro").write_text("l\n    1\n    1UNL     C1    1   0.100   0.100   0.100\n   1.0 1.0 1.0\n")
    (tmp_path / "x_protein.top").write_text(
        '#include "amber99sb-ildn.ff/forcefield.itp"\n[ moleculetype ]\nProtein 3\n'
        '#ifdef POSRES\n#include "posre.itp"\n#endif\n'
        "[ system ]\nx\n\n[ molecules ]\nProtein_chain_A     1\n"
    )
    itp = tmp_path / "x_ligand.itp"
    itp.write_text(LIGAND_ITP)
    topology.combine_protein_ligand(tmp_path, "x", tmp_path / "x_protein.gro", tmp_path / "x_ligand.gro",
                                    tmp_path / "x_protein.top", itp)
    return itp


def test_complex_ligand_gets_heavy_atom_restraints(tmp_path):
    itp = _complex(tmp_path)
    posre = (tmp_path / "x_ligand_posre.itp").read_text()
    rows = [l.split() for l in posre.splitlines() if l.strip() and not l.startswith((";", "["))]
    assert [int(r[0]) for r in rows] == [1, 3]
    assert all(r[1:] == ["1", "1000", "1000", "1000"] for r in rows)
    # inside the ligand's moleculetype, only under -DPOSRES
    tail = itp.read_text().rstrip().splitlines()[-3:]
    assert tail == ["#ifdef POSRES", '#include "x_ligand_posre.itp"', "#endif"]
    assert "[ atomtypes ]" not in itp.read_text()


def test_posre_falls_back_to_atom_names_without_masses(tmp_path):
    lines = ["[ atoms ]\n", " 1  ca  1  UNL  C1  1  -0.1\n", " 2  ha  1  UNL  H1  2  0.1\n"]
    out = topology.write_ligand_posre(lines, tmp_path / "p.itp")
    rows = [l for l in out.read_text().splitlines() if l.strip() and not l.startswith((";", "["))]
    assert len(rows) == 1 and rows[0].split()[0] == "1"
