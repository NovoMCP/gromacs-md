"""Tests for reading gmx_MMPBSA output and launching it."""

import subprocess
import sys
from pathlib import Path

from gromacs_md import analysis

# Real 181L + benzene output (2 frames): Complex, Receptor, Ligand and Delta sections
FIXTURES = Path(__file__).parent / "fixtures"


def test_delta_g_comes_from_the_delta_section_only():
    # the other sections are absolute energies (~-5180 kcal/mol for the complex)
    assert analysis._parse_mmgbsa_delta(FIXTURES / "mmgbsa_results.csv") == [[1, -14.43], [11, -12.67]]


def test_decomposition_is_the_per_residue_mean_of_the_deltas():
    contribs = analysis._parse_mmgbsa_decomp(FIXTURES / "mmgbsa_decomp.csv")
    assert len(contribs) == 15  # receptor residues only; the ligand's own term is dropped
    assert contribs["LEU:84"] == (-1.42 + -0.69) / 2
    assert contribs["ALA:99"] == (-0.22 + -1.47) / 2


def _fake_gmx_mmpbsa(tmp_path):
    """A stand-in GMXMMPBSA package with 1.5.0.3's unclamped get_corrstd."""
    pkg = tmp_path / "GMXMMPBSA"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "utils.py").write_text(
        "from math import sqrt\n"
        "def get_corrstd(val1, val2):\n"
        "    return sqrt(val1 ** 2 + val2 ** 2 - 2 * val1 * val2)\n"
        "def corr_sub(a, b):\n"
        "    return get_corrstd(a, b)\n"
    )
    (pkg / "app.py").write_text(
        "import sys\n"
        "from . import utils\n"
        "def gmxmmpbsa():\n"
        "    print(sys.argv[0], sys.argv[1:], utils.corr_sub(1.859062658947177, 1.8590626589471775))\n"
        "    return 0\n"
    )
    return tmp_path


def test_launcher_clamps_the_correlated_std(tmp_path):
    root = _fake_gmx_mmpbsa(tmp_path)
    stock = subprocess.run([sys.executable, "-c", "from GMXMMPBSA.utils import corr_sub;"
                            "corr_sub(1.859062658947177, 1.8590626589471775)"],
                           cwd=root, capture_output=True, text=True)
    assert "math domain error" in stock.stderr
    run = subprocess.run([sys.executable, "-c", analysis._GMX_MMPBSA_LAUNCHER, "-O", "-nogui"],
                         cwd=root, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["gmx_MMPBSA", "['-O',", "'-nogui']", "0.0"]


def test_launcher_uses_the_gmx_mmpbsa_interpreter(tmp_path, monkeypatch):
    exe = tmp_path / "gmx_MMPBSA"
    exe.write_text("#!/opt/env/bin/python3.12\nfrom GMXMMPBSA.app import gmxmmpbsa\n")
    monkeypatch.setattr(analysis.shutil, "which", lambda name: str(exe))
    assert analysis._gmx_mmpbsa_cmd() == ["/opt/env/bin/python3.12", "-c", analysis._GMX_MMPBSA_LAUNCHER]


def test_decomposition_keeps_the_chain_when_the_complex_has_chain_ids(tmp_path):
    csv = tmp_path / "dec.csv"
    csv.write_text(
        "DELTAS:\nTotal Decomposition Contribution (TDC)\n"
        "Frame #,Residue,Internal,van der Waals,Electrostatic,Polar Solvation,Non-Polar Solv.,TOTAL\n"
        "1,R:A:ASN:23,0.0,-0.09,-3.31,0.65,-0.01,-2.76\n"
        "1,R:B:SER:45,0.0,-0.5,-1.0,0.5,0.0,-1.0\n"
        "1,L:C:UNL:160,0.0,-7.0,-0.5,3.4,-1.8,-5.9\n"
    )
    assert analysis._parse_mmgbsa_decomp(csv) == {"A:ASN:23": -2.76, "B:SER:45": -1.0}
