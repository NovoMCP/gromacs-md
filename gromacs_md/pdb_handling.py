"""PDB fetching, cleaning, repair, and SMILES conversion."""

import logging
import subprocess
import tempfile
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True,
)
async def _rcsb_fetch(client: httpx.AsyncClient, url: str):
    """HTTP GET with automatic retry on transient network errors."""
    return await client.get(url)


async def fetch_pdb_from_rcsb(pdb_id: str) -> str:
    """Fetch PDB structure from RCSB. Falls back to mmCIF for newer structures."""
    pdb_id = pdb_id.upper()
    async with httpx.AsyncClient(timeout=30) as client:
        response = await _rcsb_fetch(client, f"https://files.rcsb.org/download/{pdb_id}.pdb")
        if response.status_code == 200:
            logger.info(f"Fetched PDB {pdb_id} from RCSB ({len(response.text)} bytes)")
            return response.text

        logger.info(f"PDB format not available for {pdb_id}, trying mmCIF...")
        cif_response = await _rcsb_fetch(client, f"https://files.rcsb.org/download/{pdb_id}.cif")
        if cif_response.status_code != 200:
            raise ValueError(f"PDB {pdb_id} not found on RCSB (HTTP {cif_response.status_code})")

        with tempfile.TemporaryDirectory() as tmp:
            cif_path = Path(tmp) / f"{pdb_id}.cif"
            pdb_path = Path(tmp) / f"{pdb_id}.pdb"
            cif_path.write_text(cif_response.text)
            result = subprocess.run(
                ["obabel", str(cif_path), "-O", str(pdb_path)],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0 or not pdb_path.exists():
                raise ValueError(f"Failed to convert {pdb_id} mmCIF to PDB: {result.stderr.decode(errors='replace')}")
            logger.info(f"Converted {pdb_id} from mmCIF to PDB format")
            return pdb_path.read_text()


def clean_pdb_protein_only(pdb_content: str) -> str:
    """Clean PDB to protein-only ATOM records.

    Strips HETATM (ligands, ions, cofactors, waters), alternate conformations,
    and non-standard residues.
    """
    STANDARD_AA = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
        'TYR', 'VAL'
    }
    clean_lines = []
    for line in pdb_content.splitlines(keepends=True):
        if line.startswith("ATOM"):
            res_name = line[17:20].strip()
            altloc = line[16:17]
            if res_name in STANDARD_AA and (altloc == ' ' or altloc == 'A'):
                if altloc == 'A':
                    line = line[:16] + ' ' + line[17:]
                clean_lines.append(line)
        elif line.startswith("TER") or line.startswith("END"):
            clean_lines.append(line)
    atom_count = sum(1 for l in clean_lines if l.startswith("ATOM"))
    if atom_count == 0:
        raise ValueError("PDB cleaning removed all atoms -- no standard amino acids found")
    logger.info(f"Cleaned PDB: kept {atom_count} protein ATOM records")
    return "".join(clean_lines)


def repair_missing_atoms(pdb_content: str) -> tuple[str, dict]:
    """Rebuild missing sidechain heavy atoms on standard residues via PDBFixer."""
    import io as _io
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as e:
        logger.warning(f"PDBFixer not available ({e}); skipping atom repair")
        return pdb_content, {"error": "pdbfixer_not_installed"}

    try:
        fixer = PDBFixer(pdbfile=_io.StringIO(pdb_content))
        fixer.findMissingResidues()
        chains = list(fixer.topology.chains())
        keys_to_keep = {}
        for key in list(fixer.missingResidues.keys()):
            chain_idx, ins_idx = key
            chain = chains[chain_idx]
            n_residues = len(list(chain.residues()))
            if ins_idx == 0 or ins_idx == n_residues:
                continue
            keys_to_keep[key] = fixer.missingResidues[key]
        fixer.missingResidues = keys_to_keep

        fixer.findNonstandardResidues()
        fixer.findMissingAtoms()
        n_missing_atom_residues = len(fixer.missingAtoms)
        n_missing_atoms_total = sum(len(v) for v in fixer.missingAtoms.values())
        n_added_chains = sum(len(v) for v in fixer.missingResidues.values())

        fixer.addMissingAtoms()

        buf = _io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, buf, keepIds=True)
        repaired = buf.getvalue()

        stats = {
            "added_atoms": n_missing_atoms_total,
            "added_residues_in_existing_chains": n_added_chains,
            "residues_with_added_atoms": n_missing_atom_residues,
        }
        return repaired, stats
    except Exception as e:
        logger.warning(f"PDBFixer repair failed: {e}; falling back to original PDB")
        return pdb_content, {"error": str(e)}


def smiles_to_pdb(smiles: str, output_pdb: Path):
    """Convert SMILES to 3D PDB structure using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToPDBFile(mol, str(output_pdb))
    logger.info("Generated 3D structure from SMILES")


def split_protein_ligand(pdb_path: Path, protein_pdb: Path, ligand_pdb: Path):
    """Split PDB into protein and ligand files."""
    protein_lines = []
    ligand_lines = []

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                protein_lines.append(line)
            elif line.startswith("HETATM"):
                res_name = line[17:20].strip()
                if res_name not in ["HOH", "WAT", "SOL"]:
                    ligand_lines.append(line)

    with open(protein_pdb, "w") as f:
        f.writelines(protein_lines)
        f.write("END\n")

    if ligand_lines:
        with open(ligand_pdb, "w") as f:
            f.writelines(ligand_lines)
            f.write("END\n")

    logger.info(f"Split PDB: {len(protein_lines)} protein atoms, {len(ligand_lines)} ligand atoms")
