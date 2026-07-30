"""Pydantic request models for the API."""

from typing import Optional, Dict, List
from pydantic import BaseModel


class MDSimulationRequest(BaseModel):
    compound_id: Optional[str] = None
    pdb_id: Optional[str] = None
    pdb_content: Optional[str] = None
    ligand_smiles: Optional[str] = None
    simulation_ns: float = 10.0
    temperature: float = 300.0
    pressure: float = 1.0


class LigandPreparationRequest(BaseModel):
    smiles: str
    duration_ns: float = 10.0
    temperature: float = 300.0


class BatchMDRequest(BaseModel):
    compounds: List[Dict]
    simulation_ns: float = 100.0


class AuditRequest(BaseModel):
    """System audit request -- classify a structure without running MD."""
    pdb_id: Optional[str] = None
    pdb_content: Optional[str] = None
