"""
Manifest schema for gromacs-md durable jobs.

A checkpoint manifest with stage-based resume fields. The manifest is the
single artifact that survives container death and drives resume-after-
restart: when the GPU replica is preempted, the retry container reads
this from object storage and reconstructs both the job config
(input_params) and the resume point (current_stage / completed_stages /
production_progress).

Usage:
    m = Manifest(
        job_id="gro_...",
        pipeline="soluble",
        current_stage="em",
        input_params=InputParams(pdb_id="1HSG", compound_id="ABC", duration_ns=10.0),
    )
    upload_manifest(blob_client, m.job_id, m.model_dump())
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Pipeline = Literal["soluble", "membrane"]

# Soluble pipeline stage order. Membrane stages diverge after prep; the
# manifest stores stage names as bare strings so adding membrane stages
# later does not require a schema change.
#
# Stages are the resume granularity. system_prep consolidates everything
# before EM (input prep + pdb2gmx + ligand topology + combine + editconf +
# solvate) — that block is <2% of total runtime and the intermediate files
# are tightly coupled, so a single resume point covers it.
SOLUBLE_STAGES = (
    "system_prep",
    "em",
    "nvt",
    "npt",
    "production",
    "analysis",
)


class ProductionProgress(BaseModel):
    """Sub-stage progress within the production stage.

    Updated by the 5-min md.cpt upload loop while gmx mdrun is active.
    last_cpt_step is what `gmx mdrun -cpi md.cpt -append` continues from
    on resume. Empty (all zeros) until production starts. MD's production
    stage is the dominant cost and needs sub-stage checkpointing.
    """

    last_cpt_step: int = 0
    last_cpt_time_ps: float = 0.0
    target_steps: int = 0
    target_time_ps: float = 0.0


class InputParams(BaseModel):
    """Recipe to reconstruct the job config on retry.

    The input_params block. When the first
    attempt, the Redis dispatch queue message is already consumed via
    BLPOP. The retry container reads this from the manifest to rebuild
    the job config without help from the queue.
    """

    pdb_id: str
    compound_id: str
    ligand_smiles: Optional[str] = None
    duration_ns: float
    temperature: float = 300.0
    force_field: str = "amber99sb-ildn"


class Manifest(BaseModel):
    """Resume-state manifest for one gromacs-md job.

    Uploaded to Blob after every stage completes, plus every 5 min
    during production. Always written AFTER the corresponding stage
    artifacts are uploaded so the manifest never points at state that
    is not yet durable.

    Fields:
      job_id, pipeline, input_params  — immutable for the job's lifetime
      current_stage                   — name of the stage in progress
      completed_stages                — append-only list, source of truth
                                        for resume (the DONE marker on
                                        disk is the per-stage equivalent
                                        but does not survive container
                                        death)
      production_progress             — mutated during the production
                                        upload loop; ignored for pre-
                                        production stages
    """

    job_id: str
    pipeline: Pipeline
    current_stage: str
    completed_stages: List[str] = Field(default_factory=list)
    production_progress: ProductionProgress = Field(default_factory=ProductionProgress)
    input_params: InputParams
