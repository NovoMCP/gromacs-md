# gromacs-md — architecture

GPU molecular dynamics for the NovoMCP engine. GROMACS (compiled with CUDA) wrapped
in an async HTTP service, with ligand parameterization, a structure-intake classifier,
and durable (checkpoint-resumable) long jobs.

**Port:** 8024 · **Contacted by:** the NovoMCP engine (`GROMACS_MD_URL`)

---

## Simulation pipeline

Every run is a staged pipeline, GPU-accelerated at each `mdrun`:

```
input (PDB and/or SMILES)
  → system prep     protein: pdb2gmx (AMBER99SB-ILDN, TIP3P)
                    ligand:  SMILES → 3D (RDKit) → MOL2 (OpenBabel) → GAFF2 (ACPYPE, AM1-BCC)
                    complex: merge topologies
  → box + solvate   editconf (1.0 nm padding) + solvate
  → minimize        steepest descent
  → NVT             V-rescale thermostat → target T
  → NPT             Parrinello-Rahman barostat → target P
  → production MD   configurable ns, GPU mdrun
  → analysis        RMSD (gmx rms), RMSF (gmx rmsf), Rg (gmx gyrate)
```

Simulation types: **protein + ligand complex**, **protein-only**, **ligand-only**.

## Async job model

`POST /simulate*` returns a `job_id` immediately and runs the pipeline in the
background under a GPU semaphore (`MAX_CONCURRENT_SIMS`). Progress is written to
Redis in the engine-compatible 3-key format:

```
novomcp:job:{job_id}          # hash: status, progress, result, last_updated
novomcp:job_result:{job_id}   # string: JSON result
novomcp:cache:jobs:{job_id}   # string: JSON full job data
```

Job IDs are `gro_{compound_id}_{timestamp}`. Clients poll `GET /status/{job_id}`
until `complete`, then read `GET /results/{job_id}`. If Redis is absent, tracking
falls back to in-memory (single-process).

## Durable jobs (checkpoint resume)

For long production runs, `run_md_job.py` runs a single job to completion as a
batch/Kubernetes Job. A stage-based manifest (`gromacs_md/manifest.py`) plus
periodic `md.cpt` uploads to object storage let a preempted or restarted pod
resume from the last checkpoint (`gmx mdrun -cpi md.cpt -append`) instead of
starting over. The engine dispatches jobs by pushing onto `novomcp:gromacs:job_queue`.

## Intake classifier (`intake/`)

Given a PDB, classifies the system and routes it to the correct pipeline (soluble
vs membrane) and flags special handling (metals, PTMs) using signals from RCSB,
OPM (membrane), MetalPDB (metal sites), and Pfam (functional family). Returns a
structured `RoutingDecision`. `scripts/test_intake.py` exercises it against the
live public APIs.

## Force-field details

- Protein: AMBER99SB-ILDN · Water: TIP3P · Ligand: GAFF2 (AM1-BCC charges)
- Membrane systems: CHARMM36m (bundled at build time) + semi-isotropic pressure
  coupling (see `mdp_templates/membrane/`)
- Thermostat V-rescale · Barostat Parrinello-Rahman · PME electrostatics
- LINCS on h-bonds (2 fs timestep), Verlet cutoff scheme

## Layout

```
main.py                  FastAPI app, startup (Redis/S3/GPU/GROMACS checks)
run_md_job.py            CLI/batch executor with checkpoint resume
gromacs_md/
  routes/                simulate, status, health, metal, audit
  simulation.py          the staged pipeline
  topology.py            pdb2gmx + ligand topology assembly
  analysis.py            RMSD / RMSF / Rg (+ adequacy heuristics)
  manifest.py            durable-job checkpoint schema
  s3_storage.py          results upload
  s3_checkpoint.py       checkpoint upload/restore
  redis_jobs.py          engine-compatible job tracking
  config.py              env-driven config
intake/                  structure classifier (RCSB / OPM / MetalPDB / Pfam)
mdp_templates/           GROMACS .mdp templates (membrane)
k8s/                     reference Deployment + Job template
```
