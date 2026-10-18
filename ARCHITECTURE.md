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
  → ions            genion: neutralize + 0.15 M NaCl
  → minimize        steepest descent
  → NVT             V-rescale thermostat → target T, velocities generated at T,
                    solute heavy atoms restrained (-DPOSRES)
  → NPT             C-rescale barostat → target P, solute still restrained
  → production MD   configurable ns, GPU mdrun, unrestrained, Parrinello-Rahman
  → analysis        RMSD (gmx rms), RMSF (gmx rmsf), Rg (gmx gyrate)
```

Simulation types: **protein + ligand complex**, **protein-only**, **ligand-only**.

### Topology builders

System prep has two sources, chosen per request (`builder`) or by
`MD_TOPOLOGY_BUILDER` (default `pdb2gmx`):

- **pdb2gmx**: the steps above, AMBER99SB-ILDN.
- **tleap** (`amber_build.py`): AMBER ff14SB + GAFF2 + TIP3P. Histidine
  states at pH 7 come from OpenMM, bonded cysteines become CYX, and chain
  breaks are capped instead of bonded across. tleap solvates (10 Å, cubic) and
  adds the same ions as genion, and parmed converts the result to GROMACS with
  heavy-atom `POSRES` blocks per solute molecule. Before any dynamics, a
  single-point energy from `sander` (AMBER files) and `gmx` (converted files)
  is compared term by term; a conversion error fails the job instead of
  silently changing the force field. The build also writes which atoms are
  protein and which are ligand (`<cid>_build_groups.json`); analysis uses those
  in place of residue-name rules, which can't classify lipids or a ligand
  bonded into a protein residue.

### Covalent ligands (`covalent_ligand.py`, `covalent_build.py`)

A structure whose ligand is bonded to a cysteine SG (`LINK`) routes to
`run_covalent`, and the crystal's ligand is simulated bonded:

1. The ligand's bonds and charges come from its RCSB Chemical Component
   Dictionary entry, placed on the crystal pose. The entry isn't always the
   bonded form, so the bond is formed by warhead rule: Michael acceptors
   (acrylamides) are saturated, nitriles become thioimidates, carbonyls
   hemithioacetals, a halide leaving group goes, and otherwise the hydrogen the
   entry drew in place of the sulfur is replaced.
2. The cysteine and the ligand become one residue, CVX. AM1-BCC charges come
   from the capped fragment ACE-CYS(adduct)-NME. The backbone keeps ff14SB
   types and charges, and the rest is GAFF2, with charges shifted so the residue
   carries the ligand's formal charge. tleap lists the terms between the two
   that neither force field has (CA-CB, N-CA-CB, ...), and each is taken from
   GAFF2 with the backbone types read as their GAFF2 equivalents.
3. The tleap build, energy-parity check and pipeline then run as above. If the
   start structure clashes too badly for sander to print an energy term
   (atoms rebuilt by PDBFixer), the comparison is made after a short
   minimization.

Analysis reports the S-C bond length over the trajectory and skips MM-GBSA (a
binding free energy doesn't apply to a bonded ligand). Refused: several
covalent ligands, ligands built from several HET residues (6LU7), bonds to
residues other than cysteine, and boron warheads. `COVALENT_BRANCH_ENABLED=0`
refuses every covalent ligand.

tleap is the path for systems pdb2gmx can't build (lipids, covalent adducts).
Results from the two builders use different protein force fields and aren't
strictly comparable; each result states `force_field` and `builder`.

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

- Protein: AMBER99SB-ILDN (pdb2gmx) or ff14SB (tleap) · Water: TIP3P · Ligand: GAFF2 (AM1-BCC charges)
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
  amber_build.py         tleap builder, parmed conversion, energy-parity check
  covalent_ligand.py     bonded form of a covalent ligand (CCD + crystal pose)
  covalent_build.py      CVX adduct residue: charges, types, junction terms
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
