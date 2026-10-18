# gromacs-md

GPU-accelerated molecular dynamics service for the [NovoMCP](https://github.com/NovoMCP/novomcp) engine. Runs [GROMACS](https://www.gromacs.org/) (compiled with CUDA) end-to-end from a SMILES string or PDB: topology + parameterization → solvation → minimization → NVT/NPT equilibration → production MD → trajectory analysis (RMSD, RMSF, radius of gyration).

Submit a job, get a `job_id` back immediately, and poll for progress — MD runs long, so the service is async.

## What it does

- **Protein + ligand, protein-only, or ligand-only** simulations
- **Ligand parameterization** — SMILES → 3D (RDKit) → GAFF2 topology (ACPYPE, AM1-BCC charges)
- **Force fields** — AMBER99SB-ILDN (protein) / TIP3P (water) / GAFF2 (ligand); membrane proteins in a POPC bilayer (ff14SB + Lipid21)
- **Structure intake classifier** (`intake/`) — routes a PDB to the right pipeline (soluble vs membrane) using RCSB / OPM / MetalPDB / Pfam signals
- **Metal-site parameterization** hooks (`/metal`) for MCPB.py-style bonded metal models
- **Durable jobs** — stage checkpoints so a preempted GPU replica resumes instead of restarting

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | — | GPU / Redis / storage / GROMACS version |
| `POST` | `/simulate` | API-Key | protein / ligand / complex simulation |
| `POST` | `/simulate/ligand` | API-Key | ligand-only simulation |
| `POST` | `/simulate/batch` | API-Key | multiple simulations |
| `GET` | `/status/{job_id}` | — | poll job status + progress |
| `GET` | `/results/{job_id}` | — | completed results + analysis |

Job IDs are `gro_{compound_id}_{timestamp}`; progress is tracked in Redis (`novomcp:job:{job_id}`), matching the engine's async job model.

## Run

This is a **GPU service** — GROMACS is compiled with CUDA. You need an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). See [`GPU.md`](./GPU.md) for driver/hardware details and options for getting a GPU.

**Build from source** — the recommended path; no dependency on a prebuilt image:

```bash
docker build -t gromacs-md .
docker run --gpus all -p 8024:8024 gromacs-md
curl -s localhost:8024/health
```

**Or pull the prebuilt image** (faster — skips the ~20 min build):

```bash
docker run --gpus all -p 8024:8024 ghcr.io/novomcp/gromacs-md:latest
```

The from-source build compiles GROMACS 2023.3 and installs AmberTools via micromamba (~15–20 min); subsequent builds reuse cached layers.

### A ligand-only run

```bash
curl -s -X POST localhost:8024/simulate/ligand \
  -H 'Content-Type: application/json' -H "API-Key: $API_KEY" \
  -d '{"smiles":"CC(=O)OC1=CC=CC=C1C(=O)O","duration_ns":1.0,"temperature":300.0}'
# → {"job_id":"gro_...","status":"queued"}

curl -s localhost:8024/status/gro_...      # poll
curl -s localhost:8024/results/gro_...     # when complete
```

## Wire it to the engine

```bash
export GROMACS_MD_URL=http://localhost:8024
export GROMACS_API_KEY=$API_KEY            # if you set API_KEY on the service
```

`run_molecular_dynamics` and `generate_dynamics` then light up in the engine. Because MD is long-running, the engine returns a `job_id` and smart-polls for you (or set `email` for a completion notice). See the engine's [deploying-services guide](https://github.com/NovoMCP/novomcp/tree/main/docs/deploying-services/gromacs-md.md) and [verifying-services](https://github.com/NovoMCP/novomcp/blob/main/docs/deploying-services/verifying-services.md).

## Batch / queue execution

`run_md_job.py` is a CLI executor for running one job to completion outside the HTTP server — either directly (`MD_JOB_ID` + `MD_CONFIG` env) or by consuming the `novomcp:gromacs:job_queue` dispatch queue. This is how the engine runs MD as a Kubernetes Job (see [`k8s/job-template.yaml`](./k8s/job-template.yaml)) so long simulations survive pod restarts via checkpoint resume.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8024` | HTTP listen port |
| `API_KEY` | (unset) | require an `API-Key` header on POST endpoints; unset = no inbound auth |
| `MAX_CONCURRENT_SIMS` | `3` | GPU semaphore — concurrent simulations |
| `NTOMP` | `8` | OpenMP threads per simulation |
| `REDIS_URL` | `redis://localhost:6379` | job tracking (falls back to in-memory if unavailable) |
| `AUTODOCK_GPU_URL` | (unset) | [autodock-gpu](https://github.com/NovoMCP/autodock-gpu) endpoint: protein-ligand runs start from the docked pose (see below) |
| `AUTODOCK_GPU_API_KEY` | (unset) | `X-API-Key` for autodock-gpu, if it requires one |
| `MDRUN_GPU_RETRIES` | `2` | retries of a failed GPU `mdrun` stage on the GPU |
| `MDRUN_CPU_FALLBACK` | `1` | after those, rerun the stage once on the CPU; `0` fails the job instead |
| `MDRUN_CPU_NTOMP` | CPU count | OpenMP threads for that CPU rerun |
| `MD_RESULTS_BUCKET` | (unset) | S3 bucket for results + checkpoints; results stay local if unset |
| `AWS_REGION` | `us-east-1` | region for the results bucket |

Redis and S3 are both optional — with neither, the service still runs and keeps results on the local filesystem.

### Protein-ligand starting pose

The ligand from `ligand_smiles` has to start in the binding site, not wherever its SMILES conformer lands. With `AUTODOCK_GPU_URL` set, the service docks the ligand into the protein first, with the box centred on the co-crystallized ligand when the PDB has one, otherwise on autodock-gpu's detected pocket. The MD starts from the top pose. Without docking it centres the ligand on the co-crystallized ligand's site. With neither, the result is flagged as unplaced. The job result's `ligand_placement` records which of these happened (`docked`, `native_site_centroid` or `unplaced`).

### GPU failures

If a GPU `mdrun` stage fails, it is retried on the GPU (`MDRUN_GPU_RETRIES`, default 2). If it still fails, it is rerun once with everything on the CPU. A production run that has written a checkpoint continues from that checkpoint. The job result's `mdrun_retries` lists each retried stage and how it recovered. The CPU rerun is a last resort: 100 ps of NVT on a 60,000-atom system takes about 10 s on the GB10 GPU and about 30 min on its CPU. This guards against an intermittent fault in GROMACS's GPU nonbonded kernel on GB10 (DGX Spark): with larger protein systems, the same input can fail on the GPU with LINCS warnings and then a segfault or CUDA error 700, but runs clean on the CPU.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

`scripts/test_intake.py` is a live integration check for the intake classifier against the public RCSB / OPM / MetalPDB APIs.

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
