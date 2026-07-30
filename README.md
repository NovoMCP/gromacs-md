# gromacs-md

GPU-accelerated molecular dynamics service for the [NovoMCP](https://github.com/NovoMCP/novomcp) engine. Runs [GROMACS](https://www.gromacs.org/) (compiled with CUDA) end-to-end from a SMILES string or PDB: topology + parameterization → solvation → minimization → NVT/NPT equilibration → production MD → trajectory analysis (RMSD, RMSF, radius of gyration).

Submit a job, get a `job_id` back immediately, and poll for progress — MD runs long, so the service is async.

## What it does

- **Protein + ligand, protein-only, or ligand-only** simulations
- **Ligand parameterization** — SMILES → 3D (RDKit) → GAFF2 topology (ACPYPE, AM1-BCC charges)
- **Force fields** — AMBER99SB-ILDN (protein) / TIP3P (water) / GAFF2 (ligand); CHARMM36m available for membrane systems
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

**Pull the prebuilt image** (fastest — skips the ~20 min build):

```bash
docker run --gpus all -p 8024:8024 ghcr.io/novomcp/gromacs-md:latest
curl -s localhost:8024/health
```

**Or build from source** (e.g. to customize the GROMACS/force-field setup):

```bash
docker build -t gromacs-md .
docker run --gpus all -p 8024:8024 gromacs-md
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
| `MD_RESULTS_BUCKET` | (unset) | S3 bucket for results + checkpoints; results stay local if unset |
| `AWS_REGION` | `us-east-1` | region for the results bucket |

Redis and S3 are both optional — with neither, the service still runs and keeps results on the local filesystem.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

`scripts/test_intake.py` is a live integration check for the intake classifier against the public RCSB / OPM / MetalPDB APIs.

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
