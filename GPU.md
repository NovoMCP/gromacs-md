# Running gromacs-md on a GPU

gromacs-md compiles **GROMACS with CUDA** and runs production MD on the GPU. Unlike
the CPU-optional services in the engine, this one is built for a GPU host — the
`mdrun` steps use `-nb gpu`.

## What you need

| Requirement | Detail |
|---|---|
| GPU | Any NVIDIA CUDA GPU. L4 / A10G are fine for small systems and short runs; L40S / A100 / H100 for larger systems and long production. ~8 GB GPU memory per concurrent simulation. |
| Driver | CUDA 11.8-compatible driver (the image is built `FROM nvidia/cuda:11.8.0-devel`). Check with `nvidia-smi`. |
| Container runtime | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so `docker run --gpus all` exposes the GPU. |

## Getting a GPU

You don't need a cluster — any single CUDA GPU host works:

- **Cloud GPU VM.** An image that already ships the NVIDIA driver + CUDA is quickest,
  e.g. AWS "Deep Learning OSS Nvidia Driver AMI" on a `g5`/`g6` instance, GCP/Azure GPU
  images, Lambda Cloud, RunPod. Confirm `nvidia-smi` works.
- **Your own workstation** with an NVIDIA GPU + the NVIDIA Container Toolkit.

## Run

The quickest path is to **pull the prebuilt image** — no build required:

```bash
# On the GPU host:
docker run --gpus all -p 8024:8024 ghcr.io/novomcp/gromacs-md:latest
```

**Or build from source** (to customize the GROMACS / force-field setup):

```bash
docker build -t gromacs-md .
docker run --gpus all -p 8024:8024 gromacs-md
```

The from-source build compiles GROMACS 2023.3 and installs AmberTools (via
micromamba) for ligand parameterization — **~15–20 min** the first time. Layer
caching makes code-only rebuilds fast.

Either way, first boot downloads/warms the force fields — allow a moment before
the first request; `/health` reports readiness.

## Verify the GPU is seen

```bash
curl -s localhost:8024/health
# → gpu_available: true, plus the detected GROMACS version
```

Then submit a short ligand-only run and poll it:

```bash
JOB=$(curl -s -X POST localhost:8024/simulate/ligand \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CC(=O)OC1=CC=CC=C1C(=O)O","duration_ns":0.1,"temperature":300.0}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')

curl -s localhost:8024/status/$JOB       # queued → preparation → ... → complete
curl -s localhost:8024/results/$JOB      # RMSD / RMSF / Rg + output files
```

A 0.1 ns aspirin run is a good smoke test — it exercises the full pipeline
(parameterize → solvate → minimize → equilibrate → produce → analyze) in a
minute or two on a modern GPU.

## Notes

- **`gpu_available: false`?** The container can't see a GPU — check
  `docker run --gpus all gromacs-md nvidia-smi` and that the NVIDIA Container
  Toolkit is installed on the host.
- **`MAX_CONCURRENT_SIMS`** gates how many simulations share the GPU at once
  (default 3). Lower it if you hit GPU-memory limits on large systems.
- **`NTOMP`** sets OpenMP threads per simulation (default 8).
- The image is built for the AVX2 SIMD level (`-DGMX_SIMD=AVX2_256`) so it runs
  on both AMD and Intel hosts; rebuild with a higher `GMX_SIMD` for a small
  speedup on CPUs that support it.
