"""Configuration loaded from environment variables."""

import os

PORT = int(os.environ.get("PORT", "8024"))
API_KEY = os.environ.get("API_KEY", "")
MAX_CONCURRENT_SIMS = int(os.environ.get("MAX_CONCURRENT_SIMS", "3"))
NTOMP = int(os.environ.get("NTOMP", "8"))  # OpenMP threads per simulation

# S3 (results bucket). main.py reads MD_RESULTS_BUCKET directly; this
# constant is retained for any legacy consumer that still imports it.

# Redis
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

# Protein-ligand starting pose: dock the ligand into the protein with the
# autodock-gpu service and start MD from the top pose. Unset, the ligand is
# centred on the co-crystallized ligand's site when the PDB has one.
AUTODOCK_GPU_URL = os.environ.get("AUTODOCK_GPU_URL", "")
AUTODOCK_GPU_API_KEY = os.environ.get("AUTODOCK_GPU_API_KEY", "")

# A GPU `gmx mdrun` that fails is retried on the GPU MDRUN_GPU_RETRIES times
# (the GB10 fault is intermittent, and a CPU stage is far slower), then rerun
# once with nonbonded, PME and update on the CPU with MDRUN_CPU_NTOMP OpenMP
# threads. Set MDRUN_CPU_FALLBACK=0 to fail the job instead of the CPU rerun.
MDRUN_GPU_RETRIES = int(os.environ.get("MDRUN_GPU_RETRIES", "2"))
MDRUN_CPU_FALLBACK = os.environ.get("MDRUN_CPU_FALLBACK", "1") != "0"
MDRUN_CPU_NTOMP = int(os.environ.get("MDRUN_CPU_NTOMP", str(os.cpu_count() or NTOMP)))

# A running job's heartbeat is written every JOB_HEARTBEAT_SECONDS. A job
# whose heartbeat is older than JOB_STALE_SECONDS lost its worker (the
# service restarted or crashed) and /status reports it failed.
JOB_HEARTBEAT_SECONDS = int(os.environ.get("JOB_HEARTBEAT_SECONDS", "60"))
JOB_STALE_SECONDS = int(os.environ.get("JOB_STALE_SECONDS", "300"))
