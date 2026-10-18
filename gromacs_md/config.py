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
# Unset, the CPU rerun takes the cores the other running simulations aren't
# using (NTOMP each), and at least NTOMP: all 19 on an idle GB10.
MDRUN_CPU_NTOMP = int(os.environ["MDRUN_CPU_NTOMP"]) if os.environ.get("MDRUN_CPU_NTOMP") else None

# OpenMP threads for ACPYPE's sqm (AM1-BCC charges). sqm doesn't speed up with
# threads (imatinib on the GB10: 87 s on 1, 84 s on 4, 92 s on 19), and with
# all cores it starved the mdrun of concurrent jobs.
ACPYPE_OMP_THREADS = int(os.environ.get("ACPYPE_OMP_THREADS", "1"))

# Runtime estimate before production: minutes per simulated ns. Once a
# production run has been measured on this host, the last measured rate is
# used instead. The default is calibrated on an A100.
MD_MIN_PER_NS = float(os.environ.get("MD_MIN_PER_NS", "3"))

# A running job's heartbeat is written every JOB_HEARTBEAT_SECONDS. A job
# whose heartbeat is older than JOB_STALE_SECONDS lost its worker (the
# service restarted or crashed) and /status reports it failed.
JOB_HEARTBEAT_SECONDS = int(os.environ.get("JOB_HEARTBEAT_SECONDS", "60"))
JOB_STALE_SECONDS = int(os.environ.get("JOB_STALE_SECONDS", "300"))

# Points kept per time series in the job result. The full series go to
# <compound>_analysis_full.json; a 10 ns run otherwise returns ~590 KB, which
# MCP clients truncate before reaching the summary sections.
RESULT_SERIES_MAX_POINTS = int(os.environ.get("RESULT_SERIES_MAX_POINTS", "200"))
