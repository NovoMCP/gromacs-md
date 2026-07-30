"""System diagnostics — GPU and GROMACS version checks."""

import subprocess


def check_gpu_available() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def get_gromacs_version() -> str:
    try:
        result = subprocess.run(["gmx", "--version"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "GROMACS version" in line:
                return line.strip()
        return "Unknown"
    except Exception:
        return "Not installed"
