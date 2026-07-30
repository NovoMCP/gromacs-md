# GROMACS-MD — GPU molecular dynamics service
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    libfftw3-dev \
    libopenmpi-dev \
    openmpi-bin \
    libboost-all-dev \
    gfortran \
    libnetcdf-dev \
    libnetcdff-dev \
    libxrender1 \
    libgomp1 \
    libxext6 \
    liblapack3 \
    libblas3 \
    curl \
    openbabel \
    && rm -rf /var/lib/apt/lists/*

# Install AmberTools via micromamba (standalone, no base env conflicts)
# Provides antechamber, sqm, parmchk2, tleap needed by ACPYPE for GAFF2 parameterization
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
RUN for i in 1 2 3 4 5; do \
        curl -fsSL -o /tmp/micromamba.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest && \
        tar -xvjf /tmp/micromamba.tar.bz2 -C /usr/local bin/micromamba && \
        rm /tmp/micromamba.tar.bz2 && break; \
        echo "Attempt $i failed, retrying in 10s..."; sleep 10; \
    done && \
    CONDA_OVERRIDE_CUDA=11.8 /usr/local/bin/micromamba create -n amber -c conda-forge python=3.12 ambertools=23 -y && \
    /usr/local/bin/micromamba clean -afy

# Install GROMACS 2023.3 with GPU support
WORKDIR /opt
RUN wget https://ftp.gromacs.org/gromacs/gromacs-2023.3.tar.gz && \
    tar xfz gromacs-2023.3.tar.gz && \
    cd gromacs-2023.3 && \
    mkdir build && \
    cd build && \
    cmake .. \
        -DGMX_BUILD_OWN_FFTW=ON \
        -DGMX_GPU=CUDA \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DGMX_SIMD=AVX2_256 \
        -DGMX_MPI=OFF \
        -DGMX_OPENMP=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /opt/gromacs-2023.3*

# Set GROMACS environment
ENV PATH=/usr/local/gromacs/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/gromacs/lib64:$LD_LIBRARY_PATH

# Install Python packages (use system pip3, before adding amber to PATH)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install gmx_MMPBSA into the AmberTools micromamba env (Theo P1).
# Wraps AmberTools MMPBSA.py for GROMACS trajectories — computes ΔG_bind,
# per-residue energy decomposition. Must be in the same env as MMPBSA.py
# (AmberTools 23) so sander/cpptraj are on its PATH.
#
# Install via conda-forge (not pip): gmx_MMPBSA pins pandas==1.5.3 which
# has no Python 3.12 wheel on PyPI, so pip tries to build pandas from
# source and fails. conda-forge has pre-built wheels and its resolver
# handles the strict pin gracefully.
RUN CONDA_OVERRIDE_CUDA=11.8 /usr/local/bin/micromamba install -n amber \
        -c conda-forge gmx_mmpbsa -y && \
    /usr/local/bin/micromamba clean -afy && \
    ln -sf /opt/micromamba/envs/amber/bin/gmx_MMPBSA /usr/local/bin/gmx_MMPBSA

# Symlink AmberTools binaries and data (avoid adding conda python3 to PATH)
# tleap derives AMBERHOME from its own path, so /usr/local/dat must exist too
ENV AMBERHOME=/opt/micromamba/envs/amber
RUN for bin in antechamber sqm parmchk2 tleap teLeap atomtype bondtype am1bcc \
             packmol-memgen MCPB.py \
             cpptraj cpptraj.OMP sander sander.MPI ambpdb rdparm resp; do \
        [ -f /opt/micromamba/envs/amber/bin/$bin ] && \
            ln -sf /opt/micromamba/envs/amber/bin/$bin /usr/local/bin/; \
    done && \
    ln -sf /opt/micromamba/envs/amber/dat /usr/local/dat && \
    ln -sf /opt/micromamba/envs/amber/lib /usr/local/lib/amber

# Download CHARMM36m force field for GROMACS (membrane branch, doc 11).
# The MacKerell lab publishes GROMACS-formatted CHARMM36m annually.
# We place it alongside the built-in FFs so `gmx pdb2gmx -ff charmm36m`
# finds it automatically.
RUN GROMACS_TOP=/usr/local/gromacs/share/gromacs/top && \
    cd /tmp && \
    for i in 1 2 3 4 5; do \
        wget --timeout=60 --tries=3 -q \
             "http://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-jul2022.ff.tgz" \
             -O charmm36.ff.tgz && \
        [ -s charmm36.ff.tgz ] && break; \
        echo "CHARMM36m download attempt $i/5 failed, retrying in 30s..."; \
        rm -f charmm36.ff.tgz; \
        sleep 30; \
    done && \
    [ -s /tmp/charmm36.ff.tgz ] && \
    tar xzf charmm36.ff.tgz -C "$GROMACS_TOP" && \
    rm charmm36.ff.tgz && \
    ls "$GROMACS_TOP/charmm36-jul2022.ff/" | head -5 && \
    echo "CHARMM36m force field installed at $GROMACS_TOP/charmm36-jul2022.ff/"

# Create working directory
WORKDIR /app

# Copy application code
COPY main.py .
# Batch-job CLI executor — invoked when the job resource
# overrides the image's CMD to run run_md_job.py.
# Same image serves both the FastAPI service (CMD below) and the batch job.
COPY run_md_job.py .
COPY gromacs_md/ ./gromacs_md/
# Intake classifier package (see intake/).
# main.py imports from this on startup — must land before CMD runs.
COPY intake/ ./intake/
# Integration test script (scripts/test_intake.py) — optional but useful
# for in-container smoke verification against live APIs.
COPY scripts/ ./scripts/
# Membrane MD equilibration templates (semi-isotropic, CHARMM36m, doc 11).
COPY mdp_templates/ ./mdp_templates/

# Environment variables
ENV PORT=8024
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8024/health || exit 1

# Run the service
CMD ["python3", "main.py"]
