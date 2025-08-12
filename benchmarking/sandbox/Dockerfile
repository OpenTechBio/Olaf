# GPU-capable base (Ubuntu 22.04 + CUDA 12.2 + Python 3.11)
FROM rapidsai/rapidsai-core:24.06-cuda12.2-runtime-ubuntu22.04-py3.11

# Prevent interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive
# Let pip see NVIDIA’s index (for rapids-singlecell[rapids12])
ENV PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com
# Add user-local bin to PATH (keeps your --user installs working)
ENV PATH=/home/sandboxuser/.local/bin:${PATH}

# --- Install System Dependencies ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tini \
    tzdata \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libsodium-dev \
    libzmq3-dev \
    gcc \
    g++ \
    sudo \
    curl \
    wget \
    git \
    vim \
    nano \
    unzip \
    zip && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Create Non-Root User & Group ---
ARG NB_USER="sandboxuser"
ARG NB_UID=1001
ARG NB_GID=1001
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}

RUN groupadd -g ${NB_GID} ${NB_USER} && \
    useradd -m -s /bin/bash -u ${NB_UID} -g ${NB_GID} ${NB_USER} && \
    adduser ${NB_USER} sudo && \
    echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# --- Install Python Dependencies ---
# Copy requirements after creating the user to preserve caching
COPY --chown=${NB_USER}:${NB_GID} ./requirements.txt /tmp/requirements.txt

# Run pip installs as the target user (keeps permissions clean)
USER ${NB_USER}
WORKDIR ${HOME}

# Upgrade pip and install deps (NVIDIA index is available via ENV)
RUN python -m pip install --no-cache-dir --upgrade pip --user && \
    python -m pip install --no-cache-dir --user \
      ipython==8.12.0 \
      traitlets==5.9.0 \
      jupyter_client==8.3.0 \
      jupyter_core==5.3.1 \
      pyzmq==25.1.0 \
      tornado==6.3.2 \
      ipykernel==6.25.1 \
      fastapi \
      uvicorn[standard] \
      python-multipart && \
    # Install your requirements (can include `rapids-singlecell[rapids12]`)
    python -m pip install --no-cache-dir --user \
      --extra-index-url=https://pypi.nvidia.com \
      -r /tmp/requirements.txt

# --- Application Setup ---
COPY --chown=${NB_USER}:${NB_GID} ./kernel_api.py ${HOME}/kernel_api.py
COPY --chown=${NB_USER}:${NB_GID} ./start_kernel.py ${HOME}/start_kernel.py
COPY --chown=${NB_USER}:${NB_GID} ./start.sh ${HOME}/start.sh

RUN mkdir -p ${HOME}/.local/share/jupyter \
             ${HOME}/.ipython/profile_default/startup \
             ${HOME}/.ipython/profile_default/static && \
    chmod +x ${HOME}/start_kernel.py ${HOME}/start.sh

# --- Runtime Configuration ---
EXPOSE 8000
ENV IPY_BASE_PORT=4000

# Use tini as the entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/home/sandboxuser/start.sh"]