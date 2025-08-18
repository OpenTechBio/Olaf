# Use official Python slim image based on Debian
FROM python:3.11-slim

# Set DEBIAN_FRONTEND to noninteractive to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

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
    zip \
    # Configure timezone
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    # Clean up apt cache
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Create Non-Root User & Group ---
ARG NB_USER="sandboxuser"
ARG NB_UID=1001
ARG NB_GID=1001
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}
ENV PATH=${HOME}/.local/bin:${PATH}

RUN groupadd -g ${NB_GID} ${NB_USER} && \
    useradd -m -s /bin/bash -u ${NB_UID} -g ${NB_GID} ${NB_USER} && \
    adduser ${NB_USER} sudo && \
    echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# --- Create /workspace Directory as Root ---
# This step is now done BEFORE switching to the non-root user.
RUN mkdir -p /workspace && \
    chown ${NB_USER}:${NB_GID} /workspace

# --- Install Python Dependencies ---
COPY ./requirements.txt /tmp/requirements.txt

# Now, switch to the non-root user
USER ${NB_USER}
WORKDIR ${HOME}

RUN python -m pip install --no-cache-dir --upgrade pip --user && \
    python -m pip install --no-cache-dir --user \
    ipython==8.12.0 \
    traitlets==5.9.0 \
    jupyter_client==8.3.0 \
    jupyter_core==5.3.1 \
    pyzmq==25.1.0 \

    tornado==6.3.2 \
    ipykernel==6.25.1 \
    # FastAPI dependencies
    fastapi \
    uvicorn[standard] \
    python-multipart \
    # Install user requirements from the temporary location
    -r /tmp/requirements.txt

# --- Application Setup ---
COPY --chown=${NB_USER}:${NB_GID} ./kernel_api.py ${HOME}/kernel_api.py
COPY --chown=${NB_USER}:${NB_GID} ./start_kernel.py ${HOME}/start_kernel.py
COPY --chown=${NB_USER}:${NB_GID} ./start.sh ${HOME}/start.sh

# Create user-specific directories and make scripts executable
# The /workspace creation has been moved, so we only handle user-owned files here.
RUN mkdir -p ${HOME}/.local/share/jupyter \
             ${HOME}/.ipython/profile_default/startup \
             ${HOME}/.ipython/profile_default/static && \
    chmod +x ${HOME}/start_kernel.py ${HOME}/start.sh

# --- Runtime Configuration ---
EXPOSE 8000
ENV IPY_BASE_PORT=4000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/home/sandboxuser/start.sh"]