# Use official Python slim image based on Debian (adjust version if needed)
FROM python:3.11-slim

# Set DEBIAN_FRONTEND to noninteractive to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# --- Install System Dependencies ---
# Combine apt-get operations into a single layer to leverage caching.
# This layer rarely changes unless system dependencies are added/removed.
# Install tini, tzdata, build tools, C libraries, and utilities.
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
# These arguments and user setup steps rarely change.
ARG NB_USER="sandboxuser"
ARG NB_UID=1001
ARG NB_GID=1001
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}
# Add user's local bin to PATH early
ENV PATH=${HOME}/.local/bin:${PATH}

# Create group, user, add to sudoers (run as root)
RUN groupadd -g ${NB_GID} ${NB_USER} && \
    useradd -m -s /bin/bash -u ${NB_UID} -g ${NB_GID} ${NB_USER} && \
    adduser ${NB_USER} sudo && \
    echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# --- Install Python Dependencies ---
COPY ./requirements.txt /tmp/requirements.txt

# Install Python packages. This layer is cached if requirements.txt hasn't changed.
# Run pip installs as the target user to ensure correct permissions and paths.
# Switch user and set working directory *before* pip install --user.
USER ${NB_USER}
WORKDIR ${HOME}

RUN python -m pip install --no-cache-dir --upgrade pip --user && \
    python -m pip install --no-cache-dir --user \
    # Core Jupyter components (pin versions for stability)
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
# Copy application code and scripts AFTER dependencies are installed.
# Changes to these files will only invalidate the cache from this point.
COPY --chown=${NB_USER}:${NB_GID} ./kernel_api.py ${HOME}/kernel_api.py
COPY --chown=${NB_USER}:${NB_GID} ./start_kernel.py ${HOME}/start_kernel.py
COPY --chown=${NB_USER}:${NB_GID} ./start.sh ${HOME}/start.sh

# Create user directories and make scripts executable in a single layer
RUN mkdir -p ${HOME}/.local/share/jupyter \
             ${HOME}/.ipython/profile_default/startup \
             ${HOME}/.ipython/profile_default/static && \
    chmod +x ${HOME}/start_kernel.py ${HOME}/start.sh

# --- Runtime Configuration ---
# Expose the FastAPI port (informational)
EXPOSE 8000

# Set environment variable for kernel port (used by start_kernel.py)
ENV IPY_BASE_PORT=4000

# Use tini as the entrypoint; it will execute the CMD
# Ensure tini installed via apt is in the default PATH or use /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Set the default command to run the startup script from user's home
CMD ["/home/sandboxuser/start.sh"]
