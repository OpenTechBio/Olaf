# olaf/config.py
import os
from pathlib import Path
from platformdirs import PlatformDirs

# Define app-specific identifiers for platformdirs
APP_NAME = "olaf"
APP_AUTHOR = "OpenTechBio"
dirs = PlatformDirs(APP_NAME, APP_AUTHOR)

# Define the root directory for all user-specific OLAF files.
# This respects the OLAF_HOME environment variable but has a sensible default.
OLAF_HOME = Path(os.environ.get("OLAF_HOME", dirs.user_data_dir)).expanduser()

# Define standard subdirectories
DEFAULT_AGENT_DIR = OLAF_HOME / "agent_systems"
DEFAULT_DATASETS_DIR = OLAF_HOME / "datasets"

# Define the path to the environment file for storing secrets like API keys
ENV_FILE = OLAF_HOME / ".env"

def init_olaf_home():
    """Ensures the main OLAF directory and its subdirectories exist."""
    OLAF_HOME.mkdir(parents=True, exist_ok=True)
    DEFAULT_AGENT_DIR.mkdir(exist_ok=True)
    DEFAULT_DATASETS_DIR.mkdir(exist_ok=True)

# Automatically initialize directories when this module is imported
init_olaf_home()