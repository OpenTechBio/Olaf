#!/usr/bin/env python3
"""
Interactive Agent Tester – Docker, Singularity‑API, or **Singularity‑Exec (offline‑REPL)**
=======================================================================================
Run a natural‑language chat loop that generates runnable Python, executes it inside a
container, and streams the results back. Works even on clusters where **no networking**
is allowed for Singularity by using a long‑lived REPL inside the container.

Back‑ends
---------
1. **docker**            – Docker daemon + container with FastAPI kernel.
2. **singularity**       – Singularity *instance* with FastAPI kernel.
3. **singularity-exec**  – Long‑lived `singularity exec` REPL that talks to
                           `/opt/offline_kernel.py --repl` (no TCP).
"""
from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── 3rd‑party deps ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    from openai import OpenAI, APIError
    import requests  # only needed for networked back‑ends
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.syntax import Syntax
    from rich.table import Table
except ImportError as e:
    print(f"Missing dependency: {e}. Install required packages.", file=sys.stderr)
    sys.exit(1)

# -- Local imports ---------------------------------------------------------------
from benchmarking.core.io_helpers import extract_python_code, display, select_dataset, collect_resources, get_initial_prompt, format_execute_response
from benchmarking.core.sandbox_management import init_docker, init_singularity, init_singularity_exec


console = Console()
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = PARENT_DIR / "datasets"
OUTPUTS_DIR = PARENT_DIR / "outputs"
ENV_FILE = PARENT_DIR / ".env"

# In‑container canonical paths
SANDBOX_DATA_PATH = "/workspace/dataset.h5ad"
SANDBOX_RESOURCES_DIR = "/workspace/resources"

# ==============================================================================
# 1 · Choose back‑end BEFORE importing heavy managers
# ==============================================================================
backend = Prompt.ask(
    "Choose sandbox backend",
    choices=["docker", "singularity", "singularity-exec"],
    default="docker",
)

# Ask user whether to force‑update the sandbox image/SIF
force_refresh = (
    Prompt.ask(
        "Force update sandbox environment?", choices=["y", "n"], default="n"
    ).lower()
    == "y"
)

is_exec_mode = backend == "singularity-exec"

# -----------------------------------------------------------------------------
# 1a · Docker (FastAPI) back‑end
# -----------------------------------------------------------------------------
if backend == "docker":
    _BackendManager, _SANDBOX_HANDLE, COPY_CMD, EXECUTE_ENDPOINT, STATUS_ENDPOINT = init_docker(
        SCRIPT_DIR, subprocess, console, force_refresh
    )
    SANDBOX_DATA_PATH = "dataset.h5ad"

# -----------------------------------------------------------------------------
# 1b · Singularity instance (FastAPI) back‑end
# -----------------------------------------------------------------------------

elif backend == "singularity":
    _BackendManager, _SANDBOX_HANDLE, COPY_CMD, EXECUTE_ENDPOINT, STATUS_ENDPOINT = init_singularity(
        SCRIPT_DIR, subprocess, console, force_refresh
    )
# -----------------------------------------------------------------------------
# 1c · Singularity exec (offline‑REPL) back‑end
# -----------------------------------------------------------------------------
elif backend == "singularity-exec":
    _BackendManager, _SANDBOX_HANDLE, COPY_CMD, EXECUTE_ENDPOINT, STATUS_ENDPOINT = init_singularity_exec(
        SCRIPT_DIR, SANDBOX_DATA_PATH, subprocess, console, force_refresh
    )
else:
    console.print("[red]Unknown backend.")
    sys.exit(1)

# ====================================================================================
# 4 · Networked FastAPI helpers (skipped for exec mode)
# ====================================================================================

def api_alive(max_retries: int = 10, delay: float = 1.5) -> bool:
    if is_exec_mode:
        return True  # nothing to ping
    for _ in range(max_retries):
        try:
            if requests.get(STATUS_ENDPOINT, timeout=2).json().get("status") == "ok":
                return True
        except Exception:
            time.sleep(delay)
    return False


# ====================================================================================
# 5 · Main interactive loop (unchanged)
# ====================================================================================

def run_interactive(prompt: str, dataset: Path, metadata: dict, resources: List[Tuple[Path, str]]):
    mgr = _BackendManager()
    console.print(f"Starting sandbox ({backend}) …")

    # Tell exec back‑end where data/resources are (creates bind list)
    if is_exec_mode and hasattr(mgr, "set_data"):
        mgr.set_data(dataset, resources)

    if not mgr.start_container():
        console.print("[red]Failed to start sandbox.[/red]")
        return

    if not api_alive():
        console.print("[red]Kernel API not responsive (networked back‑end).[/red]")
        return

    # For docker / singularity‑instance we still *attempt* docker cp (no‑op or warning otherwise)
    if not is_exec_mode:
        COPY_CMD(str(dataset), f"{_SANDBOX_HANDLE}:{SANDBOX_DATA_PATH}")
        for h, c in resources:
            COPY_CMD(str(h), f"{_SANDBOX_HANDLE}:{c}")

    resource_lines = [f"- {c} (from {h})" for h, c in resources] or ["- (none)"]
    sys_prompt = textwrap.dedent(
        f"""
        You are an AI assistant analysing a single‑cell dataset.
        Dataset path inside container: **{SANDBOX_DATA_PATH}**
        Additional resources:\n"""
        + "\n".join(resource_lines)
        + "\n\n"
        + textwrap.dedent(
            f"Dataset metadata:\n{json.dumps(metadata, indent=2)}\n\n"
            "Wrap runnable Python in triple‑backtick ```python blocks. Imports & variables persist within the container session."
        )
    )

    history = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    display(console, "system", sys_prompt)
    display(console, "user", prompt)

    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    turn = 0
    while True:
        turn += 1
        console.print(f"\n[bold]OpenAI call (turn {turn})…[/bold]")
        try:
            rsp = openai.chat.completions.create(
                model="gpt-4o", messages=history, temperature=0.7
            )
        except APIError as e:
            console.print(f"[red]OpenAI error: {e}[/red]")
            break
        assistant_msg = rsp.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_msg})
        display(console, "assistant", assistant_msg)

        code = extract_python_code(assistant_msg)
        if code:
            console.print("[cyan]Executing code…[/cyan]")
            try:
                if is_exec_mode:
                    exec_result = mgr.exec_code(code, timeout=300)
                else:
                    exec_result = requests.post(
                        EXECUTE_ENDPOINT, json={"code": code, "timeout": 300}, timeout=310
                    ).json()
                
                feedback = format_execute_response(exec_result, OUTPUTS_DIR)
            except Exception as exc:
                feedback = f"Code execution result:\n[Execution error on host: {exc}]"

            history.append({"role": "user", "content": feedback})
            display(console, "user", feedback)

        console.print("\n[bold]Next message (blank = continue, 'exit' to quit):[/bold]")
        try:
            user_in = input().strip()
        except (EOFError, KeyboardInterrupt):
            user_in = "exit"
        if user_in.lower() in {"exit", "quit"}:
            break
        if user_in:
            history.append({"role": "user", "content": user_in})
            display(console, "user", user_in)

    console.print("Stopping sandbox…")
    mgr.stop_container()


# ====================================================================================
# 6 · Entry‑point
# ====================================================================================

def main():
    ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(Path(ENV_FILE))
    if not os.getenv("OPENAI_API_KEY"):
        console.print(f"[red]OPENAI_API_KEY not set in {ENV_FILE}.[/red]")
        sys.exit(1)

    prompt = get_initial_prompt(console)
    data_p, meta = select_dataset(console, DATASETS_DIR)
    resources = collect_resources(console, SANDBOX_RESOURCES_DIR)
    run_interactive(prompt, data_p, meta, resources)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
