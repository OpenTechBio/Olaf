#!/usr/bin/env python3
"""
Interactive Agent System Tester (v1.1)
======================================
• **New in v1.1** – Smarter delegation detection.
  The router now recognises any of the following patterns in an assistant reply
  when deciding to switch agents:

  ```text
  //delegate_to_coder
  delegate_to_coder
  `delegate_to_coder`
  Executing command: `delegate_to_coder`
  ```

  No need to rigidly start the reply with the token – the regex scans the whole
  message. Once detected, we alert the user ("🔄 Routing to …") and prepend the
  new agent’s system prompt.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# ── Dependencies ------------------------------------------------------------
try:
    from dotenv import load_dotenv
    from openai import OpenAI, APIError
    import requests
    from rich.console import Console
    from rich.prompt import Prompt
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

# ── Agent framework ---------------------------------------------------------
try:
    from benchmarking.agents.AgentSystem import AgentSystem, Agent
except ImportError:
    print("[ERROR] Could not import backend.agents.agent_system", file=sys.stderr)
    raise

# ── Local helpers -----------------------------------------------------------
from benchmarking.core.io_helpers import (
    extract_python_code,
    display,
    select_dataset,
    collect_resources,
    get_initial_prompt,
    format_execute_response
)
from benchmarking.core.sandbox_management import (
    init_docker,
    init_singularity,
    init_singularity_exec,
)

console = Console()
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = PARENT_DIR / "datasets"
OUTPUTS_DIR = PARENT_DIR / "outputs"
ENV_FILE = PARENT_DIR / ".env"

SANDBOX_DATA_PATH = "/workspace/dataset.h5ad"
SANDBOX_RESOURCES_DIR = "/workspace/resources"

# ===========================================================================
# 1 · Backend selection
# ===========================================================================
backend = Prompt.ask("Choose backend", choices=["docker", "singularity", "singularity-exec"], default="docker")
force_refresh = Prompt.ask("Force refresh environment?", choices=["y", "n"], default="n").lower() == "y"
is_exec_mode = backend == "singularity-exec"

if backend == "docker":
    _BackendManager, _SANDBOX_HANDLE, COPY_CMD, EXECUTE_ENDPOINT, STATUS_ENDPOINT = init_docker(
        SCRIPT_DIR, subprocess, console, force_refresh
    )
    SANDBOX_DATA_PATH = "dataset.h5ad"
elif backend == "singularity":
    _BackendManager, _SANDBOX_HANDLE, COPY_CMD, EXECUTE_ENDPOINT, STATUS_ENDPOINT = init_singularity(
        SCRIPT_DIR, subprocess, console, force_refresh
    )
elif backend == "singularity-exec":
    _BackendManager, _SANDBOX_HANDLE, COPY_CMD, EXECUTE_ENDPOINT, STATUS_ENDPOINT = init_singularity_exec(
        SCRIPT_DIR, SANDBOX_DATA_PATH, subprocess, console, force_refresh
    )
else:
    console.print("[red]Unknown backend.")
    sys.exit(1)

# ===========================================================================
# 2 · Agent helpers
# ===========================================================================

def load_agent_system() -> Tuple[AgentSystem, Agent, str]:
    bp = Path(Prompt.ask("Blueprint JSON", default="system_blueprint.json")).expanduser()
    if not bp.exists():
        console.print(f"[red]Blueprint {bp} not found.")
        sys.exit(1)
    system = AgentSystem.load_from_json(str(bp))
    driver_name = Prompt.ask("Driver agent", choices=list(system.agents.keys()), default=list(system.agents)[0])
    driver = system.get_agent(driver_name)
    instr = system.get_insturctions()
    return system, driver, instr

# Smarter regex – matches inline/backtick/explicit styles
# Match variations like //<backtick>delegate_to_coder<backtick>, with optional punctuation.
_DELEG_RE = re.compile(r"delegate_to_([A-Za-z0-9_]+)")

def detect_delegation(msg: str) -> Optional[str]:
    """Return the *full* command name (e.g. 'delegate_to_coder') if present."""
    m = _DELEG_RE.search(msg)
    return f"delegate_to_{m.group(1)}" if m else None


def api_alive(url: str, tries: int = 10) -> bool:
    if is_exec_mode:
        return True
    for _ in range(tries):
        try:
            if requests.get(url, timeout=2).json().get("status") == "ok":
                return True
        except Exception:
            time.sleep(1.5)
    return False

# ===========================================================================
# 3 · Interactive loop
# ===========================================================================

def run(agent_system: AgentSystem, agent: Agent, roster_instr: str, dataset: Path, metadata: dict, resources: List[Tuple[Path, str]]):
    mgr = _BackendManager()
    console.print(f"Launching sandbox ({backend})…")

    if is_exec_mode and hasattr(mgr, "set_data"):
        mgr.set_data(dataset, resources)
    if not mgr.start_container():
        console.print("[red]Failed to start sandbox")
        return
    if not api_alive(STATUS_ENDPOINT):
        console.print("[red]Kernel API not responsive.")
        return

    if not is_exec_mode:
        COPY_CMD(str(dataset), f"{_SANDBOX_HANDLE}:{SANDBOX_DATA_PATH}")
        for hp, cp in resources:
            COPY_CMD(str(hp), f"{_SANDBOX_HANDLE}:{cp}")

    res_lines = [f"- {c} (from {h})" for h, c in resources] or ["- (none)"]
    analysis_ctx = textwrap.dedent(
        f"Dataset path: **{SANDBOX_DATA_PATH}**\nResources:\n" + "\n".join(res_lines) + "\n\nMetadata:\n" + json.dumps(metadata, indent=2)
    )

    def build_system(a: Agent) -> str:
        return roster_instr + "\n\n" + a.get_full_prompt() + "\n\n" + analysis_ctx

    history = [{"role": "system", "content": build_system(agent)}]
    first_user = "Beginning interactive session. You can ask questions or give commands."
    history.append({"role": "user", "content": first_user})
    display(console, "system", history[0]["content"])
    display(console, "user", first_user)

    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    current_agent = agent
    turn = 0

    while True:
        turn += 1
        console.print(f"\n[bold]OpenAI call (turn {turn})…")
        try:
            resp = openai.chat.completions.create(model="gpt-4o", messages=history, temperature=0.7)
        except APIError as e:
            console.print(f"[red]OpenAI error: {e}")
            break
        msg = resp.choices[0].message.content
        history.append({"role": "assistant", "content": msg})
        display(console, f"assistant ({current_agent.name})", msg)

        cmd = detect_delegation(msg)
        if cmd and cmd in current_agent.commands:
            tgt = current_agent.commands[cmd].target_agent
            new_agent = agent_system.get_agent(tgt)
            if new_agent:
                console.print(f"[yellow]🔄 Routing to '{tgt}' via {cmd}")
                history.append({"role": "assistant", "content": f"🔄 Routing to **{tgt}** (command `{cmd}`)"})
                current_agent = new_agent
                history.insert(0, {"role": "system", "content": build_system(new_agent)})
                continue

        code = extract_python_code(msg)
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

        console.print("\n[bold]Next message (blank = continue, 'exit' to quit):")
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

# ===========================================================================
# 4 · Entry point
# ===========================================================================

def main():
    load_dotenv(ENV_FILE)
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY not set in .env")
        sys.exit(1)

    sys, drv, roster = load_agent_system()
    dp, meta = select_dataset(console, DATASETS_DIR)
    res = collect_resources(console, SANDBOX_RESOURCES_DIR)
    run(sys, drv, roster, dp, meta, res)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
