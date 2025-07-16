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

from benchmarking.prompt_testing.MultiAgentAutoTester import BACKEND_CHOICE
from rich.table import Table
from rich.prompt import Prompt
BACKEND_CHOICE = Prompt.ask(
    "LLM backend",
    choices=["chatgpt", "ollama"],
    default="chatgpt",
)
OLLAMA_HOST = "http://localhost:11434"
if BACKEND_CHOICE == "ollama":
    OLLAMA_HOST = Prompt.ask(
        "Ollama base URL",
        default="http://localhost:11434",
    )
# ── Dependencies ------------------------------------------------------------
try:
    from dotenv import load_dotenv
    from openai import OpenAI, APIError
    import requests
    from rich.console import Console

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
backend = Prompt.ask("Choose sandbox backend", choices=["docker", "singularity", "singularity-exec"], default="docker")
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
    instr = system.get_instructions()
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

def run(agent_system: AgentSystem, agent: Agent, roster_instr: str, dataset: Path, metadata: dict, resources: List[Tuple[Path, str]], benchmark_module: Optional[Path] = None):
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
        return roster_instr + "\n\n" + a.get_full_prompt(agent_system.global_policy) + "\n\n" + analysis_ctx

    history = [{"role": "system", "content": build_system(agent)}]
    first_user = "Beginning interactive session. You can ask questions or give commands."
    history.append({"role": "user", "content": first_user})
    display(console, "system", history[0]["content"])
    display(console, "user", first_user)

    if BACKEND_CHOICE == "chatgpt":
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[red]OPENAI_API_KEY not set in .env")
            sys.exit(1)
        openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        # Local Ollama needs no key; model defaults to “llama2”
        openai = OpenAI(host=OLLAMA_HOST)
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
                  
                # INJECT LOADED CODE SAMPLES ON DELEGATION ---
                if new_agent.code_samples:
                    sample_context = "Here are some relevant code samples for your task:"
                    for filename, code_content in new_agent.code_samples.items():
                        sample_context += f"\n\n--- Sample from: {filename} ---\n"
                        sample_context += f"```python\n{code_content.strip()}\n```"
                    
                    history.append({"role": "user", "content": sample_context})
                    display(console, "user", sample_context) # Display for clarity

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
            
        def input_loop():
            if benchmark_module:
                console.print("\n[bold]Next message (blank = continue, 'benchmark' to run benchmarks, 'exit' to quit):[/bold]")
            else:
                console.print("\n[bold]Next message (blank = continue, 'exit' to quit):[/bold]")
            try:
                user_in = input().strip()
            except (EOFError, KeyboardInterrupt):
                user_in = "exit"
            if user_in.lower() in {"exit", "quit"}:
                return "break"
            if user_in.lower() == "benchmark" and benchmark_module:
                run_benchmark(mgr, benchmark_module)
                input_loop()  # Recurse to continue the loop after benchmarks
            if user_in:
                history.append({"role": "user", "content": user_in})
                display(console, "user", user_in)
        input_val = input_loop()
        if input_val == "break":  # User chose to exit
            break

    console.print("Stopping sandbox…")
    mgr.stop_container()


# ===========================================================================
# 4 · Benchmarking
# ===========================================================================

def get_benchmark_module(console: Console, parent_dir: Path) -> Optional[Path]:
    """
    Prompts the user to select a benchmark module from the available ones.
    Returns the path to the selected module or None if no selection is made.
    """
    benchmark_dir = parent_dir / "auto_metrics"
    if not benchmark_dir.exists():
        console.print("[red]No benchmarks directory found.[/red]")
        return None

    modules = list(benchmark_dir.glob("*.py"))
    # remove AutoMetric.py from modules (it is the base class)
    modules = [m for m in modules if m.name != "AutoMetric.py"]
    if not modules:
        console.print("[red]No benchmark modules found.[/red]")
        return None

    console.print("\n[bold]Available benchmark modules:[/bold]")
    for i, mod in enumerate(modules, start=1):
        console.print(f"{i}. {mod.name}")

    choice = Prompt.ask("Select a benchmark module by number (or press Enter to skip)", default="")
    if not choice:
        return None

    try:
        index = int(choice) - 1
        if 0 <= index < len(modules):
            return modules[index]
        else:
            console.print("[red]Invalid selection.[/red]")
            return None
    except ValueError:
        console.print("[red]Invalid input. Please enter a number.[/red]")
        return None
    
def run_benchmark(mgr, benchmark_module: str):
    """
    Runs the benchmark module and displays the results.
    """
    console.print(f"\n[bold cyan]Running benchmark module: {benchmark_module}[/bold cyan]")
    autometric_base_path = benchmark_module.parent / "AutoMetric.py"
    try:
        # Read the abstract base class definition
        with open(autometric_base_path, "r") as f:
            autometric_code = f.read()

        with open(benchmark_module, "r") as f:
            benchmark_code = f.read()
    except FileNotFoundError:
        console.print(f"[red]Benchmark module not found at: {benchmark_module}[/red]")
        return

    code_to_execute = f"""
# --- Code from AutoMetric.py --- 
{autometric_code}
# --- Code from {benchmark_module.name} ---
{benchmark_code}
"""
    console.print("[cyan]Executing benchmark code...[/cyan]")
    try:
        if is_exec_mode:
            exec_result = mgr.exec_code(code_to_execute, timeout=300)
        else:
            exec_result = requests.post(
                EXECUTE_ENDPOINT, json={"code": code_to_execute, "timeout": 300}, timeout=310
            ).json()

        # Create a table to display the results
        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        # Assuming the benchmark module returns a dictionary of results
        stdout = exec_result.get("stdout", "")
        try:
            result_dict = json.loads(stdout.strip().splitlines()[-1])  # Parse last printed line
        except Exception as e:
            console.print(f"[yellow]Warning: Could not parse JSON from stdout: {e}[/yellow]")
            result_dict = {}

        if exec_result.get("status") == "ok" and isinstance(result_dict, dict):
            for key, value in result_dict.items():
                table.add_row(str(key), str(value))
        else:
            table.add_row("Error", exec_result.get("stderr") or "An unknown error occurred.")

        console.print(table)

    except Exception as exc:
        console.print(f"[red]Benchmark execution error: {exc}[/red]")

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
    benchmark_module = get_benchmark_module(console, PARENT_DIR)
    res = collect_resources(console, SANDBOX_RESOURCES_DIR)
    run(sys, drv, roster, dp, meta, res, benchmark_module)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
