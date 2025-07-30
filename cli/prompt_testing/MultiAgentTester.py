#!/usr/bin/env python3
"""
Interactive and Auto Agent System Tester (v1.4-refactored)
=========================================================
This script combines two execution modes:
- Interactive Mode: A standard chat-like interface for manual testing.
- Automated Mode: Runs the agent with a given prompt for a set number of turns
  for benchmarking purposes.

Use the --auto flag to enable automated mode.
This version has been refactored to reduce code duplication.
"""
from __future__ import annotations

import argparse
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
from typing import Dict, List, Optional, Tuple

from rich.prompt import Prompt
from rich.table import Table
# -- Pick LLM backend ---------------------------------------------------
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

    if BACKEND_CHOICE == "ollama":
        from cli.core.ollama_wrapper import OllamaClient as OpenAI
        APIError = Exception  # Ollama does not have a specific APIError
    else:
        from openai import APIError, OpenAI

    import requests
    from rich.console import Console
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

# ── Agent framework ---------------------------------------------------------
try:
    from cli.agents.AgentSystem import Agent, AgentSystem
except ImportError:
    print("[ERROR] Could not import backend.agents.agent_system", file=sys.stderr)
    raise

# ── Local helpers -----------------------------------------------------------
from cli.core.io_helpers import (
    collect_resources,
    display,
    extract_python_code,
    format_execute_response,
    get_initial_prompt,
    load_bp_json,
    select_dataset,
)
from cli.core.sandbox_management import (
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

# ── Benchmark persistence --------------------------------------------------
timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
_LEDGER_PATH = OUTPUTS_DIR / f"benchmark_history_{timestamp}.jsonl"
_SNIPPET_DIR = OUTPUTS_DIR / "snippets"
_SNIPPET_DIR.mkdir(exist_ok=True, parents=True)
_LEDGER_PATH.parent.mkdir(exist_ok=True, parents=True)

# ===========================================================================
# 1 · Backend selection
# ===========================================================================
backend = Prompt.ask(
    "Choose sandbox backend",
    choices=["docker", "singularity", "singularity-exec"],
    default="docker",
)
force_refresh = (
    Prompt.ask("Force refresh environment?", choices=["y", "n"], default="n").lower() == "y"
)
is_exec_mode = backend == "singularity-exec"

if backend == "docker":
    (
        _BackendManager,
        _SANDBOX_HANDLE,
        COPY_CMD,
        EXECUTE_ENDPOINT,
        STATUS_ENDPOINT,
    ) = init_docker(SCRIPT_DIR, subprocess, console, force_refresh)
    SANDBOX_DATA_PATH = "dataset.h5ad"
elif backend == "singularity":
    (
        _BackendManager,
        _SANDBOX_HANDLE,
        COPY_CMD,
        EXECUTE_ENDPOINT,
        STATUS_ENDPOINT,
    ) = init_singularity(SCRIPT_DIR, subprocess, console, force_refresh)
elif backend == "singularity-exec":
    (
        _BackendManager,
        _SANDBOX_HANDLE,
        COPY_CMD,
        EXECUTE_ENDPOINT,
        STATUS_ENDPOINT,
    ) = init_singularity_exec(
        SCRIPT_DIR, SANDBOX_DATA_PATH, subprocess, console, force_refresh
    )
else:
    console.print("[red]Unknown backend.")
    sys.exit(1)


# ===========================================================================
# 2 · Common Helpers
# ===========================================================================
def load_agent_system() -> Tuple[AgentSystem, Agent, str]:
    """Load the agent system from a JSON blueprint."""
    bp = load_bp_json(console)
    if not bp.exists():
        console.print(f"[red]Blueprint {bp} not found.")
        sys.exit(1)
    system = AgentSystem.load_from_json(str(bp))
    driver_name = Prompt.ask(
        "Driver agent",
        choices=list(system.agents.keys()),
        default=list(system.agents)[0],
    )
    driver = system.get_agent(driver_name)
    instr = system.get_instructions()
    return system, driver, instr


_DELEG_RE = re.compile(r"delegate_to_([A-Za-z0-9_]+)")


def detect_delegation(msg: str) -> Optional[str]:
    """Return the *full* command name (e.g. 'delegate_to_coder') if present."""
    m = _DELEG_RE.search(msg)
    return f"delegate_to_{m.group(1)}" if m else None


def api_alive(url: str, tries: int = 10) -> bool:
    """Check if the API is responsive."""
    if is_exec_mode:
        return True
    for _ in range(tries):
        try:
            if requests.get(url, timeout=2).json().get("status") == "ok":
                return True
        except Exception:
            time.sleep(1.5)
    return False


def _dump_code_snippet(run_id: str, code: str) -> str:
    """Write <run_id>.py under outputs/snippets/ and return the relative path."""
    snippet_path = _SNIPPET_DIR / f"{run_id}.py"
    snippet_path.write_text(code, encoding="utf-8")
    return str(snippet_path.relative_to(OUTPUTS_DIR))


def _save_benchmark_record(*, run_id: str, results: dict, meta: dict, code: str | None):
    """Append a JSONL record for the benchmark run."""
    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run": run_id,
        "dataset": meta.get("name"),
        "results": results,
    }
    if code:
        record["code_path"] = _dump_code_snippet(run_id, code)
    with _LEDGER_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ===========================================================================
# 3 · Unified Benchmark Runner
# ===========================================================================
def run_benchmark(
    mgr,
    benchmark_module: Path,
    *,
    is_auto: bool,
    metadata: Optional[Dict] = None,
    agent_name: Optional[str] = None,
    code_snippet: Optional[str] = None,
) -> str:
    """
    Execute a benchmark module.
    In auto mode, saves results and returns a result string for the history.
    In interactive mode, just prints results to the console.
    """
    console.print(
        f"\n[bold cyan]Running benchmark module: {benchmark_module.name}[/bold cyan]"
    )
    autometric_base_path = benchmark_module.parent / "AutoMetric.py"
    try:
        with open(autometric_base_path, "r") as f:
            autometric_code = f.read()
        with open(benchmark_module, "r") as f:
            benchmark_code = f.read()
    except FileNotFoundError:
        err = f"Benchmark module not found at: {benchmark_module}"
        console.print(f"[red]{err}[/red]")
        return err if is_auto else ""

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

        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        stdout = exec_result.get("stdout", "")
        result_dict = {}
        try:
            result_dict = json.loads(stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as e:
            console.print(f"[yellow]Warning: Could not parse JSON from stdout: {e}[/yellow]")

        if exec_result.get("status") == "ok" and isinstance(result_dict, dict):
            for key, value in result_dict.items():
                table.add_row(str(key), str(value))
            if is_auto:
                _save_benchmark_record(
                    run_id=f"{benchmark_module.stem}:{agent_name}:{int(time.time())}",
                    results=result_dict,
                    meta=metadata,
                    code=code_snippet,
                )
        else:
            table.add_row("Error", exec_result.get("stderr") or "An unknown error occurred.")
        console.print(table)

        if is_auto:
            return "Benchmark results:\n" + json.dumps(result_dict or {"error": "see console"})
    except Exception as exc:
        err_msg = f"Benchmark execution error: {exc}"
        console.print(f"[red]{err_msg}[/red]")
        if is_auto:
            return err_msg
    return ""


# ===========================================================================
# 4 · Unified Main Execution Loop
# ===========================================================================
def run(
    agent_system: AgentSystem,
    agent: Agent,
    roster_instr: str,
    dataset: Path,
    metadata: dict,
    resources: List[Tuple[Path, str]],
    *,
    is_auto: bool,
    initial_user_message: str,
    benchmark_modules: Optional[List[Path]] = None,
    tries: int = 1,
):
    """Main driver for both interactive and automated execution."""
    last_code_snippet: str | None = None
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
        f"Dataset path: **{SANDBOX_DATA_PATH}**\nResources:\n"
        + "\n".join(res_lines)
        + "\n\nMetadata:\n"
        + json.dumps(metadata, indent=2)
    )

    def build_system(a: Agent) -> str:
        return (
            roster_instr
            + "\n\n"
            + a.get_full_prompt(agent_system.global_policy)
            + "\n\n"
            + analysis_ctx
        )

    history = [{"role": "system", "content": build_system(agent)}]
    history.append({"role": "user", "content": initial_user_message})
    display(console, "system", history[0]["content"])
    display(console, "user", initial_user_message)

    if BACKEND_CHOICE == "chatgpt":
        openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        openai = OpenAI(host=OLLAMA_HOST, model="deepseek-r1:70b")

    current_agent = agent
    turn = 0
    tries_left = tries

    while True:
        turn += 1
        console.print(f"\n[bold]OpenAI call (turn {turn})…")
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o", messages=history, temperature=0.7
            )
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
                history.append(
                    {"role": "assistant", "content": f"🔄 Routing to **{tgt}** (command `{cmd}`)"}
                )
                if new_agent.code_samples:
                    sample_context = "Here are some relevant code samples for your task:"
                    for filename, code_content in new_agent.code_samples.items():
                        sample_context += f"\n\n--- Sample from: {filename} ---\n"
                        sample_context += f"```python\n{code_content.strip()}\n```"
                    history.append({"role": "user", "content": sample_context})
                    display(console, "user", sample_context)
                current_agent = new_agent
                history.insert(0, {"role": "system", "content": build_system(new_agent)})
                continue

        code = extract_python_code(msg)
        if code:
            last_code_snippet = code
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

        # --- Mode-specific logic ---
        if is_auto:
            if benchmark_modules:  # In auto mode, this is a list with 0 or 1 module
                result_str = run_benchmark(
                    mgr,
                    benchmark_modules[0],
                    is_auto=True,
                    metadata=metadata,
                    agent_name=current_agent.name,
                    code_snippet=last_code_snippet,
                )
                history.append({"role": "user", "content": result_str})
                display(console, "user", result_str)

            tries_left -= 1
            if tries_left <= 0:
                console.print("[bold green]Auto run finished.[/bold green]")
                break
            history.append({"role": "user", "content": ""})  # Auto-continue
        else:
            # Interactive mode input loop
            while True:
                prompt_text = (
                    "\n[bold]Next message (blank = continue, 'benchmark' to run, 'exit' to quit):[/bold]"
                    if benchmark_modules
                    else "\n[bold]Next message (blank = continue, 'exit' to quit):[/bold]"
                )
                try:
                    user_input = Prompt.ask(prompt_text, default="").strip()
                except (EOFError, KeyboardInterrupt):
                    user_input = "exit"

                if user_input.lower() in {"exit", "quit"}:
                    console.print("Stopping sandbox…")
                    mgr.stop_container()
                    return  # Exit the entire run function

                if user_input.lower() == "benchmark":
                    if benchmark_modules:
                        for bm_module in benchmark_modules:
                            run_benchmark(mgr, bm_module, is_auto=False)
                        continue  # Re-prompt after running benchmarks
                    else:
                        console.print("[yellow]No benchmark modules selected at startup.[/yellow]")
                        continue
                
                if user_input:
                    history.append({"role": "user", "content": user_input})
                    display(console, "user", user_input)
                break  # Exit input loop and proceed to next agent turn

    console.print("Stopping sandbox…")
    mgr.stop_container()


# ===========================================================================
# 5 · Mode-Specific Setup Functions
# ===========================================================================
def get_benchmark_modules(console: Console, parent_dir: Path) -> Optional[List[Path]]:
    """Prompt user to select one or more benchmark modules for interactive mode."""
    benchmark_dir = parent_dir / "auto_metrics"
    if not benchmark_dir.exists():
        return None
    modules = [m for m in benchmark_dir.glob("*.py") if m.name != "AutoMetric.py"]
    if not modules:
        return None
    console.print("\n[bold]Available benchmark modules:[/bold]")
    for i, mod in enumerate(modules, start=1):
        console.print(f"{i}. {mod.name}")
    console.print(f"{len(modules)+1}. Select All")
    choices_str = Prompt.ask("Select modules (e.g., 1 2 or 1,2,3) (Enter to skip)", default="")
    choices = re.split(r"[,|\s]+", choices_str.strip())
    if not choices or choices == [""]:
        return None
    selected = []
    try:
        for choice in choices:
            if not choice: continue
            index = int(choice) - 1
            if index == len(modules): return modules  # Select All
            if 0 <= index < len(modules): selected.append(modules[index])
    except (ValueError, IndexError):
        console.print("[red]Invalid selection.[/red]")
        return None
    return selected


# ===========================================================================
# 6 · Entry Point
# ===========================================================================
def main():
    """Main entry point to parse args and start the correct mode."""
    parser = argparse.ArgumentParser(
        description="Interactive or Automated Agent System Tester.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--auto", action="store_true", help="Run in automated benchmark mode.")
    args = parser.parse_args()

    load_dotenv(ENV_FILE)
    if BACKEND_CHOICE == "chatgpt" and not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY not set in .env[/red]")
        sys.exit(1)

    sys_, drv, roster = load_agent_system()
    dp, meta = select_dataset(console, DATASETS_DIR)
    res = collect_resources(console, SANDBOX_RESOURCES_DIR)

    if args.auto:
        console.print("[bold green]🚀 Running in Automated Mode...[/bold green]")
        benchmark_module = get_benchmark_modules(console, PARENT_DIR)
        initial_user_message = Prompt.ask("Initial user message", default="What should I do with this dataset?")
        try:
            tries = int(Prompt.ask("Number of automatic turns", default="1"))
            if tries <= 0: raise ValueError
        except ValueError:
            console.print("[yellow]Invalid number – defaulting to 1.[/yellow]")
            tries = 1
        run(
            agent_system=sys_,
            agent=drv,
            roster_instr=roster,
            dataset=dp,
            metadata=meta,
            resources=res,
            is_auto=True,
            initial_user_message=initial_user_message,
            benchmark_modules=[benchmark_module] if benchmark_module else [],
            tries=tries,
        )
    else:
        console.print("[bold blue]🚀 Running in Interactive Mode...[/bold blue]")
        benchmark_modules = get_benchmark_modules(console, PARENT_DIR)
        run(
            agent_system=sys_,
            agent=drv,
            roster_instr=roster,
            dataset=dp,
            metadata=meta,
            resources=res,
            is_auto=False,
            initial_user_message="Beginning interactive session. You can ask questions or give commands.",
            benchmark_modules=benchmark_modules,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\nInterrupted by user. Exiting.")