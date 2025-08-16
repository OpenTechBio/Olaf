# src/olaf/cli/create_agent.py
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
import typer

from olaf.agents.create_agent_system import (
    DEFAULT_AGENT_DIR,
    DEFAULT_SAMPLES_DIR,
    OLAF_HOME,
    define_global_policy,
    define_agents,
    connect_agents,
    assign_code_samples,
    save_configuration,
    Colors,
)

# Initialize Typer. `no_args_is_help=False` allows the callback to run by default.
app = typer.Typer(
    no_args_is_help=False,
    help="Create OLAF agent systems. Defaults to interactive mode."
)

def _run_interactive(output_dir: str, code_samples_dir: str):
    """
    The actual logic for the interactive agent system builder.
    """
    os.environ.setdefault("OLAF_HOME", str(OLAF_HOME))

    print(f"{Colors.HEADER}{Colors.BOLD}--- OLAF: Create Agent System (Interactive) ---{Colors.ENDC}")
    print(f"Using output directory: {output_dir}")
    print(f"Using code samples dir: {code_samples_dir}")

    global_policy_text = define_global_policy()
    agents_data = define_agents()
    if agents_data:
        connect_agents(agents_data)
        assign_code_samples(agents_data)
        save_configuration(global_policy_text, agents_data, output_dir)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    output_dir: str = typer.Option(
        str(DEFAULT_AGENT_DIR),
        "--output-dir",
        "-o",
        help="Where to save the resulting JSON.",
        show_default=True,
    ),
    code_samples_dir: str = typer.Option(
        str(DEFAULT_SAMPLES_DIR),
        "--code-samples-dir",
        help="Where to look for code samples by default.",
        show_default=True,
    ),
):
    """
    Manages agent system creation.

    If no subcommand (like 'quick') is provided, this tool runs in
    interactive mode.
    """
    # If a subcommand was not invoked, run the default interactive mode.
    if ctx.invoked_subcommand is None:
        _run_interactive(output_dir=output_dir, code_samples_dir=code_samples_dir)


@app.command("quick")
def quick(
    name: str = typer.Option(..., "--name", "-n", help="Filename (without .json) for the agent system."),
    policy: str = typer.Option("", "--policy", help="Optional global policy text."),
    output_dir: str = typer.Option(
        str(DEFAULT_AGENT_DIR),
        "--output-dir",
        "-o",
        help="Where to save the resulting JSON.",
        show_default=True,
    ),
):
    """
    Create a minimal agent system non-interactively.
    """
    from typing import Any, Dict
    agents: Dict[str, Any] = {}
    final_structure = {"global_policy": policy, "agents": agents}
    
    # Ensure the output directory exists
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"{name}.json"

    # Use an atomic write to prevent corrupted files
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), prefix=path.stem, suffix=".tmp") as tmp:
            json.dump(final_structure, tmp, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
        typer.secho(f"Created {path}", fg=typer.colors.GREEN)
    except OSError as e:
        typer.secho(f"Error creating file: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e