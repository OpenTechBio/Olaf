# olaf/cli/run_cli.py
import os
import textwrap
from pathlib import Path
from typing import List, cast
import subprocess

import typer
from rich.console import Console
from rich.prompt import Prompt
from dotenv import load_dotenv

# Import your project's modules and shared configuration
from olaf.config import DEFAULT_AGENT_DIR, ENV_FILE

from olaf.agents.AgentSystem import AgentSystem
from olaf.core.io_helpers import collect_resources
from olaf.core.sandbox_management import (init_docker, init_singularity, init_singularity_exec)
from olaf.execution.runner import run_agent_session, SandboxManager
from olaf.datasets.czi_datasets import get_datasets_dir

# --- Define package-internal paths ---
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_AGENTS_DIR = PACKAGE_ROOT / "agents"
PACKAGE_DATASETS_DIR = PACKAGE_ROOT / "datasets"


# --- Helper functions for interactive prompts (unchanged) ---

def _prompt_for_file(
    console: Console,
    user_dir: Path,
    package_dir: Path,
    extension: str,
    prompt_title: str,
) -> Path:
    """
    Generic helper to find files in both user and package directories and prompt for a selection.
    User files take priority over package files with the same name.
    """
    console.print(f"[bold]Select {prompt_title}:[/bold]")
    
    found_files = []
    seen_filenames = set()

    if user_dir.exists():
        for file_path in sorted(list(user_dir.glob(f"**/*{extension}"))):
            if file_path.name not in seen_filenames:
                found_files.append({"path": file_path, "label": "User"})
                seen_filenames.add(file_path.name)

    if package_dir.exists():
        for file_path in sorted(list(package_dir.glob(f"**/*{extension}"))):
            if file_path.name not in seen_filenames:
                found_files.append({"path": file_path, "label": "Package"})
                seen_filenames.add(file_path.name)

    if not found_files:
        console.print(f"[bold red]No '{extension}' files found in your user directory ({user_dir}) or the package directory ({package_dir}).[/bold red]")
        raise typer.Exit(1)
        
    for i, file_info in enumerate(found_files, 1):
        console.print(f"  [cyan]{i}[/cyan]: {file_info['path'].name} [yellow]({file_info['label']})[/yellow]")

    choice_str = Prompt.ask("Enter the number of your choice", choices=[str(i) for i in range(1, len(found_files) + 1)])
    return found_files[int(choice_str) - 1]['path']

def _prompt_for_driver(console: Console, system: AgentSystem) -> str:
    """Prompts the user to select a driver agent from the loaded system."""
    console.print("[bold]Select a driver agent:[/bold]")
    agents = list(system.agents.keys())
    driver = Prompt.ask("Enter the name of the driver agent", choices=agents, default=agents[0])
    return driver


# --- Typer App and Context (unchanged) ---

run_app = typer.Typer(
    name="run",
    help="Run an agent system. Prompts for configuration if not provided via flags.",
    no_args_is_help=True,
)

class AppContext:
    def __init__(self):
        self.console = Console()
        self.agent_system: AgentSystem | None = None
        self.driver_agent_name: str | None = None
        self.roster_instructions: str | None = None
        self.analysis_context: str | None = None
        self.sandbox_manager: SandboxManager | None = None
        self.llm_client: object | None = None
        self.initial_history: List[dict] | None = None

@run_app.callback(invoke_without_command=True)
def main_run_callback(
    ctx: typer.Context,
    blueprint: Path = typer.Option(None, "--blueprint", "-bp", help="Path to the agent system JSON blueprint.", readable=True),
    driver_agent: str = typer.Option(None, "--driver-agent", "-d", help="Name of the agent to start with."),
    dataset: Path = typer.Option(None, "--dataset", "-ds", help="Path to the dataset file (.h5ad).", readable=True),
    resources_dir: Path = typer.Option(None, "--resources", help="Path to a directory of resource files to mount.", exists=True, file_okay=False),
    llm_backend: str = typer.Option("chatgpt", "--llm", help="LLM backend to use.", case_sensitive=False),
    ollama_host: str = typer.Option("http://localhost:11434", "--ollama-host", help="Base URL for Ollama backend."),
    sandbox: str = typer.Option(None, "--sandbox", help="Sandbox backend to use: 'docker', 'singularity', or 'singularity-exec'."), # <-- Changed default
    force_refresh: bool = typer.Option(False, "--force-refresh", help="Force refresh/rebuild of the sandbox environment."),
):
    load_dotenv(dotenv_path=ENV_FILE)
    app_context = AppContext()
    console = app_context.console
    ctx.obj = app_context

    # Steps 1, 2, and 3 are unchanged
    if blueprint is None:
        blueprint = _prompt_for_file(console, DEFAULT_AGENT_DIR, PACKAGE_AGENTS_DIR, ".json", "Agent System Blueprint")
    app_context.agent_system = AgentSystem.load_from_json(str(blueprint))
    
    if driver_agent is None:
        driver_agent = _prompt_for_driver(console, app_context.agent_system)
    if driver_agent not in app_context.agent_system.agents:
        raise typer.BadParameter(f"Driver agent '{driver_agent}' not found in blueprint.")
    app_context.driver_agent_name = driver_agent
    app_context.roster_instructions = app_context.agent_system.get_instructions()
    
    if dataset is None:
        dataset = _prompt_for_file(console, get_datasets_dir(), PACKAGE_DATASETS_DIR, ".h5ad", "Dataset")

    # --- Step 4. Configure Sandbox (Corrected Logic) ---
    # Prompt for sandbox if not provided as a flag
    if sandbox is None:
        sandbox = Prompt.ask(
            "Choose a sandbox backend",
            choices=["docker", "singularity", "singularity-exec"],
            default="docker"
        )
        
    console.print(f"[cyan]Initializing sandbox backend: {sandbox}[/cyan]")
    script_dir = Path(__file__).resolve().parent
    
    manager_class = None
    if sandbox == "docker":
        manager_class, _, _, _, _ = init_docker(script_dir, subprocess, console, force_refresh=force_refresh)
    elif sandbox == "singularity":
        manager_class, _, _, _, _ = init_singularity(script_dir, subprocess, console, force_refresh=force_refresh)
    elif sandbox == "singularity-exec":
        SANDBOX_DATA_PATH = "/workspace/dataset.h5ad"
        manager_class, _, _, _, _ = init_singularity_exec(script_dir, SANDBOX_DATA_PATH, subprocess, console, force_refresh=force_refresh)
    else:
        raise typer.BadParameter(f"Unknown sandbox type '{sandbox}'. Supported types are 'docker', 'singularity', 'singularity-exec'.")

    app_context.sandbox_manager = manager_class()

    # Step 5 and 6 are unchanged
    console.print(f"[cyan]Initializing LLM backend: {llm_backend}[/cyan]")
    if llm_backend == "chatgpt":
        from openai import OpenAI
        app_context.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif llm_backend == "ollama":
        from olaf.core.ollama_wrapper import OllamaClient as OpenAI
        app_context.llm_client = OpenAI(host=ollama_host)
    else:
        raise typer.BadParameter(f"Unknown LLM backend '{llm_backend}'.")

    resources = collect_resources(console, resources_dir) if resources_dir else []
    app_context.analysis_context = textwrap.dedent(f"Dataset path: **{dataset.name}**\n...")
    driver = app_context.agent_system.get_agent(driver_agent)
    system_prompt = (app_context.roster_instructions + "\n\n" + driver.get_full_prompt() + "\n\n" + app_context.analysis_context)
    app_context.initial_history = [{"role": "system", "content": system_prompt}]


# --- Subcommands (interactive, auto) are unchanged ---

@run_app.command("interactive")
def run_interactive(ctx: typer.Context):
    """Run the agent system in a manual, interactive chat session."""
    context: AppContext = ctx.obj
    context.console.print("\n[bold blue]🚀 Starting Interactive Mode...[/bold blue]")
    
    history = context.initial_history[:]
    history.append({"role": "user", "content": "Beginning interactive session. What is the plan?"})
    
    run_agent_session(
        console=context.console,
        agent_system=cast(AgentSystem, context.agent_system),
        driver_agent=cast(AgentSystem, context.agent_system).get_agent(cast(str, context.driver_agent_name)),
        roster_instructions=cast(str, context.roster_instructions),
        analysis_context=cast(str, context.analysis_context),
        llm_client=cast(object, context.llm_client),
        sandbox_manager=cast(SandboxManager, context.sandbox_manager),
        history=history,
        is_auto=False
    )

@run_app.command("auto")
def run_auto(
    ctx: typer.Context,
    prompt: str = typer.Option(None, "--prompt", "-p", help="Initial prompt for the auto run."),
    turns: int = typer.Option(3, "--turns", "-t", help="Number of turns to run automatically."),
):
    """Run the agent system automatically for a set number of turns."""
    context: AppContext = ctx.obj
    
    if prompt is None:
        prompt = Prompt.ask("Enter the initial prompt for the automated run", default="Analyze this dataset.")

    context.console.print(f"\n[bold green]🚀 Starting Automated Mode for {turns} turns...[/bold green]")
    
    history = context.initial_history[:]
    history.append({"role": "user", "content": prompt})
    
    run_agent_session(
        console=context.console,
        agent_system=cast(AgentSystem, context.agent_system),
        driver_agent=cast(AgentSystem, context.agent_system).get_agent(cast(str, context.driver_agent_name)),
        roster_instructions=cast(str, context.roster_instructions),
        analysis_context=cast(str, context.analysis_context),
        llm_client=cast(object, context.llm_client),
        sandbox_manager=cast(SandboxManager, context.sandbox_manager),
        history=history,
        is_auto=True,
        max_turns=turns
    )