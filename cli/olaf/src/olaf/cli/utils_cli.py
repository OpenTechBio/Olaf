# olaf/cli/utils_cli.py
import json
import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# Import from the central config to know where chat logs are stored by default
from olaf.config import OLAF_HOME

utils_app = typer.Typer(
    name="utils",
    help="Utility commands for managing OLAF artifacts like chat logs.",
    no_args_is_help=True
)

console = Console()
LOG_DIR = OLAF_HOME / "runs" / "chat_logs"

def _convert_history_to_notebook(history_path: Path, output_path: Path):
    """
    Parses an OLAF chat log and converts it into a Jupyter Notebook (.ipynb).
    """
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error: Could not read or parse the history file at {history_path}.[/bold red]\n{e}")
        raise typer.Exit(1)

    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11" # This can be made more dynamic if needed
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # This regex specifically finds Python code blocks
    code_block_re = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

    for message in history:
        role = message.get("role")
        content = message.get("content", "")
        
        # We are primarily interested in the agent's responses
        if role and "assistant" in role:
            # Split the content by code blocks to interleave markdown and code
            parts = code_block_re.split(content)
            
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                
                # Odd-indexed parts are the code blocks captured by the regex
                if i % 2 == 1:
                    cell = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": part
                    }
                    notebook["cells"].append(cell)
                # Even-indexed parts are the explanatory text outside the code blocks
                else:
                    cell = {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": part
                    }
                    notebook["cells"].append(cell)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        console.print(f"[bold green]✓ Successfully converted chat log to notebook:[/bold green] {output_path}")
    except Exception as e:
        console.print(f"[bold red]Error writing notebook file: {e}[/bold red]")
        raise typer.Exit(1)

@utils_app.command("convert-to-notebook")
def convert_to_notebook(
    chat_log: Path = typer.Argument(
        ...,
        help="Path to the interactive chat log JSON file to convert.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the output .ipynb file. Defaults to the same name as the input file.",
        writable=True,
        resolve_path=True,
    ),
):
    """
    Converts an OLAF interactive chat log into an executable Jupyter Notebook.
    
    This command parses the JSON log file, extracts all Python code blocks generated
    by the assistant, and arranges them into code cells. The explanatory text
    between code blocks is converted into markdown cells, creating a clean,
    reproducible protocol of the agent session.
    """
    if not chat_log.name.startswith("interactive_chat_"):
        console.print(f"[yellow]Warning: The input file '{chat_log.name}' does not look like a standard OLAF chat log.[/yellow]")

    output_path = output
    if output_path is None:
        # Default to the same name as the input file, but with an .ipynb extension
        output_path = chat_log.with_suffix(".ipynb")
    
    # Ensure the output path has the correct extension
    if output_path.suffix != ".ipynb":
        output_path = output_path.with_suffix(".ipynb")

    console.print(f"Converting [cyan]{chat_log.name}[/cyan] to Jupyter Notebook...")
    _convert_history_to_notebook(chat_log, output_path)
