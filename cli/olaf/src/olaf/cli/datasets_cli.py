# cli/olaf/src/olaf/cli/datasets_cli.py

import typer
from typing_extensions import Annotated

# Import the logic functions from our other file
import olaf.datasets.czi_datasets as datasets

try:
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Create a Typer app for the "datasets" subcommand group
datasets_app = typer.Typer(
    name="datasets",
    help="Browse and download datasets from the CZI CELLxGENE Census. Defaults to interactive mode.",
    no_args_is_help=False  # Allows our callback to run
)

@datasets_app.callback(invoke_without_command=True)
def datasets_main(ctx: typer.Context):
    """
    If no subcommand is specified, enter interactive mode.
    """
    if ctx.invoked_subcommand is None:
        console = datasets.Console()
        console.print("No subcommand given. Starting interactive CZI Census browser...")
        # Ensure dependencies for interactive mode are checked
        try:
            import numpy
        except ImportError:
            console.print("[bold red]Error: 'numpy' is required. Please 'pip install numpy'.[/bold red]")
            raise typer.Exit(1)
        datasets.interactive_loop()

@datasets_app.command("list-versions")
def list_versions():
    """List available CELLxGENE Census versions."""
    datasets.display_versions_list(datasets.Console())

@datasets_app.command("list-datasets")
def list_datasets(
    version: Annotated[str, typer.Option(help='Census version tag (e.g., "stable", "latest").')],
    limit: Annotated[int, typer.Option(help="Max number of datasets to paginate through.")] = None,
    page_size: Annotated[int, typer.Option(help="Number of datasets per page.")] = 5,
):
    """List source datasets within a specific Census version."""
    datasets.display_paginated_datasets(datasets.Console(), version, limit, page_size)

@datasets_app.command("show-metadata")
def show_metadata(
    version: Annotated[str, typer.Option(help='Census version tag (e.g., "stable").')],
    dataset_id: Annotated[str, typer.Option(help="The dataset_id to view.")],
):
    """Show all metadata for a specific source dataset."""
    datasets.display_dataset_metadata(datasets.Console(), version, dataset_id)

@datasets_app.command("download")
def download(
    version: Annotated[str, typer.Option(help='Census version tag (e.g., "stable").')],
    dataset_id: Annotated[str, typer.Option(help="The dataset_id to download.")],
):
    """Download a dataset's H5AD file and metadata JSON."""
    console = datasets.Console()
    try:
        import numpy
    except ImportError:
        console.print("[bold red]Error: 'numpy' is required for this command. Please 'pip install numpy'.[/bold red]")
        raise typer.Exit(1)
    datasets.download_dataset(console, version, dataset_id)