# cli/olaf/src/olaf/cli/datasets.py

import os
import re
import json
import math
import shlex
import sys
from pathlib import Path

import cellxgene_census
from platformdirs import PlatformDirs

try:
    from rich.console import Console
    from rich.table import Table
    from rich.pretty import pprint
    from rich.prompt import Prompt
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    # Define simple fallback classes if rich is not installed
    def pprint(obj): print(obj)
    class Console:
        def print(self, *args, **kwargs): print(*args)
    class Table:
        def __init__(self, title=""):
            self._title = title
            self._rows = []
            self._columns = []
        def add_column(self, header, **kwargs):
            self._columns.append(header)
        def add_row(self, *items):
            if len(items) != len(self._columns):
                raise ValueError("Number of items in row does not match number of columns")
            self._rows.append(items)
        def print_table(self, console):
            console.print(self._title)
            if not self._columns:
                return
            col_widths = [len(h) for h in self._columns]
            for row in self._rows:
                for i, item in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(item)))
            header_line = "  ".join(f"{h:<{w}}" for h, w in zip(self._columns, col_widths))
            separator = "-" * len(header_line)
            console.print(header_line)
            console.print(separator)
            for row in self._rows:
                row_line = "  ".join(f"{str(item):<{w}}" for item, w in zip(row, col_widths))
                console.print(row_line)
    class Prompt:
        @staticmethod
        def ask(prompt, choices=None, default=None):
            p_text = f"{prompt} "
            if choices:
                p_text += f"({'/'.join(choices)}) "
            if default:
                p_text += f"[{default}] "
            return input(p_text).strip()

# --- Path Configuration ---
APP_NAME = "olaf"
APP_AUTHOR = "OpenTechBio"
dirs = PlatformDirs(APP_NAME, APP_AUTHOR)

OLAF_HOME = Path(os.environ.get("OLAF_HOME", dirs.user_data_dir)).expanduser()
DEFAULT_DATASETS_DIR = OLAF_HOME / "datasets"

def get_datasets_dir() -> Path:
    """
    Returns the path to the datasets directory, creating it if it doesn't exist.
    """
    DEFAULT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATASETS_DIR

# --- Helper Functions ---
def sanitize_filename(name: str) -> str:
    """Removes invalid characters and replaces spaces for use in filenames."""
    name = re.sub(r'[^\w\-.]+', '_', name)
    return re.sub(r'_+', '_', name).strip('_').lower()

# --- Core Data Fetching and Download Functions ---

def get_census_versions_data():
    """Fetches available CELLxGENE Census versions data."""
    try:
        census_versions = cellxgene_census.get_census_version_directory()
        versions_list = []
        sorted_versions = sorted(
            census_versions.keys(),
            key=lambda v: ('0' if v == 'stable' else '1' if v == 'latest' else '2') + v,
            reverse=True
        )
        for version in sorted_versions:
            desc = census_versions[version]
            versions_list.append({
                "version": version,
                "description": desc.get('description', desc.get('uri', 'N/A')),
                "release_date": desc.get("release_date", "N/A")
            })
        return versions_list
    except Exception as e:
        raise RuntimeError(f"Error listing versions: {e}")

def fetch_source_datasets_data(census_version: str):
    """Fetches source datasets DataFrame for a specific Census version."""
    console = Console()
    console.print(f"Fetching source datasets info for Census version: [cyan]{census_version}[/cyan]...")
    try:
        with cellxgene_census.open_soma(census_version=census_version) as census:
            datasets_df = census["census_info"]["datasets"].read().concat().to_pandas()
            if datasets_df.empty:
                console.print(f"No source dataset information found for version {census_version}.")
            return datasets_df
    except Exception as e:
        raise RuntimeError(f"Error fetching datasets for version {census_version}: {e}")

def get_dataset_metadata_data(census_version: str, dataset_id: str):
    """Fetches metadata dictionary for a specific source dataset."""
    console = Console()
    console.print(f"Fetching metadata for [cyan]{dataset_id}[/cyan] in Census version: [cyan]{census_version}[/cyan]...")
    try:
        datasets_df = fetch_source_datasets_data(census_version)
        if datasets_df is None or datasets_df.empty:
             raise ValueError(f"Could not retrieve datasets for version {census_version}.")
        
        dataset_metadata = datasets_df[datasets_df['dataset_id'] == dataset_id]
        if dataset_metadata.empty:
            raise ValueError(f"Dataset ID '{dataset_id}' not found in Census version '{census_version}'.")
        return dataset_metadata.iloc[0].to_dict()
    except Exception as e:
        raise RuntimeError(f"Error fetching metadata for {dataset_id}: {e}")

def download_dataset(console: Console, census_version: str, dataset_id: str):
    """Downloads H5AD file and saves metadata JSON for a dataset."""
    try:
        # 1. Get target directory using the new function
        target_dir = get_datasets_dir()
        console.print(f"Target directory: [blue]{target_dir}[/blue]")

        # 2. Fetch metadata
        metadata = get_dataset_metadata_data(census_version, dataset_id)
        dataset_title = metadata.get('dataset_title', f'dataset_{dataset_id}')
        base_filename = sanitize_filename(dataset_title) or f"dataset_{dataset_id}"
        
        h5ad_filepath = target_dir / f"{base_filename}.h5ad"
        json_filepath = target_dir / f"{base_filename}.json"

        console.print(f"Preparing to download dataset [green]{dataset_title}[/green]...")
        if h5ad_filepath.exists() or json_filepath.exists():
            console.print("[yellow]Warning: Output file(s) already exist. Skipping download.[/yellow]")
            return

        # 3. Download H5AD
        console.print(f"Downloading H5AD to [blue]{h5ad_filepath}[/blue]...")
        cellxgene_census.download_source_h5ad(dataset_id, to_path=str(h5ad_filepath), census_version=census_version)
        console.print("[bold green]H5AD Download complete.[/bold green]")

        # 4. Save Metadata JSON
        console.print(f"Saving metadata to [blue]{json_filepath}[/blue]...")
        import numpy as np
        def convert_types(obj):
            if isinstance(obj, np.generic): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.void): return None
            return obj
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, default=convert_types, ensure_ascii=False)
        console.print("[bold green]Metadata JSON saved successfully.[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Download failed:[/bold red] {e}")
        sys.exit(1)

# --- Display and Interaction Functions ---

def display_versions_list(console: Console):
    """Displays available versions."""
    try:
        versions_data = get_census_versions_data()
        if not versions_data:
             console.print("[yellow]No Census versions found.[/yellow]")
             return

        table = Table(title="Available CELLxGENE Census Versions")
        table.add_column("Version Tag", style="cyan")
        table.add_column("Release Date", style="green")
        table.add_column("Description", style="magenta")

        for v_data in versions_data:
            table.add_row(v_data["version"], v_data["release_date"], v_data["description"])

        if HAS_RICH:
            console.print(table)
        else:
            table.print_table(console)
    except Exception as e:
        console.print(f"[bold red]Error displaying versions:[/bold red] {e}")

def display_paginated_datasets(console: Console, census_version: str, limit: int = None, page_size: int = 5):
    """Fetches and displays datasets with pagination."""
    try:
        datasets_df = fetch_source_datasets_data(census_version)
        if datasets_df is None or datasets_df.empty:
            return

        df_view = datasets_df.head(limit) if limit and limit > 0 else datasets_df
        total_items_in_view = len(df_view)
        if total_items_in_view == 0:
            console.print(f"No datasets found for version {census_version}.")
            return

        total_pages = math.ceil(total_items_in_view / page_size)
        current_page = 1

        while True:
            start_index = (current_page - 1) * page_size
            end_index = start_index + page_size
            page_df = df_view.iloc[start_index:end_index]

            if page_df.empty:
                break

            range_end = min(end_index, total_items_in_view)
            table = Table(title=f"Source Datasets in Census {census_version} (Showing {start_index+1}-{range_end} of {total_items_in_view})")
            table.add_column("Dataset ID", style="cyan", no_wrap=True)
            table.add_column("Dataset Title", style="green", overflow="fold")
            table.add_column("Cell Count", style="yellow", justify="right")

            for _, row in page_df.iterrows():
                 cell_count_str = f"{int(row.get('cell_count', 0)):,}" if row.get('cell_count') else 'N/A'
                 table.add_row(row.get('dataset_id', 'N/A'), row.get('dataset_title', 'N/A'), cell_count_str)
            
            console.print(f"\n--- Page {current_page} of {total_pages} ---")
            if HAS_RICH:
                console.print(table)
            else:
                table.print_table(console)

            if total_pages <= 1: break

            prompt_text = "[P]revious, [N]ext, [Q]uit?"
            action = Prompt.ask(prompt_text, default="N" if current_page < total_pages else "Q").upper()

            if action == "N" and current_page < total_pages: current_page += 1
            elif action == "P" and current_page > 1: current_page -= 1
            elif action == "Q": break
            else: console.print("[yellow]Invalid choice.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error displaying datasets:[/bold red] {e}")

def display_dataset_metadata(console: Console, census_version: str, dataset_id: str):
     """Displays metadata for a specific dataset."""
     try:
         metadata_dict = get_dataset_metadata_data(census_version, dataset_id)
         console.print(f"\nMetadata for Dataset: [bold green]{dataset_id}[/bold green]")
         pprint(metadata_dict)
     except Exception as e:
         console.print(f"[bold red]Error displaying metadata:[/bold red] {e}")

def print_interactive_help(console: Console):
     """Prints help message for interactive mode."""
     console.print("\n[bold cyan]Available Commands:[/bold cyan]")
     console.print("  [green]list_versions[/green]                    List available CELLxGENE Census versions.")
     console.print("  [green]list_datasets[/green] <version> [limit]  List source datasets (paginated).")
     console.print("  [green]show_metadata[/green] <version> <dataset_id> Show metadata for a specific dataset.")
     console.print("  [green]download[/green] <version> <dataset_id>      Download dataset H5AD and metadata JSON.")
     console.print("  [green]help[/green]                         Show this help message.")
     console.print("  [green]exit[/green]                         Exit the interactive browser.")
     console.print("\nExample: [yellow]download stable <some_dataset_id>[/yellow]")

def interactive_loop():
    """Runs the interactive command loop."""
    console = Console()
    console.print("[bold blue]Welcome to the Interactive CZI CELLxGENE Census Browser![/bold blue]")
    print_interactive_help(console)

    while True:
        try:
            raw_command = Prompt.ask("\nEnter command ('help' or 'exit')")
            if not raw_command: continue

            command_parts = shlex.split(raw_command)
            if not command_parts: continue

            command = command_parts[0].lower()
            args = command_parts[1:]

            if command == "exit": break
            elif command == "help": print_interactive_help(console)
            elif command == "list_versions":
                if not args: display_versions_list(console)
                else: console.print("[yellow]Usage: list_versions[/yellow]")
            elif command == "list_datasets":
                if not args:
                    console.print("[yellow]Usage: list_datasets <version> [limit][/yellow]")
                    continue
                version = args[0]
                limit = int(args[1]) if len(args) > 1 else None
                display_paginated_datasets(console, version, limit=limit, page_size=5)
            elif command == "show_metadata":
                if len(args) < 2:
                    console.print("[yellow]Usage: show_metadata <version> <dataset_id>[/yellow]")
                    continue
                display_dataset_metadata(console, args[0], args[1])
            elif command == "download":
                if len(args) < 2:
                    console.print("[yellow]Usage: download <version> <dataset_id>[/yellow]")
                    continue
                download_dataset(console, args[0], args[1])
            else:
                console.print(f"[red]Unknown command: '{command}'. Type 'help' for options.[/red]")
        except EOFError:
             console.print("\n[yellow]EOF detected. Exiting.[/yellow]")
             break
        except KeyboardInterrupt:
             console.print("\n[yellow]Interrupted by user. Type 'exit' to quit.[/yellow]")
        except Exception as e:
             console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")

    console.print("[bold blue]Exiting browser. Goodbye![/bold blue]")