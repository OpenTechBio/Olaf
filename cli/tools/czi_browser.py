#!/usr/bin/env python
import argparse
import cellxgene_census
import sys
import math
import shlex  # For parsing interactive commands safely
import os     # For path operations and directory creation
import json   # For saving metadata
import re     # For sanitizing filenames

try:
    from rich.console import Console
    from rich.table import Table
    from rich.pretty import pprint
    from rich.prompt import Prompt # For interactive prompts
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    # Simple print/input fallback if rich is not installed
    def pprint(obj): print(obj)
    class Console:
        def print(self, *args, **kwargs): print(*args)
    class Table:
        # Basic fallback Table class
        def __init__(self, title=""):
            self._title = title
            self._rows = []
            self._columns = []
            self._styles = {} # Dummy style storage
        def add_column(self, header, style=""):
            self._columns.append(header)
            self._styles[header] = style # Store style info even if unused
        def add_row(self, *items):
             # Ensure row has same number of items as columns
             if len(items) != len(self._columns):
                 raise ValueError("Number of items in row does not match number of columns")
             self._rows.append(items)
        def __rich_console__(self, console, options): # Dummy method for rich compatibility
             # Basic text rendering for fallback
             yield self._title
             yield "\t".join(self._columns)
             for row in self._rows: yield "\t".join(map(str, row))
        def print_table(self, console): # Custom print method if rich not available
             console.print(self._title)
             if self._columns: # Only print header/rows if columns exist
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
                choices_str = '/'.join(choices)
                p_text += f"({choices_str}) "
            if default:
                p_text += f"[{default}] "
            return input(p_text).strip()

# --- Helper Functions ---

def sanitize_filename(name):
    """Removes invalid characters and replaces spaces for use in filenames."""
    # Remove characters that are not alphanumeric, underscore, or hyphen
    name = re.sub(r'[^\w\-]+', '_', name)
    # Replace multiple underscores with a single one
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Convert to lowercase
    return name.lower()

def ensure_datasets_dir_exists(base_dir="../datasets"):
    """Checks if the target directory exists and creates it if not."""
    # Get the absolute path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.abspath(os.path.join(script_dir, base_dir))

    if not os.path.exists(target_dir):
        print(f"Creating target directory: {target_dir}")
        try:
            os.makedirs(target_dir)
        except OSError as e:
            raise OSError(f"Failed to create directory {target_dir}: {e}")
    elif not os.path.isdir(target_dir):
        raise NotADirectoryError(f"Target path {target_dir} exists but is not a directory.")
    return target_dir


# --- Core Data Fetching Functions ---

def get_census_versions_data():
    """Fetches available CELLxGENE Census versions data."""
    try:
        census_versions = cellxgene_census.get_census_version_directory()
        versions_list = []
        # Prioritize 'stable', then 'latest', then sort others reverse chronologically
        sorted_versions = sorted(
            census_versions.keys(),
            key=lambda v: ('0' if v == 'stable' else '1' if v == 'latest' else '2') + v,
            reverse=True # Puts stable/latest effectively first, then sorts dates reverse
        )

        for version in sorted_versions:
            description = census_versions[version]
            release_date = "N/A"
            try:
                # Avoid fetching description again if already present
                release_date = description.get("release_date")
                if not release_date:
                     details = cellxgene_census.get_census_version_description(version)
                     release_date = details.get("release_date", "N/A")
            except Exception:
                pass # Ignore if details can't be fetched
            versions_list.append({
                "version": version,
                "description": description.get('description', description.get('uri', 'N/A')),
                "release_date": release_date
            })
        return versions_list
    except Exception as e:
        raise RuntimeError(f"Error listing versions: {e}")

def fetch_source_datasets_data(census_version):
    """Fetches source datasets DataFrame for a specific Census version."""
    console = Console()
    console.print(f"Fetching source datasets info for Census version: [cyan]{census_version}[/cyan]...")
    try:
        # Check if version is valid before opening (optional, but good practice)
        available_versions = cellxgene_census.get_census_version_directory()
        if census_version not in available_versions:
             console.print(f"[bold red]Error:[/bold red] Census version '{census_version}' not found.")
             # Attempt to list versions to help user
             try:
                 versions_data = get_census_versions_data()
                 console.print("Available versions:")
                 for v in versions_data:
                     console.print(f"  - {v['version']} ({v.get('release_date', 'N/A')})")
             except Exception:
                 console.print("(Could not fetch list of available versions)")
             return None

        # Inform user about specific date mapping if using 'stable'/'latest'
        try:
             version_description = cellxgene_census.get_census_version_description(census_version)
             actual_version = version_description.get("release_date", census_version)
             if census_version in ["stable", "latest"] and actual_version != census_version:
                 console.print(f"The \"{census_version}\" release is currently [bold green]{actual_version}[/bold green]. Specify 'census_version=\"{actual_version}\"' in future calls to open_soma() to ensure data consistency.")
        except Exception:
             console.print(f"[yellow]Warning: Could not verify exact date for '{census_version}'. Proceeding...[/yellow]")


        with cellxgene_census.open_soma(census_version=census_version) as census:
            if "census_info" not in census or "datasets" not in census["census_info"]:
                 raise RuntimeError("Census object structure unexpected: 'census_info' or 'datasets' missing.")

            datasets_df = census["census_info"]["datasets"].read().concat().to_pandas()
            if datasets_df.empty:
                console.print(f"No source dataset information found for version {census_version}.")
                return datasets_df # Return empty DataFrame
            return datasets_df
    except Exception as e:
        raise RuntimeError(f"Error fetching datasets for version {census_version}: {e}")


def get_dataset_metadata_data(census_version, dataset_id):
    """Fetches metadata dictionary for a specific source dataset."""
    console = Console()
    console.print(f"Fetching metadata for dataset [cyan]{dataset_id}[/cyan] in Census version: [cyan]{census_version}[/cyan]...")
    try:
        # Reuse fetch_source_datasets_data which includes version check
        datasets_df = fetch_source_datasets_data(census_version)
        if datasets_df is None: # Check if fetch failed (e.g., invalid version)
             raise ValueError(f"Could not retrieve dataset list for version {census_version}.")
        if datasets_df.empty: # Check if fetch succeeded but returned empty
             raise ValueError(f"No datasets found for version {census_version}, cannot fetch metadata.")

        dataset_metadata = datasets_df[datasets_df['dataset_id'] == dataset_id]

        if dataset_metadata.empty:
            raise ValueError(f"Dataset ID '{dataset_id}' not found in Census version '{census_version}'.")

        return dataset_metadata.iloc[0].to_dict()
    except Exception as e:
        # Catch specific errors if needed, otherwise re-raise or wrap
        raise RuntimeError(f"Error fetching metadata for dataset {dataset_id}: {e}")


# --- Download Function ---

def download_dataset(console, census_version, dataset_id):
    """Downloads the H5AD file and saves metadata JSON for a dataset."""
    try:
        # 1. Ensure target directory exists
        target_dir = ensure_datasets_dir_exists()
        console.print(f"Target directory: [blue]{target_dir}[/blue]")

        # 2. Fetch metadata first to get the title and verify dataset exists
        metadata = get_dataset_metadata_data(census_version, dataset_id) # Handles errors
        dataset_title = metadata.get('dataset_title', f'dataset_{dataset_id}') # Fallback title

        # 3. Generate filenames
        base_filename = sanitize_filename(dataset_title)
        if not base_filename: # Handle cases where title sanitizes to empty string
            base_filename = f"dataset_{dataset_id}"
        h5ad_filename = f"{base_filename}.h5ad"
        json_filename = f"{base_filename}.json"
        h5ad_filepath = os.path.join(target_dir, h5ad_filename)
        json_filepath = os.path.join(target_dir, json_filename)

        console.print(f"Preparing to download dataset:")
        console.print(f"  ID:      [cyan]{dataset_id}[/cyan]")
        console.print(f"  Title:   [green]{dataset_title}[/green]")
        console.print(f"  Version: [cyan]{census_version}[/cyan]")
        console.print(f"  Output H5AD: [blue]{h5ad_filepath}[/blue]")
        console.print(f"  Output JSON: [blue]{json_filepath}[/blue]")

        # Check if files already exist (optional, add overwrite flag later if needed)
        if os.path.exists(h5ad_filepath) or os.path.exists(json_filepath):
             console.print("[yellow]Warning: One or both output files already exist. Skipping download.[/yellow]")
             console.print("[yellow]         (Delete existing files or implement an --overwrite flag to replace.)[/yellow]")
             return # Or prompt user, or add an overwrite flag

        # 4. Download H5AD
        console.print(f"Downloading H5AD file...")
        cellxgene_census.download_source_h5ad(
            dataset_id=dataset_id,
            to_path=h5ad_filepath,
            census_version=census_version
        )
        console.print("[bold green]H5AD Download complete.[/bold green]")

        # 5. Save Metadata JSON
        console.print("Saving metadata JSON file...")
        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                # Convert numpy types to standard Python types if necessary
                def convert_types(obj):
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                        np.int16, np.int32, np.int64, np.uint8,
                                        np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.ndarray,)): # Handle arrays if needed
                        return obj.tolist() # Or other representation
                    elif isinstance(obj, (np.bool_)):
                        return bool(obj)
                    elif isinstance(obj, (np.void)): # Handle complex types if they appear
                        return None # Or suitable representation
                    return obj

                # Import numpy locally for type checking if needed
                import numpy as np
                json.dump(metadata, f, indent=4, default=convert_types, ensure_ascii=False)
            console.print("[bold green]Metadata JSON saved successfully.[/bold green]")
        except Exception as json_e:
            console.print(f"[bold red]Error saving metadata JSON:[/bold red] {json_e}")
            # Decide if we should clean up the downloaded H5AD file
            # try:
            #     os.remove(h5ad_filepath)
            #     console.print(f"[yellow]Cleaned up partially downloaded H5AD file.[/yellow]")
            # except OSError:
            #     pass

    except (ValueError, RuntimeError, OSError, NotADirectoryError, Exception) as e:
        console.print(f"[bold red]Download failed:[/bold red] {e}")
        # Potentially add more specific error handling based on exception type

# --- Display and Interaction Functions ---

def display_versions_list(console):
    """Displays available versions."""
    try:
        versions_data = get_census_versions_data()
        if not versions_data:
             console.print("[yellow]No Census versions found.[/yellow]")
             return

        table = Table(title="Available CELLxGENE Census Versions")
        table.add_column("Version Tag", style="cyan", justify="right")
        table.add_column("Release Date", style="green")
        table.add_column("Description / URL", style="magenta")


        for v_data in versions_data:
            table.add_row(v_data["version"], v_data["release_date"], v_data["description"])

        if HAS_RICH:
            console.print(table)
        else:
            table.print_table(console) # Use fallback print
    except Exception as e:
        console.print(f"[bold red]Error displaying versions:[/bold red] {e}")


def display_paginated_datasets(console, census_version, limit=None, page_size=5):
    """Fetches and displays datasets with pagination."""
    try:
        datasets_df = fetch_source_datasets_data(census_version)
        if datasets_df is None: # Error handled in fetch
            return
        if datasets_df.empty: # Message handled in fetch
             return

        if limit is not None and limit > 0:
             datasets_df = datasets_df.head(limit)
             total_items_in_view = len(datasets_df) # Number we are actually paging through
             if total_items_in_view == 0:
                 console.print(f"No datasets found matching the criteria within the limit of {limit}.")
                 return
        else:
             total_items_in_view = len(datasets_df)
             limit = total_items_in_view # Set limit for display consistency

        if total_items_in_view == 0:
             console.print(f"No datasets found for version {census_version}.")
             return

        total_pages = math.ceil(total_items_in_view / page_size)
        current_page = 1

        while True:
            start_index = (current_page - 1) * page_size
            end_index = start_index + page_size
            page_df = datasets_df.iloc[start_index:end_index]

            if page_df.empty and current_page > 1: # Handle reaching end with partial page
                console.print("[yellow]No more datasets to display.[/yellow]")
                break
            elif page_df.empty: # Only happens if total_items_in_view was 0 initially
                 console.print("[yellow]No datasets to display.[/yellow]")
                 break

            range_end = min(end_index, total_items_in_view)
            table = Table(title=f"Source Datasets in Census {census_version} (Showing {start_index+1}-{range_end} of {total_items_in_view})")
            table.add_column("Dataset ID", style="cyan", no_wrap=True)
            table.add_column("Collection Name", style="magenta", overflow="fold")
            table.add_column("Dataset Title", style="green", overflow="fold")
            table.add_column("Cell Count", style="yellow", justify="right")

            for _, row in page_df.iterrows():
                 # Safely format cell_count, handling potential None or non-numeric types
                 cell_count = row.get('cell_count')
                 cell_count_str = 'N/A'
                 if cell_count is not None:
                     try:
                         cell_count_str = f"{int(cell_count):,}"
                     except (ValueError, TypeError):
                         cell_count_str = str(cell_count) # Fallback to string if not int-convertible

                 table.add_row(
                     row.get('dataset_id', 'N/A'),
                     row.get('collection_name', 'N/A'),
                     row.get('dataset_title', 'N/A'),
                     cell_count_str
                 )

            console.print(f"\n--- Page {current_page} of {total_pages} ---")
            if HAS_RICH:
                 console.print(table)
            else:
                 table.print_table(console)

            if total_pages <= 1:
                break # No more pages

            choices = []
            prompt_text = "Action"
            if current_page > 1: choices.append("P")
            if current_page < total_pages: choices.append("N")
            choices.append("Q")

            prompt_parts = []
            if "P" in choices: prompt_parts.append("[P]revious")
            if "N" in choices: prompt_parts.append("[N]ext")
            prompt_parts.append("[Q]uit listing")
            prompt_text = ", ".join(prompt_parts) + "?"


            default_action = "Q"
            if current_page < total_pages: default_action = "N"
            elif current_page > 1: default_action = "P"


            action = Prompt.ask(
                prompt_text,
                choices=choices,
                default=default_action
            ).upper()

            if action == "N" and current_page < total_pages:
                current_page += 1
            elif action == "P" and current_page > 1:
                current_page -= 1
            elif action == "Q":
                break
            else:
                console.print("[yellow]Invalid choice.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error displaying datasets:[/bold red] {e}")

def display_dataset_metadata(console, census_version, dataset_id):
     """Displays metadata for a specific dataset."""
     try:
         metadata_dict = get_dataset_metadata_data(census_version, dataset_id)
         console.print(f"\nMetadata for Dataset: [bold green]{dataset_id}[/bold green]")
         pprint(metadata_dict) # Use rich's pprint or fallback print
     except Exception as e:
         console.print(f"[bold red]Error displaying metadata:[/bold red] {e}")


def print_interactive_help(console):
     """Prints help message for interactive mode."""
     console.print("\n[bold cyan]Available Commands:[/bold cyan]")
     console.print("  [green]list_versions[/green]                    List available CELLxGENE Census versions.")
     console.print("  [green]list_datasets[/green] <version> [limit]  List source datasets (paginated).")
     console.print("                                     <version>: stable, latest, or YYYY-MM-DD")
     console.print("                                     [limit] (optional): Total number of datasets to fetch.")
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
            if HAS_RICH:
                 raw_command = Prompt.ask("\nEnter command (\'help\' or \'exit\')")
            else:
                 raw_command = input("\nEnter command ('help' or 'exit'): ").strip()

            if not raw_command:
                continue

            try:
                command_parts = shlex.split(raw_command)
            except ValueError as e:
                console.print(f"[red]Error parsing command (check quotes?): {e}[/red]")
                continue

            if not command_parts: continue

            command = command_parts[0].lower()
            args = command_parts[1:]

            if command == "exit":
                break
            elif command == "help":
                print_interactive_help(console)
            elif command == "list_versions":
                if len(args) == 0:
                     display_versions_list(console)
                else:
                     console.print("[yellow]Usage: list_versions[/yellow]")
            elif command == "list_datasets":
                version = args[0] if len(args) > 0 else None
                limit = None
                if len(args) > 1:
                    try:
                        limit = int(args[1])
                        if limit <= 0:
                             console.print("[red]Limit must be a positive integer.[/red]")
                             continue
                    except ValueError:
                        console.print(f"[red]Invalid limit '{args[1]}'. Must be an integer.[/red]")
                        continue
                if version:
                    display_paginated_datasets(console, version, limit=limit, page_size=5)
                else:
                    console.print("[yellow]Usage: list_datasets <version> [limit][/yellow]")
            elif command == "show_metadata":
                version = args[0] if len(args) > 0 else None
                dataset_id = args[1] if len(args) > 1 else None
                if version and dataset_id:
                    display_dataset_metadata(console, version, dataset_id)
                else:
                    console.print("[yellow]Usage: show_metadata <version> <dataset_id>[/yellow]")
            elif command == "download":
                version = args[0] if len(args) > 0 else None
                dataset_id = args[1] if len(args) > 1 else None
                if version and dataset_id:
                    download_dataset(console, version, dataset_id)
                else:
                    console.print("[yellow]Usage: download <version> <dataset_id>[/yellow]")
            else:
                console.print(f"[red]Unknown command: '{command}'. Type 'help' for options.[/red]")

        except EOFError:
             console.print("\n[yellow]EOF detected. Exiting.[/yellow]")
             break
        except KeyboardInterrupt:
             console.print("\n[yellow]Interrupted by user. Type 'exit' to quit.[/yellow]")
        except Exception as e:
             console.print(f"[bold red]An unexpected error occurred in the interactive loop:[/bold red] {e}")


    console.print("[bold blue]Exiting browser. Goodbye![/bold blue]")


def main():
    # Check if running interactively (no arguments other than script name)
    if len(sys.argv) == 1:
        interactive_loop()
        sys.exit(0)

    # --- Original argparse logic for non-interactive mode ---
    parser = argparse.ArgumentParser(
        description="CZI CELLxGENE Census Browser CLI. Run without arguments for interactive mode.",
        formatter_class=argparse.RawTextHelpFormatter # Keep help text formatting
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands (run without arguments for interactive mode)')

    # Subparser for listing census versions
    parser_list_versions = subparsers.add_parser('list-versions', help='List available CELLxGENE Census versions')
    parser_list_versions.set_defaults(func=lambda args: display_versions_list(Console()))

    # Subparser for listing datasets within a version
    parser_list_datasets = subparsers.add_parser('list-datasets', help='List source datasets within a specific Census version (paginated)')
    parser_list_datasets.add_argument('--version', required=True, help='Census version tag (e.g., "stable", "latest", "YYYY-MM-DD")')
    parser_list_datasets.add_argument('--limit', type=int, default=None, help='Maximum number of datasets to fetch and paginate through')
    parser_list_datasets.add_argument('--page-size', type=int, default=5, help='Number of datasets to show per page (default: 5)')
    parser_list_datasets.set_defaults(func=lambda args: display_paginated_datasets(Console(), args.version, args.limit, args.page_size))

    # Subparser for showing metadata for a specific dataset
    parser_show_metadata = subparsers.add_parser('show-metadata', help='Show metadata for a specific source dataset')
    parser_show_metadata.add_argument('--version', required=True, help='Census version tag')
    parser_show_metadata.add_argument('--dataset-id', required=True, help='The dataset_id')
    parser_show_metadata.set_defaults(func=lambda args: display_dataset_metadata(Console(), args.version, args.dataset_id))

    # Subparser for downloading a dataset
    parser_download = subparsers.add_parser('download', help='Download dataset H5AD and metadata JSON')
    parser_download.add_argument('--version', required=True, help='Census version tag')
    parser_download.add_argument('--dataset-id', required=True, help='The dataset_id to download')
    parser_download.set_defaults(func=lambda args: download_dataset(Console(), args.version, args.dataset_id))


    # Allow showing help if no subcommand is given when args are present
    if len(sys.argv) > 1 and sys.argv[1] not in ['list-versions', 'list-datasets', 'show-metadata', 'download', '-h', '--help']:
         args = parser.parse_args(sys.argv[1:2]) # Parse just the first potential command
    else:
         args = parser.parse_args()

    if hasattr(args, 'func'):
         try:
             args.func(args)
         except Exception as e:
             Console().print(f"[bold red]Command failed:[/bold red] {e}")
             sys.exit(1)
    else:
         if len(sys.argv) > 1:
             parser.print_help()


if __name__ == "__main__":
    # Need numpy for JSON conversion of metadata types
    try:
        import numpy as np
    except ImportError:
        print("Error: The 'numpy' package is required for saving metadata. Please install it (`pip install numpy`).")
        sys.exit(1)
    main()
