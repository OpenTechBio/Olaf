# cli/olaf/src/olaf/__main__.py

import typer

# Import the app for the 'create-system' command
from .create_agent_cli import app as create_system_app

# Import the app for the new 'datasets' command
from .datasets_cli import datasets_app

# Main OLAF application
app = typer.Typer(
    name="olaf",
    help="OLAF: The Open-source Language Agent Framework",
    no_args_is_help=True
)

# Register the command groups
app.add_typer(create_system_app, name="create-system")
app.add_typer(datasets_app, name="datasets")


def main():
    app()

if __name__ == "__main__":
    main()