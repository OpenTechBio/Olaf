# olaf/cli/config_cli.py
import re
import typer
from rich.console import Console

# Import the centralized ENV_FILE path
from olaf.config import ENV_FILE

config_app = typer.Typer(
    name="config",
    help="Manage OLAF configuration and API keys.",
    no_args_is_help=True
)

console = Console()

@config_app.command("set-openai-key")
def set_api_key(
    api_key: str = typer.Argument(..., help="Your OpenAI API key (e.g., 'sk-...')")
):
    """
    Saves your OpenAI API key to the OLAF environment file.
    """
    if not api_key.startswith("sk-"):
        console.print("[yellow]Warning: Key does not look like a standard OpenAI API key (should start with 'sk-').[/yellow]")

    # Ensure the .env file exists
    if not ENV_FILE.exists():
        ENV_FILE.touch()

    content = ENV_FILE.read_text()
    key_to_set = f'OPENAI_API_KEY="{api_key}"'

    # Use regex to safely replace the key if it already exists
    if re.search(r"^OPENAI_API_KEY=.*$", content, flags=re.MULTILINE):
        new_content = re.sub(r"^OPENAI_API_KEY=.*$", key_to_set, content, flags=re.MULTILINE)
    else:
        new_content = content + f"\n{key_to_set}\n"

    ENV_FILE.write_text(new_content.strip())
    console.print(f"[bold green]✅ OpenAI API key has been set successfully in:[/bold green] {ENV_FILE}")