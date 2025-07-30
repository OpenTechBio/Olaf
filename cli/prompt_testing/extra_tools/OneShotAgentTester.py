import argparse
import os
import sys
import json
import re
import shlex
import time
from pathlib import Path
import subprocess # Still needed for docker cp (for dataset copy)
import base64 # For decoding image data from API
from datetime import datetime # For timestamp in filename

# --- Dependency Imports ---
try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv library not found. Please install it: pip install python-dotenv", file=sys.stderr)
    sys.exit(1)

try:
    from openai import OpenAI, APIError
except ImportError:
    print("Error: openai library not found. Please install it: pip install openai", file=sys.stderr)
    sys.exit(1)

try:
    import requests # For making HTTP requests to the FastAPI service
except ImportError:
    print("Error: requests library not found. Please install it: pip install requests", file=sys.stderr)
    sys.exit(1)


try:
    # Assumes benchmarking_sandbox_management.py is in a 'sandbox' subdirectory
    # We still need the manager for start/stop and the container name constant
    sandbox_dir = os.path.join(os.path.dirname(__file__), 'sandbox')
    sys.path.insert(0, sandbox_dir)
    from benchmarking_sandbox_management import SandboxManager, CONTAINER_NAME as SANDBOX_CONTAINER_NAME, API_PORT_HOST
except ImportError as e:
    print(f"Error: Could not import SandboxManager or constants from {sandbox_dir}.", file=sys.stderr)
    print("Ensure benchmarking_sandbox_management.py (FastAPI version) is present in the 'sandbox' directory.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    if 'benchmarking_sandbox_management' in sys.modules:
        sys.path.pop(0)


# Optional: Use rich for better formatting
try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.markdown import Markdown # For potentially displaying markdown output
    from rich.text import Text # For handling plain text better
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    # Simple print/input fallback if rich is not installed
    class Console:
        def print(self, *args, **kwargs): print(*args)
    class Prompt:
        @staticmethod
        def ask(prompt, choices=None, default=None):
            p_text = f"{prompt} "
            if choices: choices_str = '/'.join(choices); p_text += f"({choices_str}) "
            if default: p_text += f"[{default}] "
            return input(p_text).strip()
        @staticmethod
        def get_input(console, prompt, password=False):
             return input(f"{prompt}: ")
    class Confirm:
        @staticmethod
        def ask(prompt, default=False):
            val = input(f"{prompt} [y/N] " if not default else f"{prompt} [Y/n] ").lower().strip()
            if not val: return default
            return val == 'y'
    class Panel:
        def __init__(self, content, title="", border_style=""): self.content=str(content); self.title=title # Ensure content is string
        def __rich_console__(self, console, options): yield self.title; yield self.content
    class Syntax:
        def __init__(self, code, lexer, theme="", line_numbers=False): self.code = code; self.lexer = lexer
        def __rich_console__(self, console, options): yield f"--- Code ({self.lexer}) ---\n{self.code}\n--- End Code ---"
    class Table:
        def __init__(self, title=""): self._title=title; self._rows=[]; self._columns=[]
        def add_column(self, header, style="", justify="left", no_wrap=False): self._columns.append(header)
        def add_row(self, *items):
            if len(items) != len(self._columns): raise ValueError("Row items != columns")
            self._rows.append(items)
        def __rich_console__(self, console, options):
            yield self._title;
            if self._columns:
                yield "\t".join(self._columns)
                for row in self._rows: yield "\t".join(map(str, row))
        def print_table(self, console):
            console.print(self._title)
            if self._columns:
                col_widths = [len(h) for h in self._columns]
                for row in self._rows:
                    for i, item in enumerate(row): col_widths[i] = max(col_widths[i], len(str(item)))
                header = "  ".join(f"{h:<{w}}" for h, w in zip(self._columns, col_widths))
                separator = "-" * len(header)
                console.print(header); console.print(separator)
                for row in self._rows:
                    console.print("  ".join(f"{str(item):<{w}}" for item, w in zip(row, col_widths)))
    # Dummy classes for rich elements not used directly but potentially in display logic
    class Markdown:
         def __init__(self, content): self.content = content
         def __rich_console__(self, console, options): yield f"--- Markdown ---\n{self.content}\n--- End Markdown ---"
    class Text:
         def __init__(self, content): self.content = content
         def __rich_console__(self, console, options): yield self.content


# --- Constants ---
SCRIPT_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = SCRIPT_DIR / "datasets"
OUTPUTS_DIR = SCRIPT_DIR / "outputs" # Define output directory
ENV_FILE = SCRIPT_DIR / ".env"
SANDBOX_DATA_PATH = "/home/sandboxuser/data.h5ad" # Where data will be copied inside container
# URL for the FastAPI service running in the container (mapped to host)
API_BASE_URL = f"http://localhost:{API_PORT_HOST}"
EXECUTE_ENDPOINT = f"{API_BASE_URL}/execute"
STATUS_ENDPOINT = f"{API_BASE_URL}/status"

# --- Configuration Loading ---
console = Console()
load_dotenv(dotenv_path=ENV_FILE)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    console.print(f"[bold red]Error:[/bold red] OPENAI_API_KEY not found in {ENV_FILE}.")
    console.print("Please run the 'create_benchmarking_env.sh' script first.")
    sys.exit(1)

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    console.print(f"[bold red]Error initializing OpenAI client:[/bold red] {e}")
    sys.exit(1)


# --- Helper Functions ---
def extract_python_code(text):
    """Extracts the first Python code block from text."""
    match = re.search(r"```python\s*([\s\S]+?)\s*```", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None

def display_message(role, content):
    """Displays messages with nice formatting."""
    # Simplified display, as results are now processed separately
    if role == "system":
        console.print(Panel(content, title="SYSTEM PROMPT", border_style="dim blue"))
    elif role == "user":
        # Check if it's the special code execution result message
        if content.startswith("Code execution result:\n"):
             # This will now contain formatted output from the API call
             console.print(Panel(content, title="CODE EXECUTION RESULT (Sent as User)", border_style="yellow"))
        else:
             console.print(Panel(content, title="USER (Input Prompt)", border_style="blue"))
    elif role == "assistant":
        code = extract_python_code(content)
        if code:
            text_part = re.sub(r"```python\s*([\s\S]+?)\s*```", "", content, count=1).strip()
            if text_part:
                console.print(Panel(text_part, title="ASSISTANT (Text)", border_style="green"))
            if HAS_RICH:
                console.print(Panel(Syntax(code, "python", theme="default", line_numbers=True), title="ASSISTANT (Code)", border_style="green"))
            else:
                console.print(f"--- ASSISTANT (Code) ---\n{code}\n--- End Code ---")
        else:
            console.print(Panel(content, title="ASSISTANT (Text Only)", border_style="green"))
    else:
        console.print(f"[bold]{role.upper()}:[/bold]\n{content}")
    console.print("-" * 20) # Separator

def format_api_response_for_llm(response_data):
    """Formats the JSON response from the /execute endpoint into a string for the LLM."""
    output_lines = ["Code execution result:"]
    final_status = response_data.get("final_status", "unknown")
    outputs = response_data.get("outputs", [])

    stdout_lines = []
    stderr_lines = []
    error_info = None
    display_items = [] # Store items for potential later display/saving
    max_len = 1000 # Max length for stdout/stderr truncation

    for item in outputs:
        output_type = item.get("type")
        if output_type == "stream":
            if item.get("name") == "stdout":
                stdout_lines.append(item.get("text", ""))
            elif item.get("name") == "stderr":
                stderr_lines.append(item.get("text", ""))
        elif output_type == "error":
            error_info = item # Store the whole error dict
            # Add error info to stderr for LLM visibility
            stderr_lines.append(f"Error: {item.get('ename', 'UnknownError')}: {item.get('evalue', '')}\n")
            stderr_lines.extend(line + '\n' for line in item.get('traceback', []))
        elif output_type == "display_data":
            # Indicate that display data was generated
            mime_types = list(item.get("data", {}).keys())
            display_items.append(item) # Store for later processing if needed
            output_lines.append(f"[Display data generated: {', '.join(mime_types)}]")
            # Optionally include plain text representation if available
            if 'text/plain' in item.get('data', {}):
                 stdout_lines.append(item['data']['text/plain'] + '\n')
        elif output_type == "execute_result":
             # Append plain text representation to stdout
             if 'text/plain' in item.get('data', {}):
                 stdout_lines.append(item['data']['text/plain'] + '\n')

    # Combine stdout
    if stdout_lines:
        output_lines.append("--- STDOUT ---")
        full_stdout = "".join(stdout_lines)
        if len(full_stdout) > max_len:
             output_lines.append(full_stdout[:max_len] + "\n... (stdout truncated)")
        else:
             output_lines.append(full_stdout)
        output_lines.append("--------------")
    else:
        output_lines.append("[No standard output]")

    # Combine stderr
    if stderr_lines:
        output_lines.append("--- STDERR ---")
        full_stderr = "".join(stderr_lines)
        # --- ADDED STDERR TRUNCATION ---
        if len(full_stderr) > max_len:
             output_lines.append(full_stderr[:max_len] + "\n... (stderr truncated)")
        else:
             output_lines.append(full_stderr)
        # --- END STDERR TRUNCATION ---
        output_lines.append("--------------")
    # No need for an else block if stderr is empty, unlike stdout

    output_lines.append(f"Final Status: {final_status}")

    # Optionally save images/plots from display_items here
    # Create outputs directory if it doesn't exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(display_items):
        if item['type'] == 'display_data':
            for mime, b64_data in item.get('data', {}).items():
                if mime.startswith('image/'):
                    try:
                        image_data = base64.b64decode(b64_data)
                        ext = mime.split('/')[-1].split('+')[0] # Handle things like image/svg+xml
                        # Create a more descriptive filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = OUTPUTS_DIR / f"output_image_{timestamp}_{i}.{ext}"
                        with open(filename, "wb") as f:
                            f.write(image_data)
                        output_lines.append(f"[Saved image data ({mime}) to {filename}]")
                        console.print(f"[bold yellow]Saved image data ({mime}) to {filename}[/bold yellow]")
                    except Exception as e:
                        output_lines.append(f"[Error processing/saving display data {mime}: {e}]")
                        console.print(f"[red]Error processing/saving display data {mime}: {e}[/red]")

    return "\n".join(output_lines)


# --- Core Logic Functions ---

def get_agent_prompts():
    """Gets agent prompt(s) based on user input method."""
    # (No changes needed here)
    prompts = {}
    while True:
        console.print("\n[bold cyan]Select Agent Prompt Input Method:[/bold cyan]")
        console.print("  1. Paste prompt directly into the terminal.")
        console.print("  2. Provide path to a single .txt file.")
        console.print("  3. Provide path to a folder containing .txt prompt files.")
        choice = Prompt.ask("Enter choice (1/2/3)", choices=["1", "2", "3"], default="1")
        if choice == "1":
            console.print("Paste your prompt below. Press Ctrl+D (Unix) or Ctrl+Z+Enter (Windows) when done:")
            try:
                prompt_text = sys.stdin.read().strip()
                if prompt_text: prompts["pasted_prompt"] = prompt_text; return prompts
                else: console.print("[yellow]No prompt pasted. Please try again.[/yellow]")
            except EOFError: console.print("\n[yellow]No prompt pasted. Please try again.[/yellow]")
        elif choice == "2":
            file_path_str = Prompt.ask("Enter the path to the .txt prompt file")
            file_path = Path(file_path_str).resolve()
            if file_path.is_file() and file_path.suffix.lower() == ".txt":
                try:
                    prompt_text = file_path.read_text(encoding='utf-8').strip()
                    if prompt_text: prompts[file_path.stem] = prompt_text; return prompts
                    else: console.print(f"[yellow]File '{file_path}' is empty.[/yellow]")
                except Exception as e: console.print(f"[red]Error reading file '{file_path}': {e}[/red]")
            else: console.print(f"[red]Invalid path or not a .txt file: '{file_path_str}'[/red]")
        elif choice == "3":
            folder_path_str = Prompt.ask("Enter the path to the folder containing .txt prompt files")
            folder_path = Path(folder_path_str).resolve()
            if folder_path.is_dir():
                txt_files = list(folder_path.glob("*.txt"))
                if not txt_files: console.print(f"[yellow]No .txt files found in folder '{folder_path_str}'.[/yellow]"); continue
                for file_path in txt_files:
                    try:
                        prompt_text = file_path.read_text(encoding='utf-8').strip()
                        if prompt_text: prompts[file_path.stem] = prompt_text
                        else: console.print(f"[yellow]Skipping empty file: '{file_path.name}'[/yellow]")
                    except Exception as e: console.print(f"[red]Error reading file '{file_path.name}': {e}[/red]")
                if prompts: console.print(f"Found {len(prompts)} non-empty prompt files."); return prompts
                else: console.print("[yellow]No valid, non-empty prompts found in the folder.[/yellow]")
            else: console.print(f"[red]Invalid path or not a directory: '{folder_path_str}'[/red]")


def select_dataset():
    """Scans datasets directory and prompts user for selection."""
    # (No changes needed here)
    if not DATASETS_DIR.is_dir():
        console.print(f"[bold red]Error:[/bold red] Datasets directory not found at '{DATASETS_DIR}'")
        console.print("Please ensure datasets are downloaded using 'czi_browser.py download ...'")
        return None, None
    datasets = []
    for h5ad_path in DATASETS_DIR.glob("*.h5ad"):
        json_path = h5ad_path.with_suffix(".json")
        if json_path.is_file():
            try:
                with open(json_path, 'r', encoding='utf-8') as f: metadata = json.load(f)
                datasets.append({ "h5ad_path": h5ad_path, "json_path": json_path, "metadata": metadata, "display_name": metadata.get("dataset_title", h5ad_path.stem)})
            except Exception as e: console.print(f"[yellow]Warning: Could not load metadata for '{h5ad_path.name}': {e}[/yellow]")
        else: console.print(f"[yellow]Warning: Missing metadata file for '{h5ad_path.name}'. Skipping.[/yellow]")
    if not datasets:
        console.print(f"[bold red]Error:[/bold red] No valid datasets found in '{DATASETS_DIR}'")
        return None, None
    console.print("\n[bold cyan]Available Datasets:[/bold cyan]")
    table = Table(title="Select a Dataset")
    table.add_column("Index", style="dim", justify="right")
    table.add_column("Dataset Title / Filename", style="green")
    table.add_column("Cell Count", style="magenta", justify="right")
    table.add_column("Organism", style="blue")
    for i, ds in enumerate(datasets):
        meta = ds["metadata"]; cell_count = meta.get('cell_count', 'N/A'); organism = ", ".join(meta.get('organism', [])) if isinstance(meta.get('organism'), list) else meta.get('organism', 'N/A')
        try: cell_count_str = f"{int(cell_count):,}" if cell_count != 'N/A' else 'N/A'
        except (ValueError, TypeError): cell_count_str = str(cell_count)
        table.add_row(str(i + 1), ds["display_name"], cell_count_str, organism)
    if HAS_RICH: console.print(table)
    else: table.print_table(console)
    while True:
        choice_str = Prompt.ask(f"Enter the index of the dataset to use (1-{len(datasets)})")
        try:
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(datasets):
                selected_ds = datasets[choice_idx]
                console.print(f"Selected dataset: [green]{selected_ds['display_name']}[/green]")
                return selected_ds["h5ad_path"], selected_ds["metadata"]
            else: console.print(f"[red]Invalid index. Please enter a number between 1 and {len(datasets)}.[/red]")
        except ValueError: console.print("[red]Invalid input. Please enter a number.[/red]")

def get_code_tries():
    """Prompts user for the number of code execution attempts."""
    # (No changes needed here)
    while True:
        tries_str = Prompt.ask("Enter the maximum number of code execution attempts for the agent", default="5")
        try:
            tries = int(tries_str)
            if tries > 0: return tries
            else: console.print("[red]Number of tries must be positive.[/red]")
        except ValueError: console.print("[red]Invalid input. Please enter an integer.[/red]")

def check_api_status(max_retries=5, delay=2):
    """Checks if the FastAPI service is responsive."""
    console.print(f"Checking API status at {STATUS_ENDPOINT}...")
    for attempt in range(max_retries):
        try:
            response = requests.get(STATUS_ENDPOINT, timeout=5) # Short timeout for status check
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            data = response.json()
            if data.get("status") == "ok":
                console.print("[green]API service is responsive.[/green]")
                return True
            else:
                 console.print(f"[yellow]API status endpoint returned unexpected data: {data}[/yellow]")
        except requests.exceptions.ConnectionError:
            console.print(f"[yellow]API connection failed (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...[/yellow]")
        except requests.exceptions.Timeout:
             console.print(f"[yellow]API status check timed out (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...[/yellow]")
        except requests.exceptions.RequestException as e:
            console.print(f"[red]API status check error (attempt {attempt+1}/{max_retries}): {e}[/red]")
            # Don't retry immediately on other request errors
            break
        time.sleep(delay)
    console.print("[bold red]API service did not become responsive.[/bold red]")
    return False


def run_agent_test(agent_prompt_id, agent_prompt, dataset_h5ad_path, dataset_metadata, max_code_tries):
    """Runs a single agent test loop using the FastAPI kernel service."""
    console.print(f"\n[bold cyan]----- Starting Test: '{agent_prompt_id}' ----- [/bold cyan]")
    console.print(f"Dataset: [green]{dataset_metadata.get('dataset_title', dataset_h5ad_path.stem)}[/green]")
    console.print(f"Max Code Tries: [yellow]{max_code_tries}[/yellow]")

    sandbox_manager = None
    conversation_history = []
    code_tries_left = max_code_tries
    # Add metadata to the conversation start for saving context
    initial_context = {
        "prompt_id": agent_prompt_id,
        "dataset_file": str(dataset_h5ad_path.name),
        "dataset_metadata": dataset_metadata,
        "max_code_tries": max_code_tries,
        "start_time": datetime.now().isoformat()
    }
    # Store the raw API responses alongside the conversation turns
    full_conversation_data = {"context": initial_context, "turns": []}


    try:
        # 1. Initialize Manager and Start Sandbox Container with API service
        console.print("\nInitializing Sandbox Manager...")
        sandbox_manager = SandboxManager() # Manager now just handles container lifecycle
        console.print("Starting sandbox container with API service...")
        if not sandbox_manager.start_container(): # start_container now waits briefly
            console.print("[bold red]Failed to start sandbox container. Aborting test.[/bold red]")
            return None # Return None if setup fails

        # 1b. Check if API is responsive
        if not check_api_status():
             console.print("[bold red]API service failed to start or respond. Aborting test.[/bold red]")
             # Attempt cleanup
             sandbox_manager.stop_container(remove=True)
             return None # Return None if setup fails

        # 2. Copy Dataset to Sandbox (Still needed)
        console.print(f"Copying dataset '{dataset_h5ad_path.name}' to sandbox ({SANDBOX_DATA_PATH})...")
        # Ensure container name constant is correct
        copy_command = ['docker', 'cp', str(dataset_h5ad_path), f"{SANDBOX_CONTAINER_NAME}:{SANDBOX_DATA_PATH}"]
        try:
            # Use subprocess.run, check for errors
            result = subprocess.run(copy_command, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                 console.print(f"[bold red]Error copying dataset to container:[/bold red]")
                 console.print(f"Command: {' '.join(copy_command)}")
                 console.print(f"Return Code: {result.returncode}")
                 console.print(f"Stderr: {result.stderr}")
                 console.print(f"Stdout: {result.stdout}")
                 # Decide if this is fatal
                 raise subprocess.CalledProcessError(result.returncode, copy_command, output=result.stdout, stderr=result.stderr)
            else:
                 console.print("[green]Dataset copied successfully.[/green]")
        except subprocess.CalledProcessError as e:
             console.print(f"[bold red]Dataset copy failed. Aborting test.[/bold red]")
             raise # Re-raise to be caught by outer try/except for cleanup

        # 3. Prepare Initial Agent Message
        system_message_content = f"""You are an AI assistant tasked with analyzing a single-cell transcriptomics dataset.
Your goal is to characterize this dataset based on its metadata and by generating Python code to be executed.
The dataset file is located inside the execution environment at: {SANDBOX_DATA_PATH}
Standard libraries like pandas, numpy, scipy, scikit-learn, and anndata should be available.
Variables and imports **persist** between your code executions within this session.

Dataset Metadata:
{json.dumps(dataset_metadata, indent=2)}

You have a maximum of {max_code_tries} attempts to generate Python code blocks for execution.
When you want to execute code, enclose it **only** in a single triple-backtick block with the language specified as python, like this:
```python
# Your analysis code here. Imports and variables persist.
# Example: Load data in the first turn:
import anndata as ad
adata = ad.read_h5ad('{SANDBOX_DATA_PATH}')
print(adata.shape)

# Example: Use adata in a later turn:
print(adata.obs['cell_type'].value_counts())

# Example: Generate a plot (it will be captured if possible)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(adata.obsm['X_umap'][:,0], adata.obsm['X_umap'][:,1])
# plt.title('UMAP Plot')
# plt.show() # Or savefig
```
I will execute the code you provide and return the results (stdout, stderr, errors, and potentially image data). Use the results to inform your next step.
Focus on providing meaningful characterizations and insights based on the data and metadata. Plan your {max_code_tries} code executions wisely. Start by loading the data.

While you can generate plots, please prioritize investigating via text as you do not have the ability to understand images.
"""
        user_message_content = agent_prompt

        # Store initial messages for saving later
        full_conversation_data["turns"].append({"role": "system", "content": system_message_content})
        full_conversation_data["turns"].append({"role": "user", "content": user_message_content})

        # Prepare history for the API call (needs to be in the format OpenAI expects)
        conversation_history = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]
        display_message("system", system_message_content)
        display_message("user", user_message_content)

        # 4. Agent Interaction Loop
        while code_tries_left > 0:
            console.print(f"\n[bold]Sending request to OpenAI... (Code tries left: {code_tries_left})[/bold]")
            api_call_successful = False
            response_data = None # To store API response for saving
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o", # Or your preferred model
                    messages=conversation_history,
                    temperature=0.7,
                )
                assistant_message = response.choices[0].message
                assistant_content = assistant_message.content
                api_call_successful = True # Mark OpenAI call as successful

                # Add assistant message to both histories
                conversation_history.append({"role": "assistant", "content": assistant_content})
                full_conversation_data["turns"].append({"role": "assistant", "content": assistant_content})
                display_message("assistant", assistant_content)

                # 5. Check for and Execute Code via API
                agent_code = extract_python_code(assistant_content)
                if agent_code:
                    console.print(f"\n[bold cyan]Executing Code via API (Attempt {max_code_tries - code_tries_left + 1}/{max_code_tries})...[/bold cyan]")
                    code_tries_left -= 1
                    user_feedback_content = "[Code execution failed or API unreachable]" # Default feedback
                    execution_api_response = None # Store raw API response

                    try:
                        payload = {"code": agent_code, "timeout": 120}
                        headers = {"Content-Type": "application/json"}
                        api_response = requests.post(EXECUTE_ENDPOINT, json=payload, headers=headers, timeout=130)
                        api_response.raise_for_status()
                        execution_api_response = api_response.json() # Store successful response
                        user_feedback_content = format_api_response_for_llm(execution_api_response)

                    except requests.exceptions.RequestException as e:
                         console.print(f"[bold red]API Request Error during execution: {e}[/bold red]")
                         error_detail = str(e)
                         if e.response is not None:
                              console.print(f"Response Status: {e.response.status_code}")
                              error_detail = e.response.text
                              try: # Try to get detail from JSON
                                   detail_json = e.response.json().get("detail", error_detail)
                                   error_detail = f"API Error {e.response.status_code}: {detail_json}"
                              except json.JSONDecodeError:
                                   error_detail = f"API Error {e.response.status_code}: {e.response.text}"
                         user_feedback_content = f"Code execution result:\n[{error_detail}]"
                         # Store error info instead of successful response
                         execution_api_response = {"error": error_detail, "status_code": e.response.status_code if e.response else None}
                         # break # Decide if API errors should stop the loop

                    # Append execution result back to conversation history for LLM
                    conversation_history.append({"role": "user", "content": user_feedback_content})
                    # Store user feedback and API response in the full data log
                    full_conversation_data["turns"].append({
                        "role": "user",
                        "content": user_feedback_content,
                        "api_response": execution_api_response # Add raw API response here
                    })
                    display_message("user", user_feedback_content) # Display formatted results

                    if code_tries_left == 0:
                        console.print("[bold yellow]Maximum code execution attempts reached.[/bold yellow]")
                        break

                else: # No code found in assistant response
                    console.print("[yellow]No code block found in assistant's response this turn.[/yellow]")
                    # Add a placeholder turn to keep track
                    full_conversation_data["turns"].append({"role": "user", "content": "[No code executed this turn]"})


            except APIError as e:
                console.print(f"[bold red]OpenAI API Error:[/bold red] {e}")
                if hasattr(e, 'body') and e.body: console.print(f"Error Body: {e.body}")
                # Store error in results
                full_conversation_data["error"] = f"OpenAI API Error: {e}"
                break # Stop test on OpenAI error
            except Exception as e:
                console.print(f"[bold red]Error during agent interaction: {e}[/bold red]")
                import traceback
                traceback.print_exc() # Print traceback for unexpected errors
                full_conversation_data["error"] = f"Agent Interaction Error: {e}\n{traceback.format_exc()}"
                break # Stop test on other errors

        console.print(f"\n[bold cyan]----- Test Finished: '{agent_prompt_id}' ----- [/bold cyan]")
        # Return the detailed conversation data including context and API responses
        return full_conversation_data

    except Exception as e:
        console.print(f"[bold red]An error occurred during test setup or execution for '{agent_prompt_id}':[/bold red] {e}")
        import traceback
        traceback.print_exc()
        # Return error information if setup failed
        return {"context": initial_context, "error": f"Setup/Execution Error: {e}\n{traceback.format_exc()}"}
    finally:
        # 6. Stop and Cleanup Sandbox
        if sandbox_manager:
            console.print("\nStopping sandbox container...")
            if not sandbox_manager.stop_container(remove=True):
                console.print("[yellow]Warning: Could not cleanly stop/remove sandbox container.[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="Run AI agent benchmarks against datasets in a sandbox (API Mode).")
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory to save results JSON file (default: outputs)"
    )
    args = parser.parse_args()

    # Use Path object for output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Create output dir if needed

    console.print("[bold blue]Welcome to the One-Shot Agent Tester (API Mode)![/bold blue]")

    agent_prompts = get_agent_prompts()
    if not agent_prompts: console.print("[red]No agent prompts provided. Exiting.[/red]"); sys.exit(1)

    dataset_h5ad_path, dataset_metadata = select_dataset()
    if not dataset_h5ad_path or not dataset_metadata: console.print("[red]No dataset selected or available. Exiting.[/red]"); sys.exit(1)

    max_code_tries = get_code_tries()

    # Dictionary to hold results for all prompts run in this session
    all_results = {}

    for prompt_id, prompt_text in agent_prompts.items():
        # Run the test and get the detailed conversation data
        test_result_data = run_agent_test(
            prompt_id,
            prompt_text,
            dataset_h5ad_path,
            dataset_metadata,
            max_code_tries
        )
        # Store the result under the prompt ID
        all_results[prompt_id] = test_result_data

        if len(agent_prompts) > 1:
             if not Confirm.ask(f"\nTest for '{prompt_id}' finished. Continue with the next agent prompt?", default=True):
                 console.print("[yellow]Aborting remaining tests.[/yellow]"); break
             console.print("\n" + "="*40 + "\n"); time.sleep(1) # Separator and pause

    console.print("\n[bold blue]All specified agent tests have concluded.[/bold blue]")

    # --- Save all results to a single JSON file ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include dataset name stem in filename for clarity
    dataset_stem = dataset_h5ad_path.stem if dataset_h5ad_path else "unknown_dataset"
    output_filename = output_dir / f"benchmark_results_{dataset_stem}_{timestamp}.json"

    console.print(f"Saving all results to [cyan]{output_filename}[/cyan]...")
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            # Use default=str to handle potential non-serializable objects like Path
            json.dump(all_results, f, indent=2, default=str)
        console.print("[green]Results saved successfully.[/green]")
    except TypeError as e:
         console.print(f"[bold red]Error: Failed to serialize results to JSON:[/bold red] {e}")
         console.print("Check if non-serializable objects (like Path) are in the results data.")
    except Exception as e:
         console.print(f"[bold red]Error saving results to {output_filename}:[/bold red] {e}")

if __name__ == "__main__":
    main()
