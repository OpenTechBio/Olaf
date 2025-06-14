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
from datetime import datetime
import copy # For deep copying conversation history

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

# Assume sandbox manager is in a 'sandbox' subdirectory relative to this script
try:
    sandbox_dir = os.path.join(os.path.dirname(__file__), 'sandbox')
    sys.path.insert(0, sandbox_dir)
    # Import manager and constants needed for running the sandbox
    from benchmarking_sandbox_management import SandboxManager, CONTAINER_NAME as SANDBOX_CONTAINER_NAME, API_PORT_HOST
except ImportError as e:
    print(f"Error: Could not import SandboxManager or constants from {sandbox_dir}.", file=sys.stderr)
    print("Ensure benchmarking_sandbox_management.py (FastAPI version) is present in the 'sandbox' directory.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    # Clean up sys.path modification
    if 'benchmarking_sandbox_management' in sys.modules and sandbox_dir in sys.path:
         # Check if the path is still the one we added before removing
         if sys.path[0] == sandbox_dir:
              sys.path.pop(0)
         else: # If paths changed unexpectedly, search and remove
              try:
                  sys.path.remove(sandbox_dir)
              except ValueError:
                   pass # Path wasn't there

# Optional: Use rich for better formatting
try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None
    # Simple print/input fallback if rich is not installed
    class Console:
        def print(self, *args, **kwargs): print(*args)
    class Prompt:
        @staticmethod
        def ask(prompt, default=None):
            p_text = f"{prompt} "
            if default: p_text += f"[{default}] "
            return input(p_text).strip()
    class Confirm:
        @staticmethod
        def ask(prompt, default=False):
            val = input(f"{prompt} [y/N] " if not default else f"{prompt} [Y/n] ").lower().strip()
            if not val: return default
            return val == 'y'
    class Panel:
         def __init__(self, content, title="", border_style=""): self.content=str(content); self.title=title
         def __rich_console__(self, console, options): yield self.title; yield self.content
    class Syntax:
        def __init__(self, code, lexer, theme="", line_numbers=False): self.code = code; self.lexer = lexer
        def __rich_console__(self, console, options): yield f"--- Code ({self.lexer}) ---\n{self.code}\n--- End Code ---"
    class Table: # Basic fallback Table class
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
        def print_table(self, console): # Custom print method if rich not available
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


# --- Constants ---
SCRIPT_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = SCRIPT_DIR / "datasets"
OUTPUTS_DIR = SCRIPT_DIR / "outputs" # Default output directory for evolution logs
ENV_FILE = SCRIPT_DIR / ".env"
SANDBOX_DATA_PATH = "/home/sandboxuser/data.h5ad" # Path inside container
API_BASE_URL = f"http://localhost:{API_PORT_HOST}"
EXECUTE_ENDPOINT = f"{API_BASE_URL}/execute"
STATUS_ENDPOINT = f"{API_BASE_URL}/status"

# --- Configuration Loading ---
load_dotenv(dotenv_path=ENV_FILE)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Define models for different roles
AGENT_MODEL = "gpt-4o" # Model for the agent being tested
EVALUATOR_MODEL = "gpt-4o" # Model for evaluating the agent's performance
EVOLVER_MODEL = "gpt-4o" # Model for evolving the prompt

if not OPENAI_API_KEY:
    if console: console.print(f"[bold red]Error:[/bold red] OPENAI_API_KEY not found in {ENV_FILE}.")
    else: print(f"Error: OPENAI_API_KEY not found in {ENV_FILE}.")
    sys.exit(1)

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    if console: console.print(f"OpenAI client initialized.")
    else: print(f"OpenAI client initialized.")
except Exception as e:
    if console: console.print(f"[bold red]Error initializing OpenAI client:[/bold red] {e}")
    else: print(f"Error initializing OpenAI client: {e}")
    sys.exit(1)

# --- Helper Functions (Adapted from Tester/Evaluator) ---

def extract_python_code(text):
    """Extracts the first Python code block from text."""
    if text is None: return None
    match = re.search(r"```python\s*([\s\S]+?)\s*```", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None

def select_dataset():
    """Scans datasets directory and prompts user for selection."""
    # (Copied from OneShotAgentTester.py - requires DATASETS_DIR constant)
    if not DATASETS_DIR.is_dir():
        console.print(f"[bold red]Error:[/bold red] Datasets directory not found at '{DATASETS_DIR}'")
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

def check_api_status(max_retries=5, delay=2):
    """Checks if the FastAPI service is responsive."""
    # (Copied from OneShotAgentTester.py)
    console.print(f"Checking API status at {STATUS_ENDPOINT}...")
    for attempt in range(max_retries):
        try:
            response = requests.get(STATUS_ENDPOINT, timeout=5)
            response.raise_for_status()
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
            break
        time.sleep(delay)
    console.print("[bold red]API service did not become responsive.[/bold red]")
    return False

def format_api_response_for_llm(response_data):
    """Formats the JSON response from the /execute endpoint into a string for the LLM."""
    # (Copied from OneShotAgentTester.py - simplified image handling)
    output_lines = ["Code execution result:"]
    final_status = response_data.get("final_status", "unknown")
    outputs = response_data.get("outputs", [])
    stdout_lines = []
    stderr_lines = []
    max_len = 1500 # Slightly shorter truncation for evaluator context

    for item in outputs:
        output_type = item.get("type")
        if output_type == "stream":
            if item.get("name") == "stdout": stdout_lines.append(item.get("text", ""))
            elif item.get("name") == "stderr": stderr_lines.append(item.get("text", ""))
        elif output_type == "error":
            stderr_lines.append(f"Error: {item.get('ename', 'UnknownError')}: {item.get('evalue', '')}\n")
            stderr_lines.extend(line + '\n' for line in item.get('traceback', []))
        elif output_type == "display_data":
            mime_types = list(item.get("data", {}).keys())
            output_lines.append(f"[Display data generated: {', '.join(mime_types)}]")
            if 'text/plain' in item.get('data', {}): stdout_lines.append(item['data']['text/plain'] + '\n')
        elif output_type == "execute_result":
             if 'text/plain' in item.get('data', {}): stdout_lines.append(item['data']['text/plain'] + '\n')

    if stdout_lines:
        output_lines.append("--- STDOUT ---")
        full_stdout = "".join(stdout_lines)
        output_lines.append(full_stdout[:max_len] + ("\n... (stdout truncated)" if len(full_stdout) > max_len else ""))
        output_lines.append("--------------")
    else: output_lines.append("[No standard output]")

    if stderr_lines:
        output_lines.append("--- STDERR ---")
        full_stderr = "".join(stderr_lines)
        output_lines.append(full_stderr[:max_len] + ("\n... (stderr truncated)" if len(full_stderr) > max_len else ""))
        output_lines.append("--------------")

    output_lines.append(f"Final Status: {final_status}")
    return "\n".join(output_lines)

# --- ADDED FUNCTION DEFINITION ---
def format_conversation_for_eval(test_data):
    """ Formats the conversation turns into a readable string for the evaluator prompt. """
    if not test_data or "turns" not in test_data:
        return "[No conversation turns found]"

    formatted_lines = []
    for i, turn in enumerate(test_data.get("turns", [])):
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "[No content]")

        # Shorten system prompt for brevity in evaluation context
        if role == "SYSTEM":
             content = "[System Prompt Provided - see original log for details]"

        # Format code execution results more clearly
        if role == "USER" and content.startswith("Code execution result:"):
             # Check if this is the *actual* result turn by looking at previous turn
             # This avoids mislabeling the initial user prompt if it somehow contained the phrase
             if i > 0 and test_data["turns"][i-1].get("role") == "assistant":
                  content = content.replace("Code execution result:", "**CODE EXECUTION RESULT:**")
                  content = content.replace("--- STDOUT ---", "**STDOUT:**")
                  content = content.replace("--- STDERR ---", "**STDERR:**")
                  content = content.replace("--------------", "---") # Shorten separator
             else:
                  # Treat as regular user prompt if it wasn't preceded by assistant turn
                  role = "USER PROMPT (Initial)"


        # Add role separator, handling potential consecutive roles if needed
        formatted_lines.append(f"--- {role} ---")
        formatted_lines.append(content)
        formatted_lines.append("\n") # Add space between turns

    return "\n".join(formatted_lines)
# --- END ADDED FUNCTION DEFINITION ---

def run_single_test_iteration(agent_prompt_id, agent_prompt, dataset_h5ad_path, dataset_metadata, max_code_tries=5):
    """
    Runs one iteration using the agent prompt, adapted from OneShotAgentTester.
    Returns the detailed conversation data including context and API responses.
    """
    console.print(f"\n[magenta]--- Running Test Iteration for Prompt ID: '{agent_prompt_id}' ---[/magenta]")
    sandbox_manager = None
    full_conversation_data = {} # Initialize

    # Create initial context for this specific run
    initial_context = {
        "prompt_id": agent_prompt_id,
        "dataset_file": str(dataset_h5ad_path.name),
        "dataset_metadata": dataset_metadata,
        "max_code_tries": max_code_tries,
        "start_time": datetime.now().isoformat()
    }
    full_conversation_data = {"context": initial_context, "turns": []}

    try:
        sandbox_manager = SandboxManager()
        if not sandbox_manager.start_container():
            raise RuntimeError("Failed to start sandbox container.")
        if not check_api_status():
             raise RuntimeError("API service failed to start or respond.")

        # Copy dataset - Ensure SANDBOX_DATA_PATH is defined globally
        copy_command = ['docker', 'cp', str(dataset_h5ad_path), f"{SANDBOX_CONTAINER_NAME}:{SANDBOX_DATA_PATH}"]
        result = subprocess.run(copy_command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[bold red]Error copying dataset to container (Code: {result.returncode}):[/bold red]\n{result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, copy_command, output=result.stdout, stderr=result.stderr)
        console.print("[green]Dataset copied successfully.[/green]")

        # Prepare conversation
        system_message_content = f"""You are an AI assistant tasked with analyzing a single-cell transcriptomics dataset.
The dataset file is located inside the execution environment at: {SANDBOX_DATA_PATH}
Variables and imports persist between your code executions.

Dataset Metadata:
{json.dumps(dataset_metadata, indent=2)}

Max code attempts: {max_code_tries}. Generate Python code in ```python ... ``` blocks.
Prioritize text analysis over plots. Start by loading the data.
"""
        user_message_content = agent_prompt # The prompt being tested

        full_conversation_data["turns"].append({"role": "system", "content": system_message_content})
        full_conversation_data["turns"].append({"role": "user", "content": user_message_content})
        conversation_history = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]
        console.print(Panel(system_message_content, title="SYSTEM PROMPT (Iteration)", border_style="dim blue"))
        console.print(Panel(user_message_content, title="USER PROMPT (Iteration)", border_style="blue"))

        # Agent Interaction Loop
        code_tries_left = max_code_tries
        while code_tries_left > 0:
            console.print(f"\nSending request to Agent ({AGENT_MODEL})... (Tries left: {code_tries_left})")
            response = openai_client.chat.completions.create(
                model=AGENT_MODEL, messages=conversation_history, temperature=0.7,
            )
            assistant_content = response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": assistant_content})
            full_conversation_data["turns"].append({"role": "assistant", "content": assistant_content})
            console.print(Panel(assistant_content, title="ASSISTANT RESPONSE", border_style="green"))

            agent_code = extract_python_code(assistant_content)
            if agent_code:
                console.print(f"Executing Code via API (Attempt {max_code_tries - code_tries_left + 1}/{max_code_tries})...")
                code_tries_left -= 1
                user_feedback_content = "[Code execution failed]"
                execution_api_response = None
                try:
                    payload = {"code": agent_code, "timeout": 120}
                    headers = {"Content-Type": "application/json"}
                    api_response = requests.post(EXECUTE_ENDPOINT, json=payload, headers=headers, timeout=130)
                    api_response.raise_for_status()
                    execution_api_response = api_response.json()
                    user_feedback_content = format_api_response_for_llm(execution_api_response)
                except requests.exceptions.RequestException as e:
                     console.print(f"[bold red]API Request Error during execution: {e}[/bold red]")
                     error_detail = str(e)
                     status_code = None
                     if e.response is not None:
                          status_code = e.response.status_code
                          console.print(f"Response Status: {status_code}")
                          error_detail = e.response.text
                          try: # Try to get detail from JSON
                               detail_json = e.response.json().get("detail", error_detail)
                               error_detail = f"API Error {status_code}: {detail_json}"
                          except json.JSONDecodeError:
                               error_detail = f"API Error {status_code}: {e.response.text}"
                     user_feedback_content = f"Code execution result:\n[{error_detail}]"
                     execution_api_response = {"error": error_detail, "status_code": status_code}

                conversation_history.append({"role": "user", "content": user_feedback_content})
                full_conversation_data["turns"].append({
                    "role": "user", "content": user_feedback_content, "api_response": execution_api_response
                })
                console.print(Panel(user_feedback_content, title="CODE EXECUTION RESULT", border_style="yellow"))

                if code_tries_left == 0:
                    console.print("[yellow]Maximum code execution attempts reached.[/yellow]")
                    break
            else:
                console.print("[yellow]No code block found. Ending interaction loop for this iteration.[/yellow]")
                full_conversation_data["turns"].append({"role": "user", "content": "[No code executed, agent finished or failed to provide code]"})
                break # Assume conversation ends if agent doesn't provide code

        return full_conversation_data # Return the captured data

    except Exception as e:
        console.print(f"[bold red]Error during test iteration for '{agent_prompt_id}': {e}[/bold red]")
        import traceback
        traceback.print_exc()
        full_conversation_data["error"] = f"Test Iteration Error: {e}\n{traceback.format_exc()}"
        return full_conversation_data # Return data with error
    finally:
        if sandbox_manager:
            console.print("Stopping sandbox container for iteration...")
            sandbox_manager.stop_container(remove=True)

def call_openai_evaluator(conversation_text, context):
    """ Sends the formatted conversation to OpenAI for evaluation. """
    # (Copied from evaluator.py - requires EVALUATOR_MODEL constant)
    evaluator_prompt = f"""You are an expert evaluator assessing the performance of an AI assistant acting as a bioinformatician.
The assistant was given a task related to analyzing a single-cell transcriptomics dataset.
The expected performance level is that of an **entry-level post-graduate bioinformatician**.

**Dataset Context:**
- Dataset File: {context.get('dataset_file', 'N/A')}
- Key Metadata: {json.dumps(context.get('dataset_metadata', {}), indent=1, default=str)}

**Task Context:**
- Initial User Prompt: See the first USER prompt below.
- Max Code Attempts Allowed: {context.get('max_code_tries', 'N/A')}

**Conversation Log:**
{conversation_text}

**Evaluation Task:**
Based on the conversation log, evaluate the AI assistant's performance. Consider the following:
1. Correctness: Was the generated code correct and did it achieve the intended analysis steps?
2. Efficiency: Was the approach reasonable? Were there unnecessary steps?
3. Interpretation: Did the assistant correctly interpret the results of its code execution?
4. Planning: Did the assistant use its allowed code execution attempts effectively towards the goal?
5. Clarity: Was the assistant's text explanation clear and accurate?
6. Overall Skill: Does the performance align with an entry-level post-graduate bioinformatician?

**Output Format:**
Please provide your evaluation strictly in the following JSON format ONLY. Do not include any other text before or after the JSON block:
{{
  "grade": <integer between 0 and 100>,
  "comments": "<string containing your detailed evaluation justifying the grade>"
}}
"""
    console.print(f"Sending evaluation request ({EVALUATOR_MODEL})...")
    try:
        response = openai_client.chat.completions.create(
            model=EVALUATOR_MODEL,
            messages=[{"role": "user", "content": evaluator_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        eval_content = response.choices[0].message.content
        console.print("[green]Evaluation received.[/green]")
        try:
            eval_json = json.loads(eval_content)
            if "grade" in eval_json and "comments" in eval_json and \
               isinstance(eval_json["grade"], int) and isinstance(eval_json["comments"], str):
                 return eval_json
            else: raise ValueError("Invalid format in evaluation JSON.")
        except (json.JSONDecodeError, ValueError) as e:
            console.print(f"[bold red]Error parsing evaluation JSON: {e}[/bold red]")
            console.print(f"Raw response:\n{eval_content}")
            return {"grade": -1, "comments": f"Error parsing evaluation: {e}\nRaw: {eval_content}"}
    except Exception as e:
        console.print(f"[bold red]Error calling evaluation API: {e}[/bold red]")
        return {"grade": -1, "comments": f"API Error: {e}"}

def call_openai_evolver(objective, previous_prompt, conversation_text, evaluation):
    """ Calls OpenAI to generate an improved prompt. """
    evolver_prompt = f"""You are an AI Prompt Engineer specializing in bioinformatics tasks.
Your goal is to refine a user prompt to improve the performance of another AI assistant on a specific objective, based on past performance.

**Overall Objective:**
{objective}

**Previous Prompt Attempt:**
```
{previous_prompt}
```

**Resulting Conversation Log (summary):**
{conversation_text[:3000]}... (log truncated for brevity)

**Evaluation of Previous Attempt:**
- Grade (0-100): {evaluation.get('grade', 'N/A')}
- Evaluator Comments: {evaluation.get('comments', 'N/A')}

**Task:**
Based on the objective, the previous prompt, the conversation summary, and the evaluation feedback, generate a **new, improved prompt** for the AI assistant.
The new prompt should:
- Be clearer and more specific about the desired analysis steps and output.
- Address the weaknesses identified in the evaluator comments.
- Guide the assistant towards better correctness, efficiency, interpretation, and planning.
- Aim to help the assistant perform like an entry-level post-graduate bioinformatician.

**Output Format:**
Please provide ONLY the refined prompt text itself. Do not include any explanations, greetings, or markdown formatting like backticks around the prompt. Just the raw text of the new prompt.
"""
    console.print(f"Sending prompt evolution request ({EVOLVER_MODEL})...")
    try:
        response = openai_client.chat.completions.create(
            model=EVOLVER_MODEL,
            messages=[{"role": "user", "content": evolver_prompt}],
            temperature=0.6, # Allow for some creativity in prompt generation
            max_tokens=500 # Adjust based on expected prompt length
        )
        new_prompt = response.choices[0].message.content.strip()
        # Optional: Basic cleaning if the model adds quotes or markdown
        new_prompt = re.sub(r"^```\s*|\s*```$", "", new_prompt).strip()
        console.print("[green]Received evolved prompt.[/green]")
        return new_prompt
    except Exception as e:
        console.print(f"[bold red]Error calling prompt evolver API: {e}[/bold red]")
        return None # Return None on error, keep using previous prompt


# --- Main Evolution Loop ---
def main_evolution_loop():
    if console: console.print("\n--- Prompt Evolver ---")
    else: print("\n--- Prompt Evolver ---")

    # 1. Get Inputs
    objective = Prompt.ask("Enter the overall objective for the prompt")
    while not objective:
        objective = Prompt.ask("Objective cannot be empty. Please enter the objective")

    initial_prompt_path_str = Prompt.ask("Enter path to initial prompt .txt file (or paste directly if empty)")
    initial_prompt = ""
    if initial_prompt_path_str:
        initial_prompt_path = Path(initial_prompt_path_str)
        if initial_prompt_path.is_file():
            try:
                initial_prompt = initial_prompt_path.read_text(encoding='utf-8').strip()
                console.print(f"Loaded initial prompt from: [cyan]{initial_prompt_path}[/cyan]")
            except Exception as e:
                console.print(f"[red]Error reading prompt file '{initial_prompt_path}': {e}. Please paste prompt.[/red]")
                initial_prompt = ""
        else:
            console.print(f"[yellow]Initial prompt file not found. Please paste prompt.[/yellow]")

    if not initial_prompt:
         console.print("Paste your initial prompt below. Press Ctrl+D (Unix) or Ctrl+Z+Enter (Windows) when done:")
         try:
             initial_prompt = sys.stdin.read().strip()
             if not initial_prompt:
                  console.print("[red]Error: Initial prompt cannot be empty.[/red]")
                  sys.exit(1)
         except EOFError:
              console.print("[red]\nError: No prompt pasted.[/red]")
              sys.exit(1)

    dataset_h5ad_path, dataset_metadata = select_dataset()
    if not dataset_h5ad_path:
        console.print("[red]Dataset selection failed. Exiting.[/red]")
        sys.exit(1)

    while True:
        try:
            iterations_str = Prompt.ask("Enter number of evolution iterations", default="3")
            num_iterations = int(iterations_str)
            if num_iterations > 0: break
            else: console.print("[red]Number of iterations must be positive.[/red]")
        except ValueError: console.print("[red]Invalid input. Please enter an integer.[/red]")

    default_output = str(OUTPUTS_DIR.resolve())
    output_dir_str = Prompt.ask("Enter output directory for evolution logs", default=default_output)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Evolution logs will be saved in: [cyan]{output_dir.resolve()}[/cyan]")

    # --- Evolution Process ---
    current_prompt = initial_prompt
    evolution_log = [] # List to store results of each iteration

    for i in range(num_iterations):
        iteration_id = f"iteration_{i+1}"
        console.print(f"\n[bold blue]===== Starting Evolution Iteration {i+1}/{num_iterations} =====[/bold blue]")
        console.print(f"Current Prompt:\n---\n{current_prompt}\n---")

        # 2. Run Test with Current Prompt
        test_data = run_single_test_iteration(
            agent_prompt_id=iteration_id,
            agent_prompt=current_prompt,
            dataset_h5ad_path=dataset_h5ad_path,
            dataset_metadata=dataset_metadata,
            max_code_tries=5 # Or make this configurable
        )

        if test_data is None or test_data.get("error"):
             console.print(f"[bold red]Test iteration {i+1} failed. Skipping evaluation and evolution.[/bold red]")
             iteration_result = {
                 "iteration": i + 1,
                 "prompt": current_prompt,
                 "test_data": test_data, # Contains error info
                 "evaluation": None,
                 "evolved_prompt": None
             }
             evolution_log.append(iteration_result)
             # Decide whether to stop or continue with the same prompt
             if not Confirm.ask("Test failed. Continue evolution with the *same* prompt?", default=False):
                  break
             else:
                  continue # Try the same prompt again next iteration

        # 3. Evaluate the Result
        # Use the newly added function
        conversation_text = format_conversation_for_eval(test_data)
        evaluation = call_openai_evaluator(conversation_text, test_data.get("context", {}))
        console.print(f"Evaluation Grade: {evaluation.get('grade', 'N/A')}")
        console.print(f"Evaluation Comments:\n---\n{evaluation.get('comments', 'N/A')}\n---")

        # 4. Evolve the Prompt (unless it's the last iteration)
        evolved_prompt = None
        if i < num_iterations - 1:
            evolved_prompt = call_openai_evolver(
                objective=objective,
                previous_prompt=current_prompt,
                conversation_text=conversation_text, # Pass summary
                evaluation=evaluation
            )
            if evolved_prompt:
                console.print(f"Evolved Prompt for Next Iteration:\n---\n{evolved_prompt}\n---")
            else:
                console.print("[yellow]Failed to generate evolved prompt. Reusing previous prompt for next iteration.[/yellow]")
        else:
             console.print("Last iteration reached. No further prompt evolution.")


        # 5. Log Iteration Result
        iteration_result = {
            "iteration": i + 1,
            "prompt": current_prompt,
            "test_data": test_data, # Contains full conversation and context
            "evaluation": evaluation,
            "evolved_prompt": evolved_prompt # Will be None on last iteration or error
        }
        evolution_log.append(iteration_result)

        # Update prompt for next iteration if evolution was successful
        if evolved_prompt:
            current_prompt = evolved_prompt
        else:
             # If evolution failed, decide whether to continue with the same prompt
             if i < num_iterations - 1 and not Confirm.ask("Prompt evolution failed. Continue with the *same* prompt?", default=True):
                  break


    # --- Save Final Results ---
    console.print("\n[bold blue]===== Evolution Complete =====[/bold blue]")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_stem = dataset_h5ad_path.stem if dataset_h5ad_path else "unknown_dataset"
    final_log_filename = output_dir / f"evolution_log_{dataset_stem}_{timestamp}.json"
    final_prompt_filename = output_dir / f"final_prompt_{dataset_stem}_{timestamp}.txt"

    console.print(f"Saving evolution log to [cyan]{final_log_filename}[/cyan]...")
    try:
        with open(final_log_filename, "w", encoding="utf-8") as f:
            json.dump(evolution_log, f, indent=2, default=str)
        console.print("[green]Evolution log saved successfully.[/green]")
    except Exception as e:
        console.print(f"[bold red]Error saving evolution log: {e}[/bold red]")

    console.print(f"Saving final evolved prompt to [cyan]{final_prompt_filename}[/cyan]...")
    try:
        with open(final_prompt_filename, "w", encoding="utf-8") as f:
            f.write(current_prompt) # Save the last used prompt
        console.print("[green]Final prompt saved successfully.[/green]")
    except Exception as e:
        console.print(f"[bold red]Error saving final prompt: {e}[/bold red]")


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure output directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    main_evolution_loop()
