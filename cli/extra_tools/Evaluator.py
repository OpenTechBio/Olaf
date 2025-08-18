import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import re

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

# Optional: Use rich for better formatting
try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
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


# --- Constants ---
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" # Default to save back into input dir
ENV_FILE = SCRIPT_DIR / ".env"

# --- Configuration Loading ---
load_dotenv(dotenv_path=ENV_FILE)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o" # Or your preferred model for evaluation

if not OPENAI_API_KEY:
    if console: console.print(f"[bold red]Error:[/bold red] OPENAI_API_KEY not found in {ENV_FILE}.")
    else: print(f"Error: OPENAI_API_KEY not found in {ENV_FILE}.")
    sys.exit(1)

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    if console: console.print(f"OpenAI client initialized for model [cyan]{OPENAI_MODEL}[/cyan].")
    else: print(f"OpenAI client initialized for model {OPENAI_MODEL}.")
except Exception as e:
    if console: console.print(f"[bold red]Error initializing OpenAI client:[/bold red] {e}")
    else: print(f"Error initializing OpenAI client: {e}")
    sys.exit(1)

# --- Helper Functions ---

def format_conversation_for_eval(test_data):
    """ Formats the conversation turns into a readable string for the evaluator prompt. """
    if not test_data or "turns" not in test_data:
        return "[No conversation turns found]"

    formatted_lines = []
    for turn in test_data.get("turns", []):
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "[No content]")

        # Shorten system prompt for brevity in evaluation context if desired
        if role == "SYSTEM":
             # Extract key parts or just indicate system prompt presence
             content = "[System Prompt Provided - see original log for details]"
             # Or keep it: content = turn.get("content", "[No content]")

        # Format code execution results more clearly if they are part of user turn
        if role == "USER" and content.startswith("Code execution result:"):
             # Reformat slightly for clarity
             content = content.replace("Code execution result:", "**CODE EXECUTION RESULT:**")
             content = content.replace("--- STDOUT ---", "**STDOUT:**")
             content = content.replace("--- STDERR ---", "**STDERR:**")
             content = content.replace("--------------", "---") # Shorten separator

        formatted_lines.append(f"--- {role} ---")
        formatted_lines.append(content)
        formatted_lines.append("\n") # Add space between turns

    return "\n".join(formatted_lines)


def call_openai_evaluator(conversation_text, context):
    """ Sends the formatted conversation to OpenAI for evaluation. """
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
1.  **Correctness:** Was the generated code correct and did it achieve the intended analysis steps?
2.  **Efficiency:** Was the approach reasonable? Were there unnecessary steps?
3.  **Interpretation:** Did the assistant correctly interpret the results of its code execution?
4.  **Planning:** Did the assistant use its allowed code execution attempts effectively towards the goal?
5.  **Clarity:** Was the assistant's text explanation clear and accurate?
6.  **Overall Skill:** Does the performance align with an entry-level post-graduate bioinformatician?

**Output Format:**
Please provide your evaluation strictly in the following JSON format ONLY. Do not include any other text before or after the JSON block:
{{
  "grade": <integer between 0 and 100>,
  "comments": "<string containing your detailed evaluation justifying the grade>"
}}
"""

    if console: console.print(f"Sending evaluation request for context: {context.get('prompt_id', 'unknown')[:20]}...")
    else: print(f"Sending evaluation request for context: {context.get('prompt_id', 'unknown')[:20]}...")

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                # Maybe a short system message for the evaluator role itself?
                # {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": evaluator_prompt}
            ],
            temperature=0.3, # Lower temperature for more deterministic evaluation
            response_format={"type": "json_object"}, # Request JSON output
            max_tokens=1000 # Adjust as needed for comments length
        )
        eval_content = response.choices[0].message.content
        if console: console.print("[green]Evaluation received from OpenAI.[/green]")
        else: print("Evaluation received from OpenAI.")

        # Attempt to parse the JSON response
        try:
            eval_json = json.loads(eval_content)
            # Validate expected keys
            if "grade" in eval_json and "comments" in eval_json:
                # Basic type check (can be more robust)
                if isinstance(eval_json["grade"], int) and isinstance(eval_json["comments"], str):
                     return eval_json
                else:
                     raise ValueError("Incorrect data types for 'grade' or 'comments'.")
            else:
                 raise ValueError("Missing 'grade' or 'comments' key in JSON response.")
        except (json.JSONDecodeError, ValueError) as e:
            if console: console.print(f"[bold red]Error parsing evaluation JSON from OpenAI: {e}[/bold red]")
            else: print(f"Error parsing evaluation JSON from OpenAI: {e}")
            if console: console.print(f"Raw response content:\n{eval_content}")
            else: print(f"Raw response content:\n{eval_content}")
            # Return a structured error
            return {"grade": -1, "comments": f"Error parsing OpenAI response: {e}\nRaw Content: {eval_content}"}

    except APIError as e:
        if console: console.print(f"[bold red]OpenAI API Error during evaluation: {e}[/bold red]")
        else: print(f"OpenAI API Error during evaluation: {e}")
        return {"grade": -1, "comments": f"OpenAI API Error: {e}"}
    except Exception as e:
        if console: console.print(f"[bold red]Unexpected error during evaluation call: {e}[/bold red]")
        else: print(f"Unexpected error during evaluation call: {e}")
        import traceback
        traceback.print_exc()
        return {"grade": -1, "comments": f"Unexpected Error: {e}"}


def process_folder(input_dir_path, output_path):
    """Finds JSON files, gets evaluations, and saves them."""
    evaluations = {}
    json_files = list(input_dir_path.glob("*.json"))

    if not json_files:
        if console: console.print(f"[yellow]No JSON files found in '{input_dir_path}'.[/yellow]")
        else: print(f"No JSON files found in '{input_dir_path}'.")
        return

    if console: console.print(f"Found {len(json_files)} JSON file(s) to evaluate.")
    else: print(f"Found {len(json_files)} JSON file(s) to evaluate.")

    for json_file in json_files:
        if console: console.print(f"\n--- Processing: [cyan]{json_file.name}[/cyan] ---")
        else: print(f"\n--- Processing: {json_file.name} ---")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)

            # Process each test run within the file (assuming structure {test_id: test_data})
            file_evaluations = {}
            for test_id, test_data in results_data.items():
                if not isinstance(test_data, dict) or "context" not in test_data or "turns" not in test_data:
                    if console: console.print(f"[yellow]Skipping invalid/incomplete data for test ID '{test_id}' in {json_file.name}.[/yellow]")
                    else: print(f"Skipping invalid/incomplete data for test ID '{test_id}' in {json_file.name}.")
                    continue

                conversation_text = format_conversation_for_eval(test_data)
                context = test_data.get("context", {})
                evaluation = call_openai_evaluator(conversation_text, context)
                file_evaluations[test_id] = evaluation # Store evaluation keyed by test_id

            # Store evaluations for this file, keyed by the original filename stem
            evaluations[json_file.stem] = file_evaluations

        except json.JSONDecodeError:
            if console: console.print(f"[red]Error decoding JSON from {json_file.name}. Skipping.[/red]")
            else: print(f"Error decoding JSON from {json_file.name}. Skipping.")
        except Exception as e:
            if console: console.print(f"[red]Error processing file {json_file.name}: {e}[/red]")
            else: print(f"Error processing file {json_file.name}: {e}")

    # --- Save Evaluations ---
    if not evaluations:
        if console: console.print("[yellow]No evaluations were generated.[/yellow]")
        else: print("No evaluations were generated.")
        return

    output_path = Path(output_path) # Ensure it's a Path object

    # Check if output is a directory or file
    if output_path.suffix == ".json":
        # Save all evaluations to a single specified file
        output_filename = output_path
        if console: console.print(f"\nSaving all evaluations to single file: [cyan]{output_filename}[/cyan]")
        else: print(f"\nSaving all evaluations to single file: {output_filename}")
        try:
             output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists
             with open(output_filename, "w", encoding="utf-8") as f:
                 json.dump(evaluations, f, indent=2)
             if console: console.print("[green]Evaluations saved successfully.[/green]")
             else: print("Evaluations saved successfully.")
        except Exception as e:
             if console: console.print(f"[bold red]Error saving aggregated evaluations to {output_filename}:[/bold red] {e}")
             else: print(f"Error saving aggregated evaluations to {output_filename}: {e}")
    else:
        # Save evaluations to individual files in the specified directory
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        if console: console.print(f"\nSaving evaluations to directory: [cyan]{output_dir}[/cyan]")
        else: print(f"\nSaving evaluations to directory: {output_dir}")
        for input_stem, file_evals in evaluations.items():
            output_filename = output_dir / f"{input_stem}_eval.json"
            try:
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(file_evals, f, indent=2)
                if console: console.print(f"  Saved: [green]{output_filename.name}[/green]")
                else: print(f"  Saved: {output_filename.name}")
            except Exception as e:
                if console: console.print(f"  [red]Error saving evaluation for {input_stem}: {e}[/red]")
                else: print(f"  Error saving evaluation for {input_stem}: {e}")


def interactive_loop():
    """Handles the interactive user prompts."""
    if console: console.print("\n--- Agent Benchmark Evaluator ---")
    else: print("\n--- Agent Benchmark Evaluator ---")

    # Get input directory
    default_input = str(DEFAULT_INPUT_DIR.resolve())
    while True:
        if console: input_dir_str = Prompt.ask("Enter path to input folder containing results JSONs", default=default_input)
        else: input_dir_str = input(f"Enter path to input folder containing results JSONs [{default_input}]: ").strip() or default_input

        input_dir_path = Path(input_dir_str).resolve()
        if input_dir_path.is_dir():
            break
        else:
            if console: console.print(f"[red]Error: Input path '{input_dir_path}' is not a valid directory.[/red]")
            else: print(f"Error: Input path '{input_dir_path}' is not a valid directory.")

    # Get output path (directory or specific file)
    default_output = str(input_dir_path) # Default output to input dir
    if console: output_path_str = Prompt.ask("Enter output directory or specific .json filename for results", default=default_output)
    else: output_path_str = input(f"Enter output directory or specific .json filename for results [{default_output}]: ").strip() or default_output

    process_folder(input_dir_path, output_path_str)


# --- Main Execution ---
if __name__ == "__main__":
    interactive_loop()
