import json
import nbformat
import re
import base64
import sys
from pathlib import Path
from datetime import datetime

# --- Configuration ---
# Default directory to look for input files and save output files
DEFAULT_DIR = Path("./outputs")

# --- Helper Functions ---
def extract_python_code(text):
    """Extracts the first Python code block from text."""
    # Handle potential None input
    if text is None:
        return None, None
    # Regex to find code block and preceding/succeeding text
    # It captures text before, the code itself, and text after
    match = re.search(r"(.*?)```python\s*([\s\S]+?)\s*```(.*)", text, re.DOTALL)
    if match:
        text_before = match.group(1).strip()
        code = match.group(2).strip()
        text_after = match.group(3).strip()
        # Combine non-code text parts
        non_code_text = (text_before + "\n\n" + text_after).strip()
        return non_code_text, code
    else:
        # No code block found, return all text as non-code
        return text.strip(), None

def create_markdown_cell(source):
    """Creates a Markdown cell for the notebook."""
    # Ignore empty source strings
    if not source or not source.strip():
        return None
    return nbformat.v4.new_markdown_cell(source=source)

def create_code_cell(code, execution_count=None):
    """Creates a Code cell for the notebook."""
    if not code or not code.strip():
        return None
    return nbformat.v4.new_code_cell(source=code, execution_count=execution_count)

def format_outputs_for_notebook(api_outputs):
    """Converts the list of outputs from the API response into notebook cell outputs."""
    notebook_outputs = []
    if not api_outputs:
        return notebook_outputs

    for item in api_outputs:
        output_type = item.get("type")

        if output_type == "stream":
            notebook_outputs.append(nbformat.v4.new_output(
                output_type="stream",
                name=item.get("name", "stdout"), # Default to stdout if name missing
                text=item.get("text", "")
            ))
        elif output_type == "error":
            notebook_outputs.append(nbformat.v4.new_output(
                output_type="error",
                ename=item.get("ename", "Error"),
                evalue=item.get("evalue", ""),
                traceback=item.get("traceback", [])
            ))
        elif output_type == "execute_result":
            # Prefer text/plain, but include others if available
            data = item.get("data", {})
            if data: # Only add if data exists
                 notebook_outputs.append(nbformat.v4.new_output(
                    output_type="execute_result",
                    data=data, # Pass the whole data dict
                    metadata=item.get("metadata", {}),
                    execution_count=None # Typically not set on individual outputs
                 ))
        elif output_type == "display_data":
            # Handle potential base64 encoded image data
            data = item.get("data", {})
            processed_data = {}
            metadata = item.get("metadata", {}) # Include metadata for renderers

            for mime, content in data.items():
                 # Keep non-image data as is (e.g., text/html, text/plain)
                 if not mime.startswith("image/"):
                      processed_data[mime] = content
                 else:
                      # Assume image data might be base64 encoded string
                      # Notebook format expects base64 string directly for images
                      if isinstance(content, str):
                           # Basic check if it looks like base64, otherwise skip/warn
                           try:
                                # Test decode just to validate format, don't store decoded
                                base64.b64decode(content)
                                processed_data[mime] = content # Store the original base64 string
                           except (TypeError, ValueError):
                                print(f"Warning: Skipping display_data for mime '{mime}' - content is string but not valid base64.", file=sys.stderr)
                      else:
                           print(f"Warning: Skipping display_data for mime '{mime}' - unexpected data type '{type(content)}'.", file=sys.stderr)

            if processed_data: # Only add if data exists and was processed
                 notebook_outputs.append(nbformat.v4.new_output(
                     output_type="display_data",
                     data=processed_data,
                     metadata=metadata
                 ))
        # Ignore 'status' type messages for cell output
        elif output_type == "status":
            pass
        else:
            print(f"Warning: Unknown output type '{output_type}' encountered.", file=sys.stderr)

    return notebook_outputs

def create_notebook_cells(test_id, test_data):
    """
    Generates a list of notebook cells from a single test run's data.
    """
    cells = []
    execution_counter = 1 # Track execution count for code cells

    # --- Add Context Cell ---
    context = test_data.get("context", {})
    context_md = f"# Test Run: {test_id}\n\n"
    context_md += f"**Dataset:** {context.get('dataset_file', 'N/A')}\n"
    context_md += f"**Max Code Tries:** {context.get('max_code_tries', 'N/A')}\n"
    context_md += f"**Start Time:** {context.get('start_time', 'N/A')}\n\n"
    if context.get("dataset_metadata"):
        context_md += "## Dataset Metadata\n\n"
        context_md += "```json\n"
        context_md += json.dumps(context.get("dataset_metadata"), indent=2)
        context_md += "\n```\n"
    if context.get("error"): # Add setup error if present
         context_md += f"\n**SETUP/EXECUTION ERROR:**\n```\n{context.get('error')}\n```\n"

    cells.append(nbformat.v4.new_markdown_cell(context_md))

    # --- Process Conversation Turns ---
    turns = test_data.get("turns", [])
    i = 0
    while i < len(turns):
        turn = turns[i]
        role = turn.get("role")
        content = turn.get("content")

        if role == "system":
            # Could add system prompt as a collapsed cell, or skip
            # cells.append(create_markdown_cell(f"**SYSTEM PROMPT:**\n\n{content}"))
            pass # Often skipped in generated notebooks
        elif role == "user":
            # Check if this is a result message or an initial prompt
            if content and content.startswith("Code execution result:\n"):
                # This turn contains results, handled by the preceding assistant code cell
                pass # Skip placing result directly, it's an output of the code cell
            else:
                # This is an initial user prompt or follow-up question
                md_cell = create_markdown_cell(f"**USER PROMPT:**\n\n{content}")
                if md_cell: cells.append(md_cell)
        elif role == "assistant":
            text_part, code_part = extract_python_code(content)

            # Add markdown cell for the text explanation
            md_cell = create_markdown_cell(f"**ASSISTANT:**\n\n{text_part}")
            if md_cell: cells.append(md_cell)

            # Add code cell if code exists
            if code_part:
                code_cell = create_code_cell(code_part, execution_count=execution_counter)
                # Look ahead for the corresponding user result turn
                if i + 1 < len(turns) and turns[i+1].get("role") == "user" and turns[i+1].get("content", "").startswith("Code execution result:"):
                    api_response = turns[i+1].get("api_response", {})
                    api_outputs = api_response.get("outputs", [])
                    code_cell.outputs = format_outputs_for_notebook(api_outputs)
                    # Increment execution counter only if code was executed
                    execution_counter += 1
                    # Skip the next turn since we've processed it as output
                    i += 1
                else:
                     # Code was generated but no result followed? Add empty output.
                     code_cell.outputs = []
                     print(f"Warning: Assistant generated code but no result message followed turn {i}.", file=sys.stderr)

                cells.append(code_cell)

        i += 1 # Move to the next turn

    return cells


def convert_json_to_ipynb(json_path_str):
    """Loads the results JSON and converts it into a Jupyter Notebook."""
    json_path = Path(json_path_str)

    if not json_path.is_file():
        print(f"Error: Input file not found at '{json_path}'", file=sys.stderr)
        return

    # Determine output path
    output_path = json_path.with_suffix(".ipynb")
    print(f"Input JSON: {json_path}")
    print(f"Output Notebook: {output_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{json_path}': {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error reading file '{json_path}': {e}", file=sys.stderr)
        return

    # Create a new notebook object
    notebook = nbformat.v4.new_notebook()
    all_cells = []

    # Iterate through each test run in the results file
    for test_id, test_data in all_results.items():
        if not isinstance(test_data, dict):
             print(f"Warning: Skipping invalid data structure for test_id '{test_id}'. Expected a dictionary.", file=sys.stderr)
             continue
        print(f"Processing test run: {test_id}...")
        test_cells = create_notebook_cells(test_id, test_data)
        all_cells.extend(test_cells)

    notebook['cells'] = all_cells

    # Write the notebook to the output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        print(f"Successfully converted results to '{output_path}'")
    except Exception as e:
        print(f"Error writing notebook file '{output_path}': {e}", file=sys.stderr)


def interactive_loop():
    """Handles the interactive user prompts."""
    print("\n--- JSON to Jupyter Notebook Converter ---")
    print(f"Searches for JSON files in: {DEFAULT_DIR.resolve()}")
    print("Enter the full path to a results JSON file,")
    print("or just the filename if it's in the default directory.")
    print("Enter 'q' or press Enter to quit.")

    while True:
        try:
            input_path_str = input("\nEnter JSON file path/name (or q to quit): ").strip()

            if not input_path_str or input_path_str.lower() == 'q':
                print("Exiting.")
                break

            input_path = Path(input_path_str)

            # If only filename is given, prepend the default directory
            if not input_path.is_absolute() and not input_path.exists():
                potential_path = DEFAULT_DIR / input_path
                if potential_path.exists() and potential_path.is_file():
                    input_path = potential_path
                else:
                    # Try adding .json suffix if missing
                    potential_path_json = DEFAULT_DIR / input_path.with_suffix(".json")
                    if potential_path_json.exists() and potential_path_json.is_file():
                         input_path = potential_path_json
                    # Also check original input path with added suffix
                    elif input_path.with_suffix(".json").exists() and input_path.with_suffix(".json").is_file():
                         input_path = input_path.with_suffix(".json")


            if input_path.is_file() and input_path.suffix.lower() == ".json":
                convert_json_to_ipynb(input_path)
            else:
                print(f"Error: File not found or not a JSON file: '{input_path_str}'")
                print(f"(Checked paths: '{input_path}')")

        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc() # Print full traceback for debugging

# --- Main Execution ---
if __name__ == "__main__":
    # Create default output directory if it doesn't exist
    DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    interactive_loop()
