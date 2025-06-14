from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from typing import Optional
import re
import json
import sys
from pathlib import Path
from typing import Tuple, List
import textwrap
import base64
from datetime import datetime



def extract_python_code(txt: str) -> Optional[str]:
    """Return the *first* fenced code block, or None if absent.

    Handles:
    * ```python ... ```
    * ``` ... ``` (no language tag)
    * Leading indentation before fences (common in Markdown transcripts)
    """
    _FENCE_RE = re.compile(
        r'^[ \t]*```(?:python)?[ \t]*\n'   # opening fence, with optional "python"
        r'([\s\S]*?)'                     # capture all lines (including blank ones)
        r'^[ \t]*```[ \t]*$',             # closing fence
        re.MULTILINE
    )
    match = _FENCE_RE.search(txt)
    if not match:
        return None
    # Dedent to normalise indentation inside the block
    code = textwrap.dedent(match.group(1))
    return code.strip() or None

# Rich display wrappers

def _panel(console, role: str, content: str):
    titles = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}
    styles = {"system": "dim blue", "user": "cyan", "assistant": "green"}
    console.print(Panel(content, title=titles.get(role, role.upper()), border_style=styles.get(role, "white")))

def display(console, role: str, content: str):
    if "assistant" in role.lower():
        code = extract_python_code(content) or ""
        text_part = re.sub(r"```python[\s\S]+?```", "", content, count=1).strip()
        if text_part:
            _panel(console, "assistant", text_part)
        if code:
            console.print(
                Panel(
                    Syntax(code, "python", line_numbers=True),
                    title="ASSISTANT (code)",
                    border_style="green",
                )
            )
    else:
        _panel(console, role, content)

def select_dataset(console, dataset_dir) -> Tuple[Path, dict]:
    if not dataset_dir.exists():
        console.print(f"[red]Datasets dir not found: {dataset_dir}[/red]")
        sys.exit(1)
    items = [
        (p, json.loads(p.with_suffix(".json").read_text()))
        for p in dataset_dir.glob("*.h5ad")
        if p.with_suffix(".json").exists()
    ]
    if not items:
        console.print("[red]No datasets found.[/red]")
        sys.exit(1)
    tbl = Table(title="Datasets")
    tbl.add_column("Idx", justify="right")
    tbl.add_column("Name")
    tbl.add_column("Cells", justify="right")
    for i, (p, meta) in enumerate(items, 1):
        tbl.add_row(str(i), meta.get("dataset_title", p.stem), str(meta.get("cell_count", "?")))
    console.print(tbl)
    idx = int(Prompt.ask("Choose index", choices=[str(i) for i in range(1, len(items) + 1)])) - 1
    return items[idx]

def get_initial_prompt(console) -> str:
    console.print("[bold cyan]Enter the initial user prompt (Ctrl+D to finish):[/bold cyan]")
    try:
        txt = sys.stdin.read().strip()
    except EOFError:
        txt = ""
    if not txt:
        console.print("[red]Empty prompt – aborting.[/red]")
        sys.exit(1)
    return txt

def collect_resources(console, sandbox_sources_dir) -> List[Tuple[Path, str]]:
    console.print("\n[bold cyan]Optional: paths to bind inside sandbox[/bold cyan] (blank line to finish)")
    res: List[Tuple[Path, str]] = []
    while True:
        p = Prompt.ask("Path", default="").strip()
        if not p:
            break
        path = Path(p).expanduser().resolve()
        if not path.exists():
            console.print(f"[yellow]Path does not exist: {path}[/yellow]")
            continue
        res.append((path, f"{sandbox_sources_dir}/{path.name}"))
    return res


def format_execute_response(resp: dict, output_dir) -> str:
    lines = ["Code execution result:"]
    print(f"Response: {resp}")
    if resp.get("final_status") != "ok":
        lines.append(f"[status: {resp.get('status')}]")
    #if the key outputs in in resp we get the second dictionary
    if 'outputs' in resp:
        outputs = resp['outputs']
        resp = outputs[1]
    stdout, stderr, text = resp.get("stdout", ""), resp.get("stderr", ""), resp.get("text", "")
    error = False
    if resp.get("type") == "error":
        error = resp.get("evalue", "")
        traceback = resp.get("traceback", "")
        if traceback:
            error += "\n" + traceback
    if text and not error:
        lines += ["--- TEXT ---", text[:1500]]
    if stdout:
        lines += ["--- STDOUT ---", stdout[:1500]]
    if stderr:
        lines += ["--- STDERR ---", stderr[:1500]]
    if error:
        lines += ["--- ERROR ---", error[:1500]]
    img_paths = []
    for b64 in resp.get("images", []):
        fname = output_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        fname.parent.mkdir(exist_ok=True, parents=True)
        with open(fname, "wb") as f:
            f.write(base64.b64decode(b64))
        img_paths.append(str(fname))
    if img_paths:
        lines.append("Saved images: " + ", ".join(img_paths))
    return "\n".join(lines)