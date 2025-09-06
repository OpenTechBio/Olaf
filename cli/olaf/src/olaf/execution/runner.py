# olaf/execution/runner.py
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

# --- Project-specific Imports ---
try:
    from olaf.config import OLAF_HOME
    from olaf.agents.AgentSystem import Agent, AgentSystem
    from olaf.core.io_helpers import display, extract_python_code, format_execute_response
    from olaf.rag.RetrievalAugmentedGeneration import RetrievalAugmentedGeneration
except ImportError as e:
    print(f"Failed to import a required OLAF module: {e}", file=sys.stderr)
    sys.exit(1)


# --- Type Hinting & Base Classes ---
class SandboxManager:
    """Abstract base class for sandbox interaction."""
    def start_container(self) -> bool:
        raise NotImplementedError

    def stop_container(self) -> None:
        raise NotImplementedError

    def exec_code(self, code: str, timeout: int) -> dict:
        raise NotImplementedError

# --- Constants and Path Setup ---
_DELEG_RE = re.compile(r"delegate_to_([A-Za-z0-9_]+)")
_OUTPUTS_DIR = OLAF_HOME / "runs"
_SNIPPET_DIR = _OUTPUTS_DIR / "snippets"
_LEDGER_PATH = _OUTPUTS_DIR / f"benchmark_history_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.jsonl"
_RAG_RE = re.compile(r"query_rag_([^>]+)")
RAG = RetrievalAugmentedGeneration()


def _init_paths():
    """Ensure output directories exist before writing."""
    _SNIPPET_DIR.mkdir(exist_ok=True, parents=True)
    _LEDGER_PATH.parent.mkdir(exist_ok=True, parents=True)

# --- Helper Functions (from original script) ---
def detect_delegation(msg: str) -> Optional[str]:
    """Return the *full* command name (e.g. 'delegate_to_coder') if present."""
    m = _DELEG_RE.search(msg)
    return f"delegate_to_{m.group(1)}" if m else None

def detect_rag(msg: str) -> Optional[str]:
    """Return the *partial* RAG command if present."""
    m = _RAG_RE.search(msg)
    return f"{m.group(1)}" if m else None

def _dump_code_snippet(run_id: str, code: str) -> str:
    """Write <run_id>.py under outputs/snippets/ and return the relative path."""
    snippet_path = _SNIPPET_DIR / f"{run_id}.py"
    snippet_path.write_text(code, encoding="utf-8")
    return str(snippet_path.relative_to(_OUTPUTS_DIR))

def _save_benchmark_record(*, run_id: str, results: dict, meta: dict, code: str | None):
    """Append a JSONL record for the benchmark run."""
    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run": run_id,
        "dataset": meta.get("name"),
        "results": results,
    }
    if code:
        record["code_path"] = _dump_code_snippet(run_id, code)
    with _LEDGER_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    
        
# --- Core Runner Functions ---
def run_benchmark(
    console: Console,
    mgr: SandboxManager,
    benchmark_module: Path,
    *,
    is_auto: bool,
    metadata: Optional[Dict] = None,
    agent_name: Optional[str] = None,
    code_snippet: Optional[str] = None,
) -> str:
    """
    Execute a benchmark module inside the sandbox.
    In auto mode, saves results and returns a result string for the history.
    In interactive mode, prints results to the console.
    """
    console.print(f"\n[bold cyan]Running benchmark module: {benchmark_module.name}[/bold cyan]")
    autometric_base_path = benchmark_module.parent / "AutoMetric.py"
    try:
        with open(autometric_base_path, "r") as f:
            autometric_code = f.read()
        with open(benchmark_module, "r") as f:
            benchmark_code = f.read()
    except FileNotFoundError as e:
        err = f"Benchmark module or AutoMetric.py not found: {e}"
        console.print(f"[red]{err}[/red]")
        return err if is_auto else ""

    code_to_execute = f"# --- Code from AutoMetric.py ---\n{autometric_code}\n# --- Code from {benchmark_module.name} ---\n{benchmark_code}"
    console.print("[cyan]Executing benchmark code...[/cyan]")
    
    try:
        exec_result = mgr.exec_code(code_to_execute, timeout=300)

        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        stdout = exec_result.get("stdout", "")
        result_dict = {}
        try:
            result_dict = json.loads(stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as e:
            console.print(f"[yellow]Warning: Could not parse JSON from stdout: {e}[/yellow]")

        if exec_result.get("status") == "ok" and isinstance(result_dict, dict):
            for key, value in result_dict.items():
                table.add_row(str(key), str(value))
            if is_auto:
                _save_benchmark_record(
                    run_id=f"{benchmark_module.stem}:{agent_name}:{int(time.time())}",
                    results=result_dict,
                    meta=metadata if metadata else {},
                    code=code_snippet,
                )
        else:
            error_message = exec_result.get("stderr") or "An unknown error occurred."
            table.add_row("Error", error_message)
        
        console.print(table)
        return "Benchmark results:\n" + json.dumps(result_dict or {"error": "see console logs"})
    
    except Exception as exc:
        err_msg = f"Benchmark execution failed: {exc}"
        console.print(f"[red]{err_msg}[/red]")
        return err_msg

def run_agent_session(
    *,
    console: Console,
    agent_system: AgentSystem,
    driver_agent: Agent,
    roster_instructions: str,
    analysis_context: str,
    llm_client: object,
    sandbox_manager: SandboxManager,
    history: List[Dict[str, str]],
    is_auto: bool,
    max_turns: int = 1,
    benchmark_modules: Optional[List[Path]] = None,
):
    """
    Main driver for both interactive and automated agent execution sessions.
    """
    from rich.prompt import Prompt
    _init_paths()
    
    # --- Display the initial context provided by the CLI ---
    for message in history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if role in ["system", "user"]:
            display(console, role, content)
            
    current_agent = driver_agent
    turn = 0
    last_code_snippet: str | None = None

    while True:
        turn += 1
        if is_auto and turn > max_turns:
            console.print("[bold green]Auto run finished: Max turns reached.[/bold green]")
            break

        console.print(f"\n[bold]LLM call (turn {turn})…[/bold]")
        
        try:
            resp = llm_client.chat.completions.create(
                model="gpt-4o",
                messages=history,
                temperature=0.7,
            )
            msg = resp.choices[0].message.content
        except Exception as e:
            console.print(f"[red]LLM API error: {e}[/red]")
            break
        
        history.append({"role": "assistant", "content": msg})
        display(console, f"assistant ({current_agent.name})", msg)  

        # --- RAG handling ---
        query_from_re = detect_rag(msg)
        if query_from_re and current_agent.is_rag_enabled:
            console.print(f"[yellow]🔍 Triggering RAG query: {query_from_re}[/yellow]")
            retrieved_docs = RAG.query(query_from_re)
            if retrieved_docs:
                console.print(f"[green] RAG query successful. [/green]")
                feedback = retrieved_docs
                console.print(feedback)
                history.append({"role": "system", "content": feedback}) 
            else:
                console.print(f"[red] RAG query unsuccessful. [/red]")
            

        cmd = detect_delegation(msg)
        if cmd and cmd in current_agent.commands:
            target_agent_name = current_agent.commands[cmd].target_agent
            new_agent = agent_system.get_agent(target_agent_name)
            if new_agent:
                console.print(f"[yellow]🔄 Routing to '{target_agent_name}' via {cmd}[/yellow]")
                history.append({"role": "assistant", "content": f"🔄 Routing to **{target_agent_name}** (command `{cmd}`)"})
                current_agent = new_agent
                system_prompt = (roster_instructions + "\n\n" + current_agent.get_full_prompt(agent_system.global_policy) + "\n\n" + analysis_context)
                # We replace the last system prompt with the new one for the new agent
                history.insert(0, {"role": "system", "content": system_prompt})
                # Remove the old system prompt to avoid confusion
                if len(history) > 1 and history[1].get("role") == "system":
                    history.pop(1)
                continue

        code = extract_python_code(msg)
        if code:
            last_code_snippet = code
            console.print("[cyan]Executing code in sandbox…[/cyan]")
            exec_result = sandbox_manager.exec_code(code, timeout=300)
            feedback = format_execute_response(exec_result, _OUTPUTS_DIR)
            history.append({"role": "assistant", "content": feedback})
            display(console, "assistant", feedback)

            stderr = exec_result.get('stderr', '')
            if stderr and current_agent.is_rag_enabled:
                func_error_patterns = [
                r"(\w+)\(.*\) missing \d+ required positional argument", # TypeError missing arguments
                r"NameError: name '(\w+)' is not defined",             # NameError
                r"AttributeError: .* has no attribute '(\w+)'",       # AttributeError
                r"'(\w+)\(.*\) got an unexpected keyword argument"         # Unexpected keyword argument
            ]
                
                function_name = ""
                retrieved_docs = ""
                
                for pat in func_error_patterns:
                    match = re.search(pat, stderr)
                    if match:
                        function_name = [g for g in match.groups() if g]
                        break
                            
                if function_name:
                    function_name = function_name[0]
                    console.print(f"[yellow]🔍 Incorrect function signature detected: {function_name}, function database search...[/yellow]")
                    retrieved_docs = RAG.retrieve_function(function_name)
                    if retrieved_docs:
                        console.print(f"[green] Query successful - Function signature found. [/green]")
                        feedback += f"\n {function_name} produced an error. The correct function signature for {function_name} is:\n{retrieved_docs}"
                        history.append({"role": "system", "content": feedback})
                        continue
                    else:
                        print(f"Error Query unsuccessful - Function signature does not exist in the current database.")
                 

        if is_auto:
            if benchmark_modules:
                result_str = run_benchmark(
                    console, sandbox_manager, benchmark_modules[0],
                    is_auto=True, metadata={"name": "auto"}, agent_name=current_agent.name, code_snippet=last_code_snippet
                )
                history.append({"role": "user", "content": result_str})
                display(console, "user", result_str)
            console.print(f"[yellow]Auto-continuing... {turn}/{max_turns} turns complete.[/yellow]")
        else:
            while True:
                prompt_text = "\n[bold]Next message ('benchmark' to run selected benchmark, 'exit' to quit)[/bold]"
                try:
                    user_input = Prompt.ask(prompt_text, default="").strip()
                except (EOFError, KeyboardInterrupt):
                    user_input = "exit"

                if user_input.lower() in {"exit", "quit"}:
                    console.print("[bold yellow]Exiting session.[/bold yellow]")
                    return

                if user_input.lower() == "benchmark":
                    if benchmark_modules:
                        for bm_module in benchmark_modules:
                            run_benchmark(console, sandbox_manager, bm_module, is_auto=False)
                        continue
                    else:
                        console.print("[yellow]No benchmark modules were specified at startup.[/yellow]")
                        continue
                
                if user_input:
                    history.append({"role": "user", "content": user_input})
                    display(console, "user", user_input)
                break
