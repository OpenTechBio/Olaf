#!/usr/bin/env python3
"""
Offline Kernel (state‑preserving REPL or single‑shot)
====================================================
• **REPL mode**  : `python /opt/offline_kernel.py --repl`
  ‑ Parent writes a code chunk to stdin, terminates with the sentinel line
    `<<<EOF>>>`. Kernel executes it in a persistent global namespace, then
    prints **one** JSON line.
  ‑ Variables (e.g. `adata`) remain available in subsequent chunks.

• **Single‑shot**: `python /opt/offline_kernel.py <file.py>`
  ‑ Runs the file in a fresh namespace and exits (legacy behaviour).

Returned JSON schema
--------------------
{
  "status" : "ok" | "error" | "timeout",
  "stdout" : "captured standard output",
  "stderr" : "captured errors / traceback",
  "images" : [ "base64_png", ... ]  # Any matplotlib figures
}
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import List, Dict

# force to save to /tmp, so that the kernel can be run in a container
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/.numba_cache")
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/.config")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
os.environ.setdefault("CELLTYPIST_HOME", "/tmp/.celltypist")
os.environ.setdefault("CELLTYPIST_FOLDER", "/tmp/.celltypist_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/.transformers")

SENTINEL = "<<<EOF>>>"           # Delimits code blocks in REPL mode
GLOBAL_NS: Dict = {"__builtins__": __builtins__}  # Persistent namespace

# ---------------------------------------------------------------------------
# Core execution helper
# ---------------------------------------------------------------------------

def _run(code: str, ns: Dict) -> Dict:
    """Execute *code* in *ns* and capture stdout / stderr / matplotlib figs."""
    out, err = io.StringIO(), io.StringIO()
    images: List[str] = []
    status = "ok"

    try:
        with redirect_stdout(out), redirect_stderr(err):
            # Set up Matplotlib (optional)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt  # noqa: F401
                ns["plt"] = plt
            except ImportError:
                pass  # Matplotlib not installed; fine unless user imports it

            exec(compile(code, "<repl>", "exec"), ns)

            if "plt" in ns:
                for fid in ns["plt"].get_fignums():
                    fig = ns["plt"].figure(fid)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    images.append(base64.b64encode(buf.getvalue()).decode())
                    ns["plt"].close(fig)
    except Exception:
        err.write(traceback.format_exc())
        status = "error"
    finally:
        # Make sure no phantom figures linger
        if "plt" in ns:
            try:
                ns["plt"].close("all")
            except Exception:
                pass

    return {
        "status": status,
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
        "images": images,
    }

# ---------------------------------------------------------------------------
# REPL mode implementation
# ---------------------------------------------------------------------------

def _repl() -> None:
    """Persistent loop: read code chunks, exec, print JSON."""
    sys.stdout.write("__REPL_READY__\n")
    sys.stdout.flush()

    buffer: List[str] = []
    for line in sys.stdin:
        if line.rstrip("\n") == SENTINEL:
            code_block = "".join(buffer)
            buffer.clear()
            result = _run(code_block, GLOBAL_NS)
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
        else:
            buffer.append(line)

# ---------------------------------------------------------------------------
# Single‑shot helper (legacy)
# ---------------------------------------------------------------------------

def _single_shot(file_path: Path):
    if not file_path.is_file():
        print(json.dumps({
            "status": "error",
            "stdout": "",
            "stderr": f"File not found: {file_path}",
            "images": []
        }))
        sys.exit(3)
    try:
        code = file_path.read_text("utf‑8")
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "stdout": "",
            "stderr": f"Error reading {file_path}: {e}",
            "images": []
        }))
        sys.exit(4)

    print(json.dumps(_run(code, GLOBAL_NS)))

# ---------------------------------------------------------------------------
# Entry‑point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--repl":
        _repl()
        sys.exit(0)

    if len(sys.argv) != 2:
        print(
            "Usage: offline_kernel.py <file.py>  OR  offline_kernel.py --repl",
            file=sys.stderr,
        )
        sys.exit(2)

    _single_shot(Path(sys.argv[1]))
