import time
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json

from benchmarking.sandbox.benchmarking_sandbox_management import (
    SandboxManager as _BackendManager,
    CONTAINER_NAME as _SANDBOX_HANDLE,
    IMAGE_TAG as _SANDBOX_IMAGE,  
    API_PORT_HOST as _API_PORT,
)


def init_docker(script_dir:str, subprocess, console, force_refresh:bool=False):
    sandbox_dir = script_dir / "workspace"
    # --- optional force‑refresh logic --------------------------------------
    if force_refresh:
        console.print("[yellow]Forcing Docker sandbox refresh…[/yellow]")
        # Stop & remove any running container gracefully
        subprocess.run(["docker", "rm", "-f", _SANDBOX_HANDLE], check=False)
        # Remove the sandbox image to ensure re‑pull/build
        subprocess.run(["docker", "image", "rm", "-f", _SANDBOX_IMAGE], check=False)
        console.print("[green]Docker image removed – it will be pulled/built on next start.[/green]")

    def COPY_CMD(src: str, dst: str):
        subprocess.run(["docker", "cp", src, dst], check=True)
    
    # create sandbox directory in docker 
    EXECUTE_ENDPOINT = f"http://localhost:{_API_PORT}/execute"
    STATUS_ENDPOINT = f"http://localhost:{_API_PORT}/status"

    return _BackendManager, _SANDBOX_HANDLE, COPY_CMD, EXECUTE_ENDPOINT, STATUS_ENDPOINT

def init_singularity(script_dir:str, subprocess, console, force_refresh:bool=False):
    import benchmarking.sandbox.benchmarking_sandbox_management_singularity as sing
    sandbox_dir = script_dir / "sandbox"

    # optional force‑refresh
    if force_refresh:
        console.print("[yellow]Forcing Singularity sandbox refresh…[/yellow]")
        try:
            sing.stop_instance()
        except Exception:
            pass  # ignore if not running
        if sing.SIF_PATH.exists():
            sing.SIF_PATH.unlink()
            console.print(
                f"[green]Deleted {sing.SIF_PATH.name} – it will be re‑downloaded on next start.[/green]"
            )

    class _SingInstanceWrapper:
        def start_container(self):
            return sing.start_instance()

        def stop_container(self):
            return sing.stop_instance()

    _BackendManager = _SingInstanceWrapper
    _SANDBOX_HANDLE = sing.INSTANCE_NAME
    _API_PORT = sing.API_PORT_HOST

    def COPY_CMD(src: str, dst: str):
        console.print(
            f"[yellow]Singularity instance: ensure {src} is reachable at {dst} via bind mount.[/yellow]"
        )

    EXECUTE_ENDPOINT = f"http://localhost:{_API_PORT}/execute"
    STATUS_ENDPOINT = f"http://localhost:{_API_PORT}/status"

    return _BackendManager, _SANDBOX_HANDLE, COPY_CMD, EXECUTE_ENDPOINT, STATUS_ENDPOINT



def init_singularity_exec(script_dir: str, sanbox_data_path, subprocess, console, force_refresh: bool = False):
    import benchmarking.sandbox.benchmarking_sandbox_management_singularity as sing
    sandbox_dir = script_dir / "sandbox"

    # optional force‑refresh
    if force_refresh:
        console.print("[yellow]Forcing Singularity sandbox refresh…[/yellow]")
        if sing.SIF_PATH.exists():
            sing.SIF_PATH.unlink()
            console.print(
                f"[green]Deleted {sing.SIF_PATH.name} – it will be re‑downloaded on next start.[/green]"
            )

    SIF_PATH = sing.SIF_PATH
    SING_BIN = sing.SING_BIN
    SENTINEL = "<<<EOF>>>"

    class _SingExecBackend:
        """Launch one long‑lived REPL inside the SIF and stream code to it."""

        def __init__(self):
            self._binds: List[str] = []
            self._proc = None

        def set_data(self, dataset: Path, resources: List[Tuple[Path, str]]):
            self._binds = [
                "--bind",
                f"{dataset.resolve()}:{sanbox_data_path}",
            ]
            for host, cont in resources:
                self._binds.extend(["--bind", f"{host.resolve()}:{cont}"])

        # ------------------------------------------------------------------
        # Container lifecycle
        # ------------------------------------------------------------------
        def start_container(self):
            if self._proc:
                return True  # already running
            if not sing.pull_sif_if_needed():
                return False

            cmd = [
                SING_BIN,
                "exec",
                "--containall",
                "--cleanenv",
                *self._binds,
                str(SIF_PATH),
                "python",
                "/opt/offline_kernel.py",
                "--repl",
            ]
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line buffered
            )
            # Wait for the REPL banner
            ready_line = self._proc.stdout.readline().strip()
            if ready_line != "__REPL_READY__":
                console.print(
                    f"[red]REPL failed to start. Got: {ready_line}[/red]"
                )
                self.stop_container()
                return False
            return True

        def stop_container(self):
            if not self._proc:
                return True
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None
            return True

        # ------------------------------------------------------------------
        # Code execution
        # ------------------------------------------------------------------
        def exec_code(self, code: str, timeout: int = 300) -> Dict:
            if not self._proc:
                raise RuntimeError("REPL not running")
            assert self._proc.stdin and self._proc.stdout

            # Send code block + sentinel
            self._proc.stdin.write(code)
            if not code.endswith("\n"):
                self._proc.stdin.write("\n")
            self._proc.stdin.write(SENTINEL + "\n")
            self._proc.stdin.flush()

            # Read exactly one JSON line
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    return {
                        "status": "timeout",
                        "stdout": "",
                        "stderr": "Execution timed out in REPL.",
                        "images": [],
                    }
                line = self._proc.stdout.readline()
                if not line:
                    continue
                line = line.strip()
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    # Non‑JSON noise; continue reading
                    continue

    _BackendManager = _SingExecBackend

    def COPY_CMD(src: str, dst: str):
        console.print("[yellow]singularity-exec mode uses bind mounts instead of docker cp.[/yellow]")
    
    return _BackendManager, None, COPY_CMD, None, None
    
    
    
