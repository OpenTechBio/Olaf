#!/usr/bin/env python3
"""
Singularity Sandbox Manager (Docker-free)
========================================
Pure-Singularity version that does not require Docker. Pulls a pre-built
`sandbox.sif` from a specified URL when necessary.

Supports:
1. Networked mode: starts a Singularity instance with a FastAPI kernel.
2. Offline mode: provides SIF management (`pull_sif_if_needed`) and the
   `SIF_PATH` / `SING_BIN` constants for the `singularity-exec` backend
   in InteractiveAgentTester.py.

For offline mode to function, the `sandbox.sif` image must contain
`/opt/offline_kernel.py`. Ensure it’s included during the SIF build.

Commands:
    build   – download/update `sandbox.sif`
    start   – start an instance exposing the FastAPI kernel on port 8000
    stop    – stop & remove the instance
    status  – show instance + port status
    logs    – tail the instance log (default 50 lines)

Run with no args for an interactive REPL.
"""
from __future__ import annotations

import argparse
import getpass
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Paths & constants (for InteractiveAgentTester.py) ─────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SIF_PATH = SCRIPT_DIR / "sandbox.sif"
CONTAINER_URL = "https://huggingface.co/datasets/djriffle/olaf_sandbox_files/resolve/main/sandbox.sif?download=true"

INSTANCE_NAME = "benchmarking_sandbox_instance"
API_PORT_INSIDE = 8000
API_PORT_HOST = 8000

SING_BIN = shutil.which("apptainer") or shutil.which("singularity")
if not SING_BIN:
    print(
        "Singularity/Apptainer executable not found in PATH. "
        "Do you need to load a module?",
        file=sys.stderr,
    )
    sys.exit(1)


def run(cmd: list[str], *, capture: bool = False, check: bool = True, timeout: int | None = None):
    logging.debug("$ %s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, text=True, capture_output=capture, check=check, timeout=timeout)


def pull_sif_if_needed(force_pull: bool = False) -> bool:
    """
    Pull sandbox.sif from the predefined URL if it doesn't exist or if force_pull is true.
    """
    if SIF_PATH.exists() and not force_pull:
        logging.info("Using existing SIF: %s", SIF_PATH)
        return True

    if SIF_PATH.exists() and force_pull:
        logging.info("Force pull enabled, removing existing SIF: %s", SIF_PATH)
        try:
            SIF_PATH.unlink()
        except OSError as e:
            logging.error("Failed to remove existing SIF %s: %s", SIF_PATH, e)
            return False

    logging.info(
        "Pulling %s from %s … (SIF must contain /opt/offline_kernel.py)",
        SIF_PATH,
        CONTAINER_URL,
    )
    cmd = [SING_BIN, "pull", "--force", str(SIF_PATH), CONTAINER_URL]
    try:
        run(cmd)
        if not SIF_PATH.exists() or SIF_PATH.stat().st_size == 0:
            logging.error("Pull succeeded but SIF is missing or empty.")
            return False
        logging.info("Pull finished ✓. SIF is at %s", SIF_PATH)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("Singularity pull failed (code %s)", e.returncode)
        if e.stderr:
            logging.error("Stderr:\n%s", e.stderr.strip())
        if e.stdout:
            logging.error("Stdout:\n%s", e.stdout.strip())
        return False
    except Exception as e:
        logging.error("Unexpected error during pull: %s", e)
        return False


def instance_running() -> bool:
    try:
        result = run([SING_BIN, "instance", "list"], capture=True, check=False)
        return INSTANCE_NAME in result.stdout
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        logging.error("Singularity/Apptainer executable not found.")
        return False


def start_instance(rebuild: bool = False) -> bool:
    """
    Start a Singularity instance for FastAPI access (networked mode).
    `rebuild=True` forces re-pulling the SIF.
    """
    if instance_running():
        logging.warning("Instance already running – restarting…")
        if not stop_instance():
            logging.error("Could not stop existing instance.")
            return False

    if not pull_sif_if_needed(force_pull=rebuild):
        logging.error("Cannot ensure SIF image. Aborting start.")
        return False

    logging.info("Starting instance %s from %s …", INSTANCE_NAME, SIF_PATH)
    cmd = [
        SING_BIN,
        "instance",
        "start",
        "--cleanenv",
        "--net",
        "--network-args",
        f"portmap={API_PORT_HOST}:tcp:{API_PORT_INSIDE}",
        str(SIF_PATH),
        INSTANCE_NAME,
    ]
    try:
        run(cmd)
        logging.info("Start command sent. Waiting for instance…")
        time.sleep(3)
        if instance_running():
            logging.info("Instance '%s' is running at http://localhost:%d", INSTANCE_NAME, API_PORT_HOST)
            return True
        logging.error("Instance failed to start.")
        _log_last_lines()
        return False
    except subprocess.CalledProcessError as e:
        logging.error("Failed to start instance: %s", e)
        if e.stderr:
            logging.error("Stderr:\n%s", e.stderr.strip())
        if e.stdout:
            logging.error("Stdout:\n%s", e.stdout.strip())
        return False
    except Exception as e:
        logging.error("Unexpected error starting instance: %s", e)
        return False


def stop_instance() -> bool:
    """
    Stop the Singularity instance (networked mode).
    """
    if not instance_running():
        logging.info("Instance '%s' not running.", INSTANCE_NAME)
        return True

    logging.info("Stopping instance %s …", INSTANCE_NAME)
    try:
        run([SING_BIN, "instance", "stop", INSTANCE_NAME])
        if not instance_running():
            logging.info("Instance '%s' stopped.", INSTANCE_NAME)
            return True
        logging.warning("Stop command executed, but instance still listed.")
        return False
    except subprocess.CalledProcessError as e:
        logging.error("Failed to stop instance: %s", e)
        if e.stderr:
            logging.error("Stderr:\n%s", e.stderr.strip())
        if e.stdout:
            logging.error("Stdout:\n%s", e.stdout.strip())
        return False
    except Exception as e:
        logging.error("Unexpected error stopping instance: %s", e)
        return False


def show_status():
    """
    Display status of the networked instance and existence of SIF.
    """
    is_running = instance_running()
    status = "RUNNING" if is_running else "STOPPED"
    logging.info("Instance '%s' status: %s", INSTANCE_NAME, status)
    if is_running:
        logging.info("API: http://localhost:%d", API_PORT_HOST)
    else:
        logging.info("API port (if running): %d", API_PORT_HOST)

    logging.info("SIF image path: %s", SIF_PATH)
    if not SIF_PATH.exists():
        logging.warning("SIF image missing. Run 'build' command.")


def show_logs(lines: int = 50):
    """
    Tail the last `lines` lines of the instance log (networked mode).
    """
    user_name = os.getenv("USER") or getpass.getuser()
    apptainer_dir = Path.home() / ".apptainer" / "instances" / "logs" / user_name
    singularity_dir = Path.home() / ".singularity" / "instances" / "logs" / user_name

    log_file = None
    if (apptainer_dir / f"{INSTANCE_NAME}.log").exists():
        log_file = apptainer_dir / f"{INSTANCE_NAME}.log"
    elif (singularity_dir / f"{INSTANCE_NAME}.log").exists():
        log_file = singularity_dir / f"{INSTANCE_NAME}.log"

    if not instance_running() and not log_file:
        logging.warning("Instance not running and no log file found.")
        return
    if not log_file:
        logging.warning(
            "Instance is running, but log file not found:\n"
            f"  - {apptainer_dir}\n  - {singularity_dir}"
        )
        return

    if not instance_running():
        logging.info("Instance '%s' not running. Displaying last logs from %s", INSTANCE_NAME, log_file)

    print(f"\n--- Logs for {INSTANCE_NAME} (last {lines} lines) ---")
    try:
        result = run(["tail", "-n", str(lines), str(log_file)], capture=True, check=True)
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logging.error("Could not read logs: %s", e)
        if e.stderr:
            logging.error("Stderr:\n%s", e.stderr.strip())
    except FileNotFoundError:
        logging.error("'tail' command not found.")
    print("-------------------------------------")


def _log_last_lines():
    """
    Helper to print the last 20 lines of the instance log if it exists.
    """
    user_name = os.getenv("USER") or getpass.getuser()
    apptainer_dir = Path.home() / ".apptainer" / "instances" / "logs" / user_name
    singularity_dir = Path.home() / ".singularity" / "instances" / "logs" / user_name

    log_file = None
    if (apptainer_dir / f"{INSTANCE_NAME}.log").exists():
        log_file = apptainer_dir / f"{INSTANCE_NAME}.log"
    elif (singularity_dir / f"{INSTANCE_NAME}.log").exists():
        log_file = singularity_dir / f"{INSTANCE_NAME}.log"

    if log_file:
        logging.error("Check instance logs: %s", log_file)
        try:
            with open(log_file, "r") as lf:
                lines = lf.readlines()[-20:]
            logging.error("Last 20 log lines:\n%s", "".join(lines))
        except Exception as e:
            logging.error("Could not read log file: %s", e)
    else:
        logging.error("Instance log file not found.")


def repl():
    """
    Interactive REPL for building, starting, stopping, status, logs.
    """
    print("Singularity Sandbox Manager (Docker-free). Type 'help'.")

    while True:
        try:
            line = input(f"{INSTANCE_NAME}> ").strip()
        except EOFError:
            print("\nExiting.")
            break
        if not line:
            continue

        try:
            parts = shlex.split(line)
            cmd = parts[0]
            args = parts[1:]
        except ValueError as e:
            print(f"Error parsing command: {e}")
            continue

        if cmd in {"exit", "quit", "q"}:
            print("Exiting.")
            break
        elif cmd == "help":
            print(
                "\nAvailable commands:\n"
                "  build [--rebuild] - Download/update SIF (offline + networked).\n"
                "  start [--rebuild] - Start networked instance (FastAPI).\n"
                "  stop              - Stop the networked instance.\n"
                "  status            - Show SIF & networked instance status.\n"
                "  logs [N]          - Tail last N lines of networked logs.\n"
                "  exit | quit | q   - Exit the manager.\n"
            )
        elif cmd == "build":
            pull_sif_if_needed(force_pull="--rebuild" in args)
        elif cmd == "start":
            start_instance(rebuild="--rebuild" in args)
        elif cmd == "stop":
            stop_instance()
        elif cmd == "status":
            show_status()
        elif cmd == "logs":
            n = 50
            if args:
                try:
                    n = int(args[0])
                except ValueError:
                    print("Invalid number of lines; using default 50.")
            show_logs(n)
        else:
            print(f"Unknown command: {cmd}. Type 'help' for available commands.")

    if instance_running():
        logging.info("REPL exiting; stopping instance '%s'.", INSTANCE_NAME)
        stop_instance()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Singularity Sandbox Manager. Pulls SIF, manages networked instances.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", title="commands", required=False)

    build_parser = subparsers.add_parser("build", help="Download/update the SIF image.")
    build_parser.add_argument(
        "--rebuild", action="store_true", help="Force re-download of the SIF."
    )

    start_parser = subparsers.add_parser("start", help="Start the networked Singularity instance.")
    start_parser.add_argument(
        "--rebuild", action="store_true", help="Force re-download of the SIF first."
    )

    subparsers.add_parser("stop", help="Stop the networked Singularity instance.")
    subparsers.add_parser("status", help="Show SIF & instance status.")

    logs_parser = subparsers.add_parser("logs", help="Tail the instance log.")
    logs_parser.add_argument("n", nargs="?", type=int, default=50, help="Number of lines (default: 50).")

    args = parser.parse_args()

    if not args.cmd:
        repl()
        sys.exit(0)

    success = True
    if args.cmd == "build":
        success = pull_sif_if_needed(force_pull=args.rebuild)
    elif args.cmd == "start":
        success = start_instance(rebuild=args.rebuild)
    elif args.cmd == "stop":
        success = stop_instance()
    elif args.cmd == "status":
        show_status()
    elif args.cmd == "logs":
        show_logs(args.n)

    sys.exit(0 if success else 1)