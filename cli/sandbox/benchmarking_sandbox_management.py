# Import logging and sys first for configuration
import logging
import sys

# --- Standard Library Imports ---
import argparse
import os
import time
import subprocess # Still needed for docker cp (if used elsewhere)
import shlex
import json
import io
import tempfile # May not be needed anymore

# --- Third-Party Imports ---
try:
    import docker
except ImportError:
    logging.error("Error: docker library not found.")
    logging.error("Please install it in your host environment: pip install docker")
    sys.exit(1)

# Optional: Use rich for better formatting if available
try:
    from rich.console import Console
    from rich.prompt import Prompt
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None
    class Prompt:
        @staticmethod
        def ask(prompt, choices=None, default=None):
            p_text = f"{prompt} "
            if choices: p_text += f"({'/'.join(choices)}) "
            if default: p_text += f"[{default}] "
            return input(p_text).strip()

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCKERFILE_PATH = os.path.join(SCRIPT_DIR, 'Dockerfile') # Assumes Dockerfile is in the same dir
IMAGE_TAG = "benchmarking-sandbox:latest"
CONTAINER_NAME = "benchmarking_sandbox_instance"
# Port mapping for the FastAPI service inside the container
API_PORT_INSIDE = 8000
API_PORT_HOST = 8000 # Host port to map to

# Centralized message printing function using logging and optional rich console
def _print_message(message, style=None, is_error=False):
    """Helper to print using rich console or standard logging."""
    level = logging.INFO # Default level
    if is_error or (style and 'red' in style):
        level = logging.ERROR
    elif style and 'yellow' in style:
        level = logging.WARNING
    elif style and 'green' in style:
        level = logging.INFO
    elif style and ('dim' in style or 'blue' in style or 'cyan' in style):
        # Keep these less critical styled messages as DEBUG if overall level is DEBUG
        # If overall level is INFO, they won't show unless changed here.
        # Let's make them INFO so they appear with INFO level logging.
        level = logging.INFO # Changed from DEBUG to INFO

    # Log the message using Python's standard logging
    # It will only appear if 'level' >= the level set in basicConfig
    logging.log(level, message)

    # Additionally print to console using rich if available and not an error
    # This provides the styling even if the log level is higher than the message level
    if HAS_RICH and console and not is_error:
        console.print(message, style=style if style else None)
    elif not HAS_RICH and not is_error and level >= logging.INFO:
         # Fallback print for non-error, non-debug messages when rich is unavailable
         # Ensures INFO messages are printed if logging level is INFO or lower
         print(message)


class SandboxManager:
    """
    Manages the lifecycle of the benchmarking sandbox Docker container,
    which now runs a kernel and a FastAPI service.
    Uses logging for internal messages.
    """
    def __init__(self):
        self.client = None
        self.container = None
        try:
            docker_host = os.environ.get("DOCKER_HOST")
            if docker_host:
                logging.info(f"Using DOCKER_HOST: {docker_host}")
            self.client = docker.from_env()
            self.client.ping()
            logging.info("Docker client initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing Docker client: {e}", exc_info=True)
            logging.error("Ensure Docker Desktop/Engine is running and DOCKER_HOST is set if needed.")
            sys.exit(1)

    def _get_container_logs(self, tail=50):
        """Retrieves recent logs from the managed container."""
        current_container = self._find_container()
        if not current_container:
            logging.warning("Attempted to get logs, but container '%s' not found.", CONTAINER_NAME)
            if self.container:
                logging.debug("Clearing stale internal container object.")
                self.container = None
            return "(Container not found or already removed)"

        target_container = current_container
        try:
            logs = target_container.logs(tail=tail).decode('utf-8', errors='ignore')
            return logs
        except Exception as log_e:
            logging.error(f"Could not retrieve logs for container '{target_container.id}': {log_e}")
            return f"(Could not retrieve logs: {log_e})"

    def _find_container(self):
        """Finds the container by name, returns container object or None."""
        try:
            container = self.client.containers.get(CONTAINER_NAME)
            return container
        except docker.errors.NotFound:
            return None
        except Exception as e:
            logging.error(f"Error finding container '{CONTAINER_NAME}': {e}")
            return None

    def build_image(self):
        """Builds the Docker image from the Dockerfile."""
        _print_message(f"Building Docker image '[cyan]{IMAGE_TAG}[/cyan]' from [blue]{DOCKERFILE_PATH}[/blue]...", style="bold blue")
        if not os.path.exists(DOCKERFILE_PATH):
            _print_message(f"Error: Dockerfile not found at {DOCKERFILE_PATH}", style="bold red", is_error=True)
            return False
        try:
            build_context = os.path.dirname(DOCKERFILE_PATH)
            _print_message(f"Using build context: [blue]{build_context}[/blue]", style="blue")
            stream = self.client.api.build(
                path=build_context,
                dockerfile=os.path.basename(DOCKERFILE_PATH),
                tag=IMAGE_TAG,
                rm=True,
                decode=True
            )
            last_status = None
            for chunk in stream:
                # Process build stream for logging/display
                if 'stream' in chunk:
                    line = chunk['stream'].strip()
                    # Log/print build output lines only if they contain content
                    # Use _print_message with no style (defaults to INFO level)
                    if line: _print_message(line) # <-- CHANGED from logging.debug
                elif 'errorDetail' in chunk:
                    error_msg = chunk['errorDetail']['message']
                    _print_message(f"Build Error: {error_msg}", style="bold red", is_error=True)
                    return False
                elif 'status' in chunk:
                    status = chunk['status']
                    # Log/print status changes, but reduce noise from download/extract steps
                    if status != last_status and "Downloading" not in status and "Extracting" not in status:
                         # Use _print_message with dim style (logs as INFO)
                         _print_message(f"Status: {status}", style="dim") # <-- CHANGED from logging.debug
                         last_status = status
            _print_message(f"Image '[cyan]{IMAGE_TAG}[/cyan]' built successfully.", style="green")
            return True
        except docker.errors.BuildError as e:
            _print_message(f"Docker build failed: {e}", style="bold red", is_error=True)
            for line in e.build_log:
                 if 'stream' in line: logging.error(f"Build Log: {line['stream'].strip()}")
            return False
        except Exception as e:
            _print_message(f"An unexpected error occurred during build: {e}", style="bold red", is_error=True)
            logging.exception("Build error details:")
            return False

    def start_container(self, rebuild=False):
        """Starts the Docker container with the FastAPI service."""
        # Handle rebuild request
        if rebuild:
            _print_message("Rebuild requested.")
            existing_container = self._find_container()
            if existing_container:
                _print_message("Stopping existing container before rebuild...")
                if not self.stop_container(remove=True, container_obj=existing_container):
                     _print_message("Failed to stop/remove existing container during rebuild. Aborting start.", style="red", is_error=True)
                     return False
            if not self.build_image():
                # Error printed by build_image
                return False

        # Ensure no old container instance is running
        existing_container = self._find_container()
        if existing_container:
            _print_message(f"Found existing container '{CONTAINER_NAME}'. Stopping and removing it first...")
            if not self.stop_container(remove=True, container_obj=existing_container):
                _print_message(f"Failed to stop/remove existing container '{CONTAINER_NAME}'. Aborting start.", style="bold red", is_error=True)
                return False

        # Start the container
        _print_message(f"Starting container '[cyan]{CONTAINER_NAME}[/cyan]' with API service...", style="cyan")
        try:
            # Check if the image exists locally, build if not
            try:
                self.client.images.get(IMAGE_TAG)
            except docker.errors.ImageNotFound:
                _print_message(f"Image '[cyan]{IMAGE_TAG}[/cyan]' not found locally. Building...", style="yellow")
                if not self.build_image():
                    return False

            # Define port mapping for the FastAPI service
            port_map = {f'{API_PORT_INSIDE}/tcp': API_PORT_HOST}
            logging.info(f"Mapping container port {API_PORT_INSIDE} to host port {API_PORT_HOST}")

            # Define container run options (using default bridge network)
            run_options = {
                'name': CONTAINER_NAME,
                'detach': True,
                'auto_remove': False, # Keep container for inspection on failure
                'ports': port_map,    # Map the API port
            }
            logging.debug(f"Docker run options: {run_options}")

            # Run the container
            self.container = self.client.containers.run(
                IMAGE_TAG,
                **run_options
            )

            _print_message(f"Container '[cyan]{CONTAINER_NAME}[/cyan]' started ([yellow]{self.container.short_id}[/yellow]).", style="cyan")

            # Wait briefly for the service inside to potentially start
            wait_time = 5 # Seconds to wait for API/kernel startup
            _print_message(f"Waiting {wait_time}s for services inside container to initialize...")
            time.sleep(wait_time)

            # Basic check: Is the container still running?
            # Use _find_container to check live status
            current_container = self._find_container()
            if not current_container or current_container.status != 'running':
                 status = current_container.status if current_container else 'not found'
                 _print_message(f"Container exited unexpectedly shortly after start (status: {status}).", style="bold red", is_error=True)
                 # Try to get logs even if stopped/gone
                 logs = self._get_container_logs()
                 _print_message("--- Container Logs ---", is_error=True)
                 _print_message(logs if logs else "(Could not retrieve logs)", is_error=True)
                 _print_message("----------------------", is_error=True)
                 self.container = None # Clear internal ref
                 return False

            _print_message(f"Container running. API should be accessible at http://localhost:{API_PORT_HOST}", style="green")
            return True # Container started successfully

        except Exception as e:
            _print_message(f"Error during container start: {e}", style="bold red", is_error=True)
            logging.exception("Container start error details:")
            # Ensure cleanup if self.container was assigned
            current_container = self._find_container()
            if current_container:
                self.stop_container(remove=True, container_obj=current_container)
            self.container = None
            return False

    def stop_container(self, remove=False, container_obj=None):
        """Stops the container and optionally removes it."""
        # Find the container to stop if not provided
        container_to_stop = container_obj or self._find_container()

        if not container_to_stop:
            _print_message(f"Container '{CONTAINER_NAME}' not found or already stopped/removed.", style="yellow")
            if container_obj is None:
                 self.container = None
            return True

        _print_message(f"Stopping container '[cyan]{CONTAINER_NAME}[/cyan]' ([yellow]{container_to_stop.short_id}[/yellow])...", style="cyan")
        stopped = False
        removed_flag = False
        try:
            container_to_stop.reload()
            current_status = container_to_stop.status
            if current_status == 'running':
                _print_message("Sending stop signal to container...")
                container_to_stop.stop(timeout=10)
                time.sleep(1)
                container_to_stop.reload()
                if container_to_stop.status == 'exited':
                     _print_message("Container stopped successfully.", style="green")
                     stopped = True
                else:
                     _print_message(f"Container status is '{container_to_stop.status}' after stop attempt. Trying force stop...", style="yellow")
                     container_to_stop.kill()
                     time.sleep(1)
                     container_to_stop.reload()
                     if container_to_stop.status == 'exited':
                          _print_message("Container force stopped.", style="green")
                          stopped = True
                     else:
                          _print_message(f"Container still '{container_to_stop.status}' after force stop.", style="red", is_error=True)
            else:
                _print_message(f"Container was not running (status: {current_status}).")
                stopped = True

            if remove and stopped:
                try:
                    _print_message("Removing container...")
                    container_to_stop.remove(force=True)
                    _print_message("Container removed.", style="green")
                    removed_flag = True
                except docker.errors.NotFound:
                    _print_message("Container was already removed.", style="yellow")
                    removed_flag = True
                except docker.errors.APIError as e:
                    _print_message(f"API error removing container: {e}", style="yellow", is_error=True)
                    try:
                         self.client.containers.get(container_to_stop.id)
                         removed_flag = False
                    except docker.errors.NotFound:
                         _print_message("Container appears removed despite API error.", style="yellow")
                         removed_flag = True
            elif remove and not stopped:
                 _print_message("Remove requested, but container failed to stop. Attempting force remove...", style="yellow")
                 try:
                     container_to_stop.remove(force=True)
                     _print_message("Container force removed.", style="green")
                     removed_flag = True
                 except Exception as fe:
                      _print_message(f"Failed to force remove container: {fe}", style="red", is_error=True)
                      removed_flag = False
            elif not remove:
                 removed_flag = True

            if self.container and self.container.id == container_to_stop.id:
                 self.container = None

            return stopped and removed_flag

        except Exception as e:
            _print_message(f"Error stopping/removing container: {e}", style="bold red", is_error=True)
            logging.exception("Stop/remove error details:")
            if remove and container_to_stop:
                 try:
                     _print_message("Attempting force remove after error...", style="yellow")
                     container_to_stop.remove(force=True)
                     _print_message("Container force removed after error.", style="green")
                 except Exception as fe:
                     _print_message(f"Failed to force remove container after error: {fe}", style="red", is_error=True)
            if self.container and container_to_stop and self.container.id == container_to_stop.id:
                self.container = None
            return False

    def get_status(self):
        """Gets the status of the container."""
        container_status = "not found"
        container = self._find_container()
        if container:
            try:
                container.reload()
                container_status = container.status
            except Exception as e:
                logging.warning(f"Error getting container status: {e}")
                container_status = "unknown (error)"

        return f"Container: {container_status}, API Port (Host): {API_PORT_HOST}"


# --- Interactive Mode Functions ---
def print_interactive_help():
    """Prints help message for interactive mode using _print_message."""
    _print_message("\n[bold cyan]Available Commands:[/bold cyan]", style="bold cyan")
    _print_message("  [green]build[/green]       Build the Docker image.", style="green")
    _print_message("  [green]start[/green] [--rebuild] Start container with API service.", style="green")
    _print_message("  [green]stop[/green]        Stop & remove container.", style="green")
    _print_message("  [green]status[/green]      Check container status.", style="green")
    _print_message("  [green]logs[/green] [N]     Show last N container logs (default 50).", style="green")
    _print_message("  [green]help[/green]        Show this help message.", style="green")
    _print_message("  [green]exit[/green]        Exit (prompts to stop container if running).", style="green")
    _print_message("\nExample: [yellow]start --rebuild[/yellow]", style="yellow")

def interactive_loop(manager):
    """Runs the interactive command loop."""
    _print_message("[bold blue]Welcome to the Stateful Benchmarking Sandbox Manager (API Mode)![/bold blue]", style="bold blue")
    print_interactive_help()
    while True:
        try:
            raw_command = Prompt.ask("\nEnter command ('help' or 'exit')")
            if not raw_command: continue

            try:
                command_parts = shlex.split(raw_command)
            except ValueError as e:
                _print_message(f"Error parsing command: {e}", style="red", is_error=True)
                continue
            if not command_parts: continue

            command = command_parts[0].lower()
            args = command_parts[1:]

            if command == "exit":
                container_running = False
                container_obj = manager._find_container()
                if container_obj:
                    try:
                        container_obj.reload()
                        container_running = container_obj.status == 'running'
                    except Exception as e:
                         logging.warning(f"Could not check container status on exit: {e}")
                if container_running:
                     should_stop_str = Prompt.ask(f"Container '{CONTAINER_NAME}' is running. Stop it?", choices=["y", "n"], default="y")
                     if should_stop_str.lower() == 'y':
                         _print_message("Stopping container on exit...")
                         manager.stop_container(remove=True, container_obj=container_obj)
                break
            elif command == "help":
                print_interactive_help()
            elif command == "build":
                if len(args) == 0: manager.build_image()
                else: _print_message("Usage: build", style="yellow")
            elif command == "start":
                rebuild = '--rebuild' in args
                if all(a == '--rebuild' for a in args if a.startswith('--')) or len(args) == 0:
                     manager.start_container(rebuild=rebuild)
                else: _print_message("Usage: start [--rebuild]", style="yellow")
            elif command == "stop":
                 if len(args) == 0: manager.stop_container(remove=True)
                 else: _print_message("Usage: stop", style="yellow")
            elif command == "status":
                 if len(args) == 0: _print_message(f"Status: {manager.get_status()}")
                 else: _print_message("Usage: status", style="yellow")
            elif command == "logs":
                 tail_count = 50
                 if len(args) == 1:
                     try:
                         tail_count = int(args[0])
                     except ValueError:
                         _print_message("Usage: logs [number_of_lines]", style="yellow")
                         continue
                 elif len(args) > 1:
                      _print_message("Usage: logs [number_of_lines]", style="yellow")
                      continue

                 logs = manager._get_container_logs(tail=tail_count)
                 _print_message(f"\n--- Last {tail_count} Container Logs ---")
                 # Use standard print for logs as they can be multiline and formatting is less critical
                 print(logs if logs else "(No logs retrieved or container not found)")
                 _print_message("-----------------------------")
            else:
                _print_message(f"Unknown command: '{command}'. Type 'help'.", style="red")

        except EOFError:
             _print_message("\nEOF detected. Exiting.", style="yellow")
             container_running = False; container_obj = manager._find_container()
             if container_obj:
                try:
                    container_obj.reload();
                    container_running = (container_obj.status == 'running')
                except Exception: pass
             if container_running:
                  should_stop_str = Prompt.ask(f"Container '{CONTAINER_NAME}' is running. Stop it?", choices=["y", "n"], default="y")
                  if should_stop_str.lower() == 'y': manager.stop_container(remove=True, container_obj=container_obj)
             break
        except KeyboardInterrupt:
             _print_message("\nInterrupted by user. Type 'exit'.", style="yellow")
        except Exception as e:
             _print_message(f"Unexpected error in interactive loop: {e}", style="bold red", is_error=True)
             logging.exception("Interactive loop error details:")

    _print_message("Exiting sandbox manager.", style="bold blue")


# --- Main Entry Point ---
def main():
    try:
        manager = SandboxManager()
    except SystemExit:
        sys.exit(1)

    if len(sys.argv) == 1:
        interactive_loop(manager)
        sys.exit(0)

    # --- Command-Line Argument Parsing (Simplified) ---
    parser = argparse.ArgumentParser(
        description="Manage the Stateful Benchmarking Sandbox (API Mode).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(
        dest='command',
        help='Action to perform',
        required=True
    )

    parser_build = subparsers.add_parser('build', help='Build the Docker image.')
    parser_build.set_defaults(func=lambda args, mgr: mgr.build_image())

    parser_start = subparsers.add_parser('start', help='Start container with API service.')
    parser_start.add_argument('--rebuild', action='store_true', help='Rebuild image first.')
    parser_start.set_defaults(func=lambda args, mgr: mgr.start_container(rebuild=args.rebuild))

    parser_stop = subparsers.add_parser('stop', help='Stop & remove container.')
    parser_stop.set_defaults(func=lambda args, mgr: mgr.stop_container(remove=True))

    parser_status = subparsers.add_parser('status', help='Check container status.')
    parser_status.set_defaults(func=lambda args, mgr: _print_message(f"Status: {mgr.get_status()}"))

    parser_logs = subparsers.add_parser('logs', help='Show last N container logs.')
    parser_logs.add_argument('n', type=int, nargs='?', default=50, help='Number of lines to show (default: 50)')
    def show_logs(args, mgr):
        logs = mgr._get_container_logs(tail=args.n)
        _print_message(f"\n--- Last {args.n} Container Logs ---")
        # Use standard print for logs
        print(logs if logs else "(No logs retrieved or container not found)")
        _print_message("-----------------------------")
        return True # Assume success for showing logs
    parser_logs.set_defaults(func=show_logs)

    args = parser.parse_args()
    # Execute the function and store success status
    success = args.func(args, manager)
    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
