import os, os.path, sys

base = int(os.environ.get("IPY_BASE_PORT", 4000))
argv = [
    sys.executable, "-Xfrozen_modules=off", "-vv", "-m", "ipykernel_launcher",
    "--ip=127.0.0.1",
    "--log-level=DEBUG",
    f"--shell={base + 0}",
    f"--iopub={base + 1}",
    f"--stdin={base + 2}",
    f"--hb={base + 3}",
    f"--control={base + 4}",
    "-f", "/home/sandboxuser/kernel-connection.json",
]
os.execvp(argv[0], argv)