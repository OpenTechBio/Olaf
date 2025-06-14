#!/usr/bin/env bash
# move *out* of benchmarking/ into its parent (Olaf/)
cd "$(dirname "$0")"/..
python -m benchmarking.agents.create_agent_system "$@"