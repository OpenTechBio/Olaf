#!/usr/bin/env bash
# move *out* of cli/ into its parent (Olaf/)
cd "$(dirname "$0")"/..
python -m cli.agents.create_agent_system "$@"