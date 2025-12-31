#!/bin/bash
set -e

JENNY_DIR="${1:-.jenny}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python runner
python3 "${SCRIPT_DIR}/obsidian_actions_runner.py" "${JENNY_DIR}"
