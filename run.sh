#!/bin/bash
# LLM Prompt Agent - Driver Script
# Activates the conda environment and starts the application

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_prompt_agent

# Install/update dependencies if needed
pip install -q -r requirements.txt

# Start the application
echo "Starting LLM Prompt Agent..."
python -m src.app
