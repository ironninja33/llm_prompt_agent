#!/bin/bash
cd "$(dirname "$0")/.."
# Tool Study — analyze agent tool usage and test alternative query strategies
#
# Usage: ./run/run_tool_study.sh <command> [args]
#
# Phase 1 — Database Analysis:
#   analyze       - Tool call patterns (no API key needed)
#   leakage       - Folder name leakage (needs ChromaDB, no API key)
#   bias          - Output bias analysis (needs Gemini API)
#   redundancy    - Refinement redundancy (needs Gemini API)
#
# Phase 2 — Query Experiments:
#   experiment    - Run query experiment
#
# Examples:
#   ./run/run_tool_study.sh analyze
#   ./run/run_tool_study.sh leakage
#   ./run/run_tool_study.sh bias --k 10
#   ./run/run_tool_study.sh redundancy --limit 5
#   ./run/run_tool_study.sh experiment --strategy baseline --k 10
#   ./run/run_tool_study.sh experiment --strategy source_balanced --training-ratio 0.5
#   ./run/run_tool_study.sh experiment --strategy decomposed --query "elegant gown dramatic lighting"

set -e

COMMAND="${1:?Usage: $0 <analyze|leakage|bias|redundancy|experiment> [args]}"
shift

case "$COMMAND" in
    analyze)
        MODULE="src.experiments.tool_study.db_analysis.tool_call_patterns"
        ;;
    leakage)
        MODULE="src.experiments.tool_study.db_analysis.folder_name_leakage"
        ;;
    bias)
        MODULE="src.experiments.tool_study.db_analysis.output_bias_analysis"
        ;;
    redundancy)
        MODULE="src.experiments.tool_study.db_analysis.refinement_redundancy"
        ;;
    experiment)
        MODULE="src.experiments.tool_study.run_experiment"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Valid commands: analyze, leakage, bias, redundancy, experiment"
        exit 1
        ;;
esac

conda run -n llm_prompt_agent --no-capture-output \
    python -m "$MODULE" "$@"
