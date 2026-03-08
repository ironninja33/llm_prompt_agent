#!/bin/bash
conda run -n llm_prompt_agent --no-capture-output python -m src.experiments.summary.keybert.run "$@"
