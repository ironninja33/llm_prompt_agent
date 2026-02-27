# LLM Prompt Agent

An agentic interface to help with prompt creation for image generation models. The idea is to pull from existing training data and output data to ground prompts but also use the agent's creativity to inspire new ideas.

The idea is to keep the dependencies to a minimum if possible.

## Environment
Always use the conda environment `llm_prompt_agent` by running commands with `conda run -n llm_prompt_agent <command>` or activate it first with `conda activate llm_prompt_agent`.

## Workflow Rules

When in plan mode and prior to making changes that span more than a module, create an implementation plan in the /docs/plans folder. This should contain:
- An overview of what changes you will make, both new features and to the existing codebase
- Pseudocode for any major new features, workflows, or algorithms
- Database schema or metadata formatting changes
- An implementation plan

Do not proceed to code mode until you have created this plan file.