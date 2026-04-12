"""Gemini function declaration schemas for all agent tools."""

from google.genai import types

TOOL_DECLARATIONS = [
    types.Tool(function_declarations=[
        # ── Exploration tools ────────────────────────────────────────
        types.FunctionDeclaration(
            name="search_similar_prompts",
            description=(
                "Search the prompt database for prompts semantically similar to a query. "
                "Use this for targeted follow-up searches when you need results from a "
                "specific concept folder. Expand concept-level requests into rich "
                "semantic queries."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(type="STRING", description="Search query text to find similar prompts"),
                    "k": types.Schema(type="INTEGER", description="Number of results to return (default 10)"),
                    "source_type": types.Schema(
                        type="STRING",
                        description="Filter by source: 'training', 'output', or null for both",
                        nullable=True,
                    ),
                    "concept": types.Schema(
                        type="STRING",
                        description=(
                            "Filter by concept folder name (e.g. 'action__cowgirl'). "
                            "Must be an exact folder name from the dataset overview or "
                            "query_dataset_map results. Omit if unsure — the query alone "
                            "usually suffices."
                        ),
                        nullable=True,
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_folder_themes",
            description=(
                "Get the intra-folder cluster themes for a specific concept folder and source type. "
                "Returns theme labels and prompt counts. Call this to explore the "
                "thematic variety within a folder before searching. Training and output "
                "have independent themes — specify which you want."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "folder_name": types.Schema(
                        type="STRING",
                        description=(
                            "The concept folder name using the full category__display_name format "
                            "(e.g. 'action__helpless', 'woman__salma_hayek'). Use the 'name' field "
                            "from the dataset overview."
                        ),
                    ),
                    "source_type": types.Schema(
                        type="STRING",
                        description="Which source's themes to retrieve: 'training' or 'output'",
                    ),
                },
                required=["folder_name", "source_type"],
            ),
        ),
        types.FunctionDeclaration(
            name="query_diverse_prompts",
            description=(
                "Primary search tool. Uses cluster-based diverse retrieval to find "
                "prompts across multiple concepts weighted by relevance. Returns "
                "results from several concept folders, balanced by slot allocation. "
                "Use rich semantic queries (expand concept-level requests into "
                "descriptive phrases)."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="STRING",
                        description="Search query text — expand concept-level requests into rich semantic queries",
                    ),
                    "k": types.Schema(
                        type="INTEGER",
                        description="Number of results to return (default 10)",
                        nullable=True,
                    ),
                    "source_type": types.Schema(
                        type="STRING",
                        description="Filter by source: 'training', 'output', or null for both",
                        nullable=True,
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="query_dataset_map",
            description=(
                "Search for dataset folders matching a query. Returns folder names, "
                "source types, prompt counts, summaries, and top themes. Use this when "
                "you need to find relevant folders after context has been truncated and "
                "the full dataset overview is no longer available, or to look up exact "
                "folder names before filtering with search_similar_prompts."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="STRING",
                        description="Search query to match against folder names, summaries, and themes",
                    ),
                    "k": types.Schema(
                        type="INTEGER",
                        description="Max folders to return (default 10)",
                        nullable=True,
                    ),
                    "source_type": types.Schema(
                        type="STRING",
                        description="Filter by source: 'training', 'output', or null for both",
                        nullable=True,
                    ),
                },
                required=["query"],
            ),
        ),
        # ── Quality signal tools ─────────────────────────────────────
        types.FunctionDeclaration(
            name="get_deletion_insights",
            description=(
                "Get patterns from deleted prompts to understand what to avoid. "
                "Returns prompts deleted for quality or wrong_direction issues, "
                "with common patterns. Use to learn from past failures before "
                "generating prompts for a concept or output folder."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "output_folder": types.Schema(
                        type="STRING",
                        description="Filter to a specific output folder",
                        nullable=True,
                    ),
                    "concept": types.Schema(
                        type="STRING",
                        description="Filter by concept/character name (searches graveyard embeddings)",
                        nullable=True,
                    ),
                    "k": types.Schema(
                        type="INTEGER",
                        description="Number of deleted prompts to return (default 5)",
                        nullable=True,
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_successful_patterns",
            description=(
                "Get prompts that led to productive regeneration chains "
                "(multiple iterations where images were kept). Shows what "
                "works well. Use for inspiration on prompt style and structure."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "output_folder": types.Schema(
                        type="STRING",
                        description="Filter to a specific output folder",
                        nullable=True,
                    ),
                    "min_depth": types.Schema(
                        type="INTEGER",
                        description="Minimum lineage depth to consider successful (default 3)",
                        nullable=True,
                    ),
                    "k": types.Schema(
                        type="INTEGER",
                        description="Number of patterns to return (default 5)",
                        nullable=True,
                    ),
                },
            ),
        ),
        # ── Generation tools ────────────────────────────────────────
        types.FunctionDeclaration(
            name="generate_image",
            description=(
                "Submit a single prompt for image generation via ComfyUI. "
                "Call this multiple times in one turn to generate different prompts "
                "with different settings. Only call when the user explicitly asks "
                "you to auto-generate. Seed should be -1 (random) unless the user "
                "explicitly specifies a seed or says 'use the same seed'. "
                "Unspecified optional settings use the user's configured defaults."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "prompt": types.Schema(type="STRING", description="The positive prompt text to generate"),
                    "negative_prompt": types.Schema(
                        type="STRING",
                        description="Negative prompt (defaults to user's configured default)",
                        nullable=True,
                    ),
                    "base_model": types.Schema(
                        type="STRING",
                        description="Diffusion model filename (defaults to user's configured default)",
                        nullable=True,
                    ),
                    "loras": types.Schema(
                        type="ARRAY",
                        description="LoRA filenames to apply (strength defaults to 1.0)",
                        items=types.Schema(type="STRING", description="LoRA filename"),
                        nullable=True,
                    ),
                    "output_folder": types.Schema(
                        type="STRING",
                        description="Output subdirectory name",
                        nullable=True,
                    ),
                    "seed": types.Schema(
                        type="INTEGER",
                        description="Seed (-1 for random, which is the default)",
                        nullable=True,
                    ),
                    "num_images": types.Schema(
                        type="INTEGER",
                        description="Number of images to generate (default 1)",
                        nullable=True,
                    ),
                    "sampler": types.Schema(
                        type="STRING",
                        description="Sampler name",
                        nullable=True,
                    ),
                    "cfg_scale": types.Schema(
                        type="NUMBER",
                        description="CFG scale value",
                        nullable=True,
                    ),
                    "scheduler": types.Schema(
                        type="STRING",
                        description="Scheduler name",
                        nullable=True,
                    ),
                    "steps": types.Schema(
                        type="INTEGER",
                        description="Number of sampling steps",
                        nullable=True,
                    ),
                },
                required=["prompt"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_available_loras",
            description=(
                "List available LoRA model filenames from ComfyUI. "
                "Call this when the user mentions a LoRA by description and you need "
                "the exact filename to pass to generate_image."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="get_output_directories",
            description=(
                "List available output subdirectories. Call this when the user references "
                "an output folder or when you need to choose where to save generated images."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="get_last_generation_settings",
            description=(
                "Get the full settings from the most recent completed generation job. "
                "Use when the user says 're-use settings', 'same as last time', or "
                "'pull settings from [folder]'. Set current_chat=true to only look at "
                "generations from this conversation."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "output_folder": types.Schema(
                        type="STRING",
                        description="If specified, get settings from the latest job in this output folder",
                        nullable=True,
                    ),
                    "current_chat": types.Schema(
                        type="BOOLEAN",
                        description="If true, only search generations from the current chat (default: false)",
                        nullable=True,
                    ),
                },
            ),
        ),
    ]),
]
