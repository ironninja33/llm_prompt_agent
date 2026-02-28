# LLM Prompt Agent

A web-based AI agent for prompt creation and image generation. It pulls from existing training data and generated output to ground prompts using semantic search, while leveraging the agent's creativity to inspire new ideas.

## Overview

LLM Prompt Agent is a Flask application that combines conversational AI with ComfyUI image generation, semantic search, and a dedicated image browser. The goal is to keep dependencies minimal.

### Core Capabilities

- **Agentic Chat**: Multi-turn conversations with Google Gemini, tool calling, and state management. The agent follows a structured workflow (gathering info, searching, generating, refining) to produce image generation prompts.
- **Semantic Search**: ChromaDB vector store for embedding-based similarity search across training data and generated output. Supports themed prompt discovery (intra-folder, cross-folder).
- **ComfyUI Integration**: Workflow submission, progress monitoring via SSE, and result retrieval. Configurable generation settings (model, sampler, scheduler, CFG, steps, LoRAs, seed).
- **Image Browser**: Dedicated `/browser` page for browsing, searching, and generating images outside of chat context.
- **Multi-Modal Support**: Image attachments in chat, metadata extraction from generated images (prompt, model, seed, sampler, etc.).

## Features

### Image Browser
- **Directory Navigation**: Hierarchical browsing of output directories with breadcrumb navigation and directory preview cards showing recent images
- **Dual Search Modes**: Toggle between keyword search (comma-separated terms against metadata/prompts) and embedding search (semantic similarity via vector store)
- **Lazy Loading**: Three-phase registration system — fast file discovery, paginated DB queries, then deferred metadata parsing — for responsive browsing of large directories
- **Thumbnail Caching**: SQLite-backed thumbnail cache with mtime-based staleness detection
- **Browser-to-Generation**: Generate new images directly from the browser without chat context
- **Live Polling**: Automatic detection of newly generated files in output directories

### Chat & Agent
- **Streaming Responses**: Real-time token-by-token streaming via Server-Sent Events
- **Tool System**: Extensible tools for prompt search and discovery (`query_themed_prompts`, `search_similar_prompts`, `search_diverse_prompts`, `get_random_prompts`, `get_opposite_prompts`)
- **Tool Call Tracking**: All tool invocations persisted with parameters and result summaries
- **Attachment Handling**: Upload and process images in conversations
- **Rate Limiting**: Built-in rate limiter for API calls

### Image Generation
- **Generation Overlay**: Configurable dialog for submitting ComfyUI jobs with model, LoRA, sampler, scheduler, CFG, steps, seed, and output folder selection
- **Session Persistence**: Generation settings persist per chat session
- **Full-Size Viewer**: Navigate generated images with sidebar showing generation parameters
- **Progress Monitoring**: Real-time generation progress via ComfyUI websocket

### Backend Services
- **Ingestion Service**: Background worker for indexing and embedding training/output data
- **Clustering Service**: K-means clustering for organizing embeddings
- **Workflow Manager**: Convert, validate, and execute ComfyUI workflows (API and UI formats)
- **Image Parser**: Extract metadata from generated images (ComfyUI workflow data, prompt, model, seed)

## Requirements

### System Requirements
- **Python**: 3.8+
- **Conda**: Anaconda or Miniconda
- **OS**: Linux, macOS, or Windows

### External Services
- **Google Gemini API**: Required for LLM and embeddings ([Google AI Studio](https://aistudio.google.com/))
- **ComfyUI**: Required for image generation ([ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI))

### Python Dependencies
Installed from `requirements.txt`:
- `flask>=3.0` - Web framework
- `google-genai>=1.0` - Google Gemini API client
- `chromadb>=0.4` - Vector database
- `Pillow>=10.0` - Image processing
- `gunicorn>=21.0` - Production WSGI server
- `scikit-learn>=1.3` - Machine learning utilities

## Installation

### 1. Clone the Repository
```bash
git clone git@github.com:ironninja33/llm_prompt_agent.git
cd llm_prompt_agent
```

### 2. Create Conda Environment
```bash
conda create -n llm_prompt_agent python=3.11
conda activate llm_prompt_agent
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure
On first launch, configure through the Settings panel in the web interface:
- **General**: Gemini API key, model selection (agent, embedding, summarization), rate limit
- **ComfyUI**: Base URL, default model, negative prompt
- **System Prompt**: Customize agent behavior

## How to Run

### Using the Run Script (Recommended)
```bash
./run.sh
```

### Manual Startup
```bash
conda activate llm_prompt_agent
python -m src.app
```

### Accessing the Application
Navigate to `http://localhost:5000`. The application will:
- Initialize the SQLite database and ChromaDB vector store
- Start background ingestion services
- Begin listening for ComfyUI generation requests (if configured)

## Project Structure

```
llm_prompt_agent/
├── src/
│   ├── agent/              # Agent loop, tools, and system prompts
│   ├── controllers/        # Request handlers (chat, generation, browser)
│   ├── models/             # Database models, vector store, browser data
│   ├── services/           # Core services (LLM, embeddings, ComfyUI,
│   │                       #   ingestion, clustering, thumbnails)
│   ├── static/
│   │   ├── css/            # Stylesheets (layout, chat, browser, generation)
│   │   └── js/
│   │       ├── browser/    # Browser page modules
│   │       ├── chat/       # Chat pane, attachments, generation bubbles
│   │       ├── core/       # App init, API client
│   │       ├── generation/ # Generation overlay, thumbnails, viewer
│   │       └── settings/   # Settings panel
│   ├── views/
│   │   ├── api/            # REST API routes (chat, generation, ComfyUI, browser)
│   │   └── templates/      # Jinja2 templates and partials
│   ├── app.py              # Application factory and entry point
│   └── config.py           # Configuration settings
├── requirements.txt
├── run.sh
└── README.md
```

## Configuration

### Application Settings (`src/config.py`)
- Database paths (SQLite and ChromaDB)
- Default models for agent, embedding, and summarization
- Flask server host and port
- Rate limiting parameters
- ComfyUI connection settings

### Runtime Settings (Web UI)
- **General**: API key, model selection, system prompt
- **ComfyUI**: Connection URL, default model, negative prompt
- **Data Directories**: Configure training and output directories for indexing

## Troubleshooting

### Application won't start
- Ensure the conda environment is activated: `conda activate llm_prompt_agent`
- Verify dependencies: `pip install -r requirements.txt`
- Check for port conflicts (default port 5000)

### No LLM responses
- Verify your Gemini API key in Settings
- Check console logs for API errors or rate limiting

### ComfyUI integration not working
- Confirm ComfyUI is running at the configured URL
- Use the connection test button in Settings > ComfyUI
- Verify output directory paths are correct and accessible

### Browser shows no images
- Add output directories in Settings > General > Data Directories
- Wait for the initial ingestion to complete (check console for progress)
- Verify image files have readable metadata

## License

[License information to be added]

## Contributing

Contact through GitHub

## Support

For issues, questions, or contributions, please refer to the project repository.
