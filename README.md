# LLM Prompt Agent

A sophisticated web-based AI agent application that provides an intelligent chat interface powered by Google Gemini, featuring tool execution, vector-based semantic search, and ComfyUI integration for image generation workflows.

## Overview

LLM Prompt Agent is a Flask-based application that combines conversational AI with powerful capabilities including:

- **Intelligent Agent Loop**: Processes user messages through an agentic workflow with tool calling and multi-turn reasoning
- **Semantic Vector Search**: ChromaDB-powered vector store for semantic retrieval and RAG (Retrieval Augmented Generation)
- **ComfyUI Integration**: Direct integration with ComfyUI for AI image generation workflows
- **Multi-Modal Support**: Handle text, images, and attachments in conversations
- **Background Processing**: Asynchronous data ingestion and clustering services
- **Persistent Storage**: SQLite database for chat history, settings, and generation tracking
- **Real-Time Streaming**: Server-Sent Events (SSE) for streaming responses and progress updates

## Features

### Core Capabilities
- **Conversational AI**: Multi-turn chat sessions with context management and agent state persistence
- **Tool Execution**: Extensible tool system for the agent to interact with external services and data
- **Vector Embeddings**: Automatic embedding generation and semantic similarity search using Google's embedding models
- **Image Generation**: ComfyUI workflow submission, progress monitoring, and result retrieval
- **Attachment Handling**: Support for uploading and processing images and other file types
- **Rate Limiting**: Built-in rate limiter for API calls to respect service quotas

### User Interface
- **Modern Web Interface**: Responsive single-page application with clean, intuitive design
- **Message Streaming**: Real-time token-by-token response streaming
- **Interactive Components**: 
  - Searchable dropdowns
  - Pill-based input for tags/selections
  - Full-size image viewer
  - Generation overlay for monitoring ComfyUI jobs
  - Refine context for iterative improvements
- **Settings Panel**: Configure API keys, models, ComfyUI connection, and system prompts
- **Sidebar Navigation**: Browse chat history and manage sessions

### Backend Services
- **LLM Service**: Google Gemini API integration with model selection and rate limiting
- **Embedding Service**: Batch embedding generation for semantic search
- **Clustering Service**: K-means clustering for organizing and analyzing embeddings
- **Ingestion Service**: Background worker for processing and indexing data
- **Workflow Management**: Convert, validate, and execute ComfyUI workflows
- **Image Parser**: Extract metadata and process images from various sources

## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Conda**: Anaconda or Miniconda for environment management
- **Operating System**: Linux, macOS, or Windows

### External Services (Optional)
- **Google Gemini API**: Required for LLM and embedding functionality (get an API key from [Google AI Studio](https://aistudio.google.com/))
- **ComfyUI**: Optional, for image generation features (install from [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI))

### Python Dependencies
The following packages are automatically installed from `requirements.txt`:
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

### 4. Configure API Keys
On first launch, the application will prompt you to configure your Google Gemini API key through the Settings panel in the web interface. Alternatively, you can set it directly in the database or through the `/api/settings` endpoint.

## How to Run

### Using the Run Script (Recommended)
The project includes a convenience script that handles environment activation and startup:

```bash
./run.sh
```

The script will:
1. Activate the `llm_prompt_agent` conda environment
2. Install/update dependencies if needed
3. Start the Flask application

### Manual Startup
Alternatively, you can start the application manually:

```bash
conda activate llm_prompt_agent
python -m src.app
```

### Accessing the Application
Once running, open your web browser and navigate to:
```
http://localhost:5000
```

The application will:
- Initialize the SQLite database (`app.db`)
- Set up the ChromaDB vector store (`chroma_db/`)
- Start background ingestion services
- Begin listening for ComfyUI generation requests (if configured)

### Development Mode
The application runs in debug mode by default (configured in [`src/config.py`](src/config.py:22)). This enables:
- Auto-reload on code changes
- Detailed error pages
- Verbose logging

For production deployment, set `FLASK_DEBUG = False` in the configuration.

## Project Structure

```
llm_prompt_agent/
├── src/
│   ├── agent/              # Agent loop, tools, and system prompts
│   ├── controllers/        # Request handlers and business logic
│   ├── models/             # Database models and vector store
│   ├── services/           # Core services (LLM, embeddings, ComfyUI, etc.)
│   ├── static/             # Frontend assets (CSS, JavaScript)
│   ├── views/              # Flask routes and templates
│   ├── app.py              # Application factory and entry point
│   └── config.py           # Configuration settings
├── tests/                  # Test suite
├── requirements.txt        # Python dependencies
├── run.sh                  # Startup script
└── README.md              # This file
```

## Configuration

### Application Settings
Edit [`src/config.py`](src/config.py) to customize:
- Database paths (SQLite and ChromaDB)
- Default models for agent, embedding, and summarization
- Flask server host and port
- Rate limiting parameters
- ComfyUI connection settings

### Runtime Settings
Configure through the web interface Settings panel:
- **Gemini API Key**: Your Google AI API key
- **Model Selection**: Choose from available Gemini models
- **System Prompt**: Customize the agent's behavior and personality
- **ComfyUI URL**: Connection string for ComfyUI server (default: `http://localhost:8188`)
- **Output Directories**: Paths for ComfyUI output images

## Usage Examples

### Basic Chat
1. Open the application in your browser
2. Type a message in the input field
3. Press Enter or click Send
4. Watch as the agent streams its response in real-time

### Image Generation with ComfyUI
1. Ensure ComfyUI is running and configured in Settings
2. Load or create a workflow in ComfyUI
3. Submit generation requests through the agent interface
4. Monitor progress in the generation overlay
5. View results directly in the chat

### Semantic Search
The application automatically indexes chat content and attachments in the vector store, enabling:
- Semantic similarity search across conversations
- Context retrieval for RAG-enhanced responses
- Clustering and organization of related content

## Troubleshooting

### Application won't start
- Ensure the conda environment is activated: `conda activate llm_prompt_agent`
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check for port conflicts (default port 5000)

### No LLM responses
- Verify your Gemini API key is set in Settings
- Check the console logs for API errors or rate limiting
- Ensure you have internet connectivity

### ComfyUI integration not working
- Confirm ComfyUI is running at the configured URL
- Check the ComfyUI console for errors
- Verify output directory paths are correct and accessible

### Database errors
- Delete `app.db` and restart to recreate the database
- Check file permissions in the project directory

## License

[License information to be added]

## Contributing

Contact through GitHub

## Support

For issues, questions, or contributions, please refer to the project repository.
