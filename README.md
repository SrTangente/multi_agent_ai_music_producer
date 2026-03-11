# Multi-Agent AI Music Producer 🎵

A LangGraph-orchestrated system that generates music segment-by-segment using specialized AI agents. Each agent handles a specific aspect of music production - from analyzing reference tracks to directing composition, generating audio, critiquing quality, and mastering the final output.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Prompt + References                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Analysis Agent                           │
│     Extracts musical features from reference tracks          │
│     (BPM, key, instruments, energy profile)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Director Agent                           │
│     Creates track plan with segment breakdown                │
│     (intro, verse, chorus, outro, transitions)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Production Loop (per segment)                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Production  │───▶│   Critic    │───▶│  Revision?  │      │
│  │   Agent     │    │   Agent     │    │             │      │
│  │ (generate)  │    │ (evaluate)  │    │ (retry/next)│      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Mastering Agent                           │
│     Concatenates segments, applies final processing          │
│     (loudness normalization, transitions)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Final Track                             │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-Agent Architecture**: Specialized agents for analysis, direction, production, critique, and mastering
- **LangGraph Orchestration**: State-based workflow with conditional routing and retries
- **Multiple LLM Providers**: Anthropic, OpenAI, HuggingFace, Ollama
- **Iterative Refinement**: Critic agent evaluates each segment with revision loop
- **Configurable**: YAML-based settings for all components, output directory auto-created

## Installation

### Using uv (recommended)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/multi_agent_ai_music_producer.git
cd multi_agent_ai_music_producer

# Create venv and install
uv venv
source .venv/bin/activate
uv pip install -e .

# For HuggingFace support
uv pip install -e ".[huggingface]"

# For all providers
uv pip install -e ".[all-providers]"
```

### Using pip

```bash
pip install -e .
pip install -e ".[huggingface]"  # For HuggingFace models
```

## Quick Start

### 1. Set up environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run with Python

```python
from src.config import Settings, LLMConfig
from src.graph.workflow import MusicProducerGraph
from src.logging.logger import MusicProducerLogger, LogLevel

# Configure
settings = Settings(
    llm=LLMConfig(
        provider="huggingface",  # or "anthropic", "openai", "ollama"
        model="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.7,
    )
)

logger = MusicProducerLogger(
    run_id="my_track",
    output_dir="output",
    console_output=True,
)

# Build and run workflow
graph = MusicProducerGraph(settings=settings, logger=logger)
graph.build()

result = graph.invoke(
    user_prompt="Create a chill lofi hip-hop beat with jazzy piano and mellow drums",
    reference_paths=["path/to/reference.mp3"],  # optional
)

print(f"Generated track: {result['final_track_path']}")
```

### 3. Run on Google Colab

Open `notebooks/colab_test.ipynb` in Colab with GPU runtime (T4 or better).

```python
# In Colab - mount Drive and clone
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone https://github.com/YOUR_USERNAME/multi_agent_ai_music_producer.git
%cd multi_agent_ai_music_producer

# Install deps
!pip install -q transformers accelerate bitsandbytes langgraph -e .

# Run with 4-bit quantization for T4 GPU
from src.llm.huggingface_provider import HuggingFaceProvider
provider = HuggingFaceProvider(
    model="Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True,  # Fits in 16GB VRAM
)
```

## Project Structure

```
multi_agent_ai_music_producer/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── base.py       # BaseAgent with tool execution
│   │   ├── analysis.py   # Reference track analysis
│   │   ├── director.py   # Track planning
│   │   ├── production.py # Audio generation
│   │   ├── critic.py     # Quality evaluation
│   │   └── mastering.py  # Final processing
│   ├── graph/            # LangGraph workflow
│   │   ├── workflow.py   # Main graph builder
│   │   └── nodes.py      # Node wrappers
│   ├── llm/              # LLM providers
│   │   ├── base.py       # Provider interface
│   │   ├── anthropic_provider.py
│   │   ├── openai_provider.py
│   │   ├── huggingface_provider.py
│   │   └── ollama_provider.py
│   ├── state/            # State management
│   │   ├── schemas.py    # Pydantic models
│   │   └── reducers.py   # State updates
│   ├── tools/            # Agent tools
│   │   └── audio_generation.py
│   ├── logging/          # Logging & tracing
│   └── config.py         # Settings
├── tests/                # Test suite
├── notebooks/            # Jupyter notebooks
│   └── colab_test.ipynb  # Colab GPU runner
├── config/
│   └── settings.yaml     # Default settings
└── output/               # Generated audio
```

## Configuration

Edit `config/settings.yaml` or pass settings programmatically:

```yaml
llm:
  provider: huggingface
  model: Qwen/Qwen2.5-7B-Instruct
  temperature: 0.7
  max_tokens: 1024

generation:
  max_retries: 3
  default_segment_duration: 15.0
  approval_threshold: 0.6

audio:
  sample_rate: 32000
  output_format: wav
```

## Supported LLM Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus | Best quality, requires API key |
| **OpenAI** | GPT-4o, GPT-4 Turbo | Good quality, requires API key |
| **HuggingFace** | Qwen2.5-7B, Llama 3.1, etc. | Free, runs locally or Colab |
| **Ollama** | Any Ollama model | Free, local inference |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_integration_hf.py -v
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Format code
black src tests
isort src tests

# Type check
mypy src

# Lint
ruff check src
```

## Kubernetes Deployment

This project includes a production-grade Helm chart for deploying to Kubernetes with GPU support for music generation.

### Architecture Overview

```
                                    ┌─────────────────┐
                                    │   Ingress       │
                                    │   (TLS/HTTPS)   │
                                    └────────┬────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Cluster                            │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │
│  │   API Server    │    │   API Server    │    │   API Server    │   │
│  │   (HPA: 2-10)   │    │   (Replica)     │    │   (Replica)     │   │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘   │
│           │                      │                      │            │
│           └──────────────────────┼──────────────────────┘            │
│                                  │                                   │
│                                  ▼                                   │
│                         ┌─────────────────┐                          │
│                         │     Redis       │                          │
│                         │   (Job Queue)   │                          │
│                         └────────┬────────┘                          │
│                                  │                                   │
│           ┌──────────────────────┼──────────────────────┐            │
│           │                      │                      │            │
│           ▼                      ▼                      ▼            │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │
│  │  GPU Job (A100) │    │  GPU Job (A100) │    │  GPU Job (T4)   │   │
│  │  Music Gen      │    │  Music Gen      │    │  Music Gen      │   │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘   │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    Persistent Volumes                          │  │
│  │  ┌──────────┐      ┌──────────┐      ┌──────────┐             │  │
│  │  │  Output  │      │References│      │Model Cache│             │  │
│  │  │  (50Gi)  │      │  (20Gi)  │      │  (100Gi)  │             │  │
│  │  └──────────┘      └──────────┘      └──────────┘             │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────┐
        │              Monitoring Stack                    │
        │  ┌──────────────┐      ┌──────────────┐         │
        │  │  Prometheus  │─────▶│   Grafana    │         │
        │  │  (metrics)   │      │  (dashboard) │         │
        │  └──────────────┘      └──────────────┘         │
        └─────────────────────────────────────────────────┘
```

### Key Features

- **Helm Chart**: Full production-ready chart with customizable values
- **GPU Scheduling**: Node affinity, tolerations, and priority classes for GPU Jobs
- **HPA**: Automatic scaling based on CPU/memory utilization
- **TLS Ingress**: cert-manager integration with Let's Encrypt
- **Observability**: Prometheus ServiceMonitor + Grafana dashboard
- **Security**: NetworkPolicy, PodSecurityContext, non-root containers

### Quick Start

```bash
# Add Helm dependencies
cd k8s/helm/music-producer
helm dependency update

# Install in staging
helm install music-producer . \
  -f values-staging.yaml \
  --namespace music-producer \
  --create-namespace \
  --set secrets.huggingfaceToken=$HF_TOKEN

# Install in production
helm install music-producer . \
  -f values-production.yaml \
  --namespace music-producer \
  --create-namespace \
  --set secrets.huggingfaceToken=$HF_TOKEN \
  --set secrets.anthropicApiKey=$ANTHROPIC_API_KEY
```

### Building Docker Images

```bash
# Build CPU image
docker build --target runtime-cpu -t music-producer:latest .

# Build GPU image
docker build --target runtime-gpu -t music-producer:latest-gpu .

# Build job runner
docker build --target job-runner -t music-producer:job-runner .

# Push to registry
docker tag music-producer:latest ghcr.io/yourusername/music-producer:latest
docker push ghcr.io/yourusername/music-producer:latest
```

### Spawning GPU Jobs

The job template can be used to spawn music generation jobs:

```bash
# Create a generation job
helm template music-producer . \
  --set jobConfig.jobId="gen-$(uuidgen)" \
  --set jobConfig.prompt="Create a chill lofi beat" \
  --set jobConfig.duration=120 \
  -s templates/job.yaml | kubectl apply -f -

# Watch job progress
kubectl get jobs -w

# Check logs
kubectl logs job/music-producer-generation-<job-id>
```

### Environment-Specific Values

| File | Use Case |
|------|----------|
| `values.yaml` | Default values |
| `values-staging.yaml` | Staging (reduced resources, T4 GPUs) |
| `values-production.yaml` | Production (full resources, A100 GPUs) |

### Monitoring

The chart includes:

1. **ServiceMonitor**: Auto-discovered by Prometheus Operator
2. **PrometheusRule**: Alerting rules for error rate, latency, job failures
3. **Grafana Dashboard**: Auto-provisioned dashboard with key metrics

Access Grafana and look for the "Music Producer Dashboard".

### Resource Requirements

| Component | CPU Request | Memory Request | GPU |
|-----------|-------------|----------------|-----|
| API Server | 500m | 1Gi | - |
| GPU Worker | 2000m | 8Gi | 1x (T4/A100) |
| Redis | 100m | 128Mi | - |

### Troubleshooting

```bash
# Check pod status
kubectl get pods -n music-producer

# View API logs
kubectl logs -f deployment/music-producer-api -n music-producer

# Check GPU node availability
kubectl get nodes -l nvidia.com/gpu.present=true

# Debug job failures
kubectl describe job/music-producer-generation-<id> -n music-producer
```

## License

MIT

## Contributing

1. Fork the repo
2. Create a feature branch
3. Make changes with tests
4. Submit a PR

---

Built with [LangGraph](https://github.com/langchain-ai/langgraph) 🦜🕸️
