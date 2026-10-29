"""
Pedagogical Roadmap: 7-Stage Ollama Local LLM Systems Masterclass.

Tracks curriculum progress across Architecture, Modelfiles, GGUF Quantization,
REST/OpenAI APIs, Local RAG, Multi-GPU Orchestration, and Staff Systems Mastery.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class StageStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


@dataclass
class Topic:
    id: str
    title: str
    summary: str
    completed: bool = False


@dataclass
class Stage:
    stage_number: int
    title: str
    description: str
    topics: List[Topic]
    status: StageStatus = StageStatus.NOT_STARTED

    @property
    def progress_percentage(self) -> float:
        if not self.topics:
            return 0.0
        completed = sum(1 for t in self.topics if t.completed)
        return (completed / len(self.topics)) * 100.0


class OllamaRoadmap:
    """Manages progress and milestones for the 7-Stage Ollama curriculum."""

    def __init__(self) -> None:
        self.stages: Dict[int, Stage] = {
            1: Stage(
                stage_number=1,
                title="Ollama Architecture & Core Engine Internals",
                description="Go server daemon, llama.cpp dynamic C bindings, unified memory, and cold-start model weight mmap mapping.",
                topics=[
                    Topic("1.1", "Daemon Architecture & llama.cpp Bindings", "How the Ollama Go daemon wraps llama.cpp runners using cgo and dynamic libraries.", True),
                    Topic("1.2", "Memory Management & MMAP", "Zero-copy weight loading from disk to unified CPU/GPU memory using mmap system calls.", True),
                    Topic("1.3", "Process Lifecycle & Keep-Alive", "Managing OLLAMA_KEEP_ALIVE idle unloading and auto-swapping models in VRAM.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            2: Stage(
                stage_number=2,
                title="Modelfiles & Custom Model Engineering",
                description="Modelfile syntax (FROM, PARAMETER, SYSTEM, TEMPLATE), Jinja2 chat templating, and custom persona building.",
                topics=[
                    Topic("2.1", "Modelfile Directives", "FROM base model, PARAMETER temperature, num_ctx, top_k, and stop token declarations.", True),
                    Topic("2.2", "Chat Template Formatting", "Implementing ChatML, Llama 3, and Mistral chat templates using Go/Jinja templates.", True),
                    Topic("2.3", "Packaging & Exporting", "Compiling custom Modelfiles with 'ollama create' and pushing to model registries.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            3: Stage(
                stage_number=3,
                title="GGUF Quantization & Model Ingestion",
                description="Converting Safetensors/PyTorch weights to GGUF, k-quant schemes (Q4_K_M, Q8_0, Q2_K), and VRAM calculation formulas.",
                topics=[
                    Topic("3.1", "GGUF File Format Specification", "Header metadata, key-value tensor dictionary, and alignment padding mechanics.", True),
                    Topic("3.2", "K-Quantization Schemes", "Evaluating perplexity loss vs memory savings across Q4_K_M, Q5_K_M, and Q8_0.", True),
                    Topic("3.3", "VRAM Formula & Sizing", "Calculating exact GPU memory requirements: (Weights * bits/8) + KV cache overhead.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            4: Stage(
                stage_number=4,
                title="REST API, OpenAI Compatibility & Tool/Function Calling",
                description="Native NDJSON streaming (/api/generate, /api/chat), drop-in OpenAI /v1 compatibility, and JSON schema tool calling.",
                topics=[
                    Topic("4.1", "Native REST API Endpoints", "Consuming /api/generate and /api/chat with streaming NDJSON deltas.", True),
                    Topic("4.2", "OpenAI Drop-in Compatibility", "Switching base_url to http://localhost:11434/v1 in standard OpenAI SDKs.", True),
                    Topic("4.3", "Local Function & Tool Calling", "Grammar-constrained JSON schema generation for deterministic local tool calling.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            5: Stage(
                stage_number=5,
                title="Python & TypeScript SDKs with Local RAG Integration",
                description="Official ollama-python / ollama-js, nomic-embed-text embeddings, Chroma/pgvector integration, and local RAG pipelines.",
                topics=[
                    Topic("5.1", "Python & JS SDK Mechanics", "Client instantiations, asynchronous generators, and embedding batching.", True),
                    Topic("5.2", "Local Embeddings Generation", "Using nomic-embed-text / mxbai-embed-large for zero-cloud privacy embeddings.", True),
                    Topic("5.3", "End-to-End Local RAG", "Building fully offline RAG search without a single external cloud API call.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            6: Stage(
                stage_number=6,
                title="Production Infrastructure, Multi-GPU Scaling & Orchestration",
                description="Docker containerization, NVIDIA CUDA container toolkit, OLLAMA_NUM_PARALLEL concurrent slots, and GPU layer offloading.",
                topics=[
                    Topic("6.1", "Multi-GPU Layer Offloading", "Splitting transformer layers across multiple GPUs with n_gpu_layers tuning.", True),
                    Topic("6.2", "High-Throughput Concurrent Serving", "Configuring OLLAMA_NUM_PARALLEL and OLLAMA_MAX_LOADED_MODELS.", True),
                    Topic("6.3", "Production Kubernetes & Docker", "Deploying Ollama on k8s clusters with NVIDIA GPU device plugins.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            7: Stage(
                stage_number=7,
                title="Production API Reference & 50 Staff-Level Interview Questions",
                description="Complete REST API specifications, system failure recovery runbooks, and staff-level interview technical questions.",
                topics=[
                    Topic("7.1", "Complete API Reference", "Exhaustive documentation of all endpoints, parameters, and error codes.", True),
                    Topic("7.2", "Operational Recovery Runbooks", "OOM troubleshooting, GPU memory leaks, and broken pipe socket recovery.", True),
                    Topic("7.3", "50 Staff-Level Questions", "Mastery of quantization mechanics, memory bandwidth limits, and local serving.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
        }

    def get_stage(self, stage_number: int) -> Optional[Stage]:
        return self.stages.get(stage_number)

    def total_progress(self) -> float:
        total_topics = sum(len(s.topics) for s in self.stages.values())
        if total_topics == 0:
            return 0.0
        completed = sum(sum(1 for t in s.topics if t.completed) for s in self.stages.values())
        return (completed / total_topics) * 100.0

    def get_summary(self) -> Dict[str, object]:
        return {
            "total_stages": len(self.stages),
            "completed_stages": sum(1 for s in self.stages.values() if s.status == StageStatus.COMPLETED),
            "overall_progress_pct": round(self.total_progress(), 2),
            "stages": [
                {
                    "stage": s.stage_number,
                    "title": s.title,
                    "status": s.status.value,
                    "progress_pct": round(s.progress_percentage, 1),
                    "topics_count": len(s.topics),
                }
                for s in self.stages.values()
            ],
        }
