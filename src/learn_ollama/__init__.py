"""
learn_ollama: Production-Grade Ollama Local Model Engineering Engine.

Provides Modelfile parsing, GGUF quantization VRAM calculators, streaming NDJSON endpoints,
OpenAI-compatible server interfaces, tool calling, and multi-GPU layer offloading.
"""

from .engine import (
    OllamaClient,
    Modelfile,
    GGUFQuantization,
    GGUFCalculator,
    ChatMessage,
    GenerateResponse,
    ChatResponse,
    ToolDefinition,
    GPULayerOffloader,
)
from .roadmap import OllamaRoadmap

__all__ = [
    "OllamaClient",
    "Modelfile",
    "GGUFQuantization",
    "GGUFCalculator",
    "ChatMessage",
    "GenerateResponse",
    "ChatResponse",
    "ToolDefinition",
    "GPULayerOffloader",
    "OllamaRoadmap",
]
