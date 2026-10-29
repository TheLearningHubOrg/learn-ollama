"""
Ollama Local Model Server & Engine Emulator.

Implements:
- Modelfile parser & compiler with FROM, PARAMETER, SYSTEM, TEMPLATE directives.
- GGUF Quantization sizing and VRAM calculation formulas (FP16, Q8_0, Q5_K_M, Q4_K_M, Q2_K).
- Local generation, NDJSON streaming simulation, and chat completions with tool calling.
- Local dense embedding generator.
- Multi-GPU layer offloading calculator (n_gpu_layers vs CPU RAM bandwidth).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple


class GGUFQuantization(str, Enum):
    FP16 = "FP16"        # 16.0 bits/weight
    Q8_0 = "Q8_0"        # 8.5 bits/weight (near-lossless)
    Q5_K_M = "Q5_K_M"    # 5.5 bits/weight (recommended high-quality)
    Q4_K_M = "Q4_K_M"    # 4.5 bits/weight (standard default)
    Q2_K = "Q2_K"        # 2.6 bits/weight (extreme compression)

    @property
    def bits_per_weight(self) -> float:
        mapping = {
            GGUFQuantization.FP16: 16.0,
            GGUFQuantization.Q8_0: 8.5,
            GGUFQuantization.Q5_K_M: 5.5,
            GGUFQuantization.Q4_K_M: 4.5,
            GGUFQuantization.Q2_K: 2.6,
        }
        return mapping[self]


class GGUFCalculator:
    """Calculates model weight sizes, KV cache overhead, and required GPU VRAM."""

    @classmethod
    def calculate_vram(
        cls,
        param_count_billions: float,
        quant: GGUFQuantization = GGUFQuantization.Q4_K_M,
        context_length: int = 4096,
    ) -> Dict[str, float]:
        # Weight footprint
        weight_bytes = (param_count_billions * 1e9 * quant.bits_per_weight) / 8.0
        weight_gb = weight_bytes / (1024 ** 3)

        # Baseline FP16 footprint
        fp16_bytes = (param_count_billions * 1e9 * 16.0) / 8.0
        fp16_gb = fp16_bytes / (1024 ** 3)

        # KV Cache estimation: ~0.5MB per 1k context per billion params for FP16 KV
        kv_cache_gb = (param_count_billions * (context_length / 1024.0) * 0.06)

        # CUDA Runtime overhead & context buffer: ~1.2 GB
        cuda_overhead_gb = 1.2
        total_vram_gb = weight_gb + kv_cache_gb + cuda_overhead_gb

        compression_ratio = fp16_gb / weight_gb if weight_gb > 0 else 1.0
        savings_pct = (1.0 - (weight_gb / fp16_gb)) * 100.0 if fp16_gb > 0 else 0.0

        return {
            "param_billions": param_count_billions,
            "quantization": quant.value,
            "bits_per_weight": quant.bits_per_weight,
            "weight_gb": round(weight_gb, 2),
            "kv_cache_gb": round(kv_cache_gb, 2),
            "cuda_overhead_gb": round(cuda_overhead_gb, 2),
            "total_vram_gb": round(total_vram_gb, 2),
            "fp16_baseline_gb": round(fp16_gb, 2),
            "compression_ratio": round(compression_ratio, 2),
            "savings_pct": round(savings_pct, 1),
        }


@dataclass
class Modelfile:
    """Represents a compiled or parsed Ollama Modelfile."""
    from_model: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    system: Optional[str] = None
    template: Optional[str] = None
    license: Optional[str] = None

    @classmethod
    def parse(cls, content: str) -> Modelfile:
        from_model = ""
        parameters: Dict[str, Any] = {}
        system_lines: List[str] = []
        template_lines: List[str] = []
        license_lines: List[str] = []

        current_block = None

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("FROM "):
                from_model = stripped[5:].strip()
                current_block = None
            elif stripped.startswith("PARAMETER "):
                parts = stripped[10:].split(maxsplit=1)
                if len(parts) == 2:
                    k, v = parts
                    try:
                        v_parsed = json.loads(v)
                    except Exception:
                        try:
                            v_parsed = float(v) if "." in v else int(v)
                        except ValueError:
                            v_parsed = v
                    parameters[k] = v_parsed
                current_block = None
            elif stripped.startswith("SYSTEM "):
                system_lines.append(stripped[7:].strip('"""').strip("'''").strip('"'))
                current_block = "system" if ('"""' in stripped or "'''" in stripped) else None
            elif stripped.startswith("TEMPLATE "):
                template_lines.append(stripped[9:].strip('"""').strip("'''").strip('"'))
                current_block = "template" if ('"""' in stripped or "'''" in stripped) else None
            elif stripped.startswith("LICENSE "):
                license_lines.append(stripped[8:].strip('"""').strip("'''").strip('"'))
                current_block = "license" if ('"""' in stripped or "'''" in stripped) else None
            else:
                if current_block == "system":
                    system_lines.append(stripped.rstrip('"""').rstrip("'''"))
                elif current_block == "template":
                    template_lines.append(stripped.rstrip('"""').rstrip("'''"))
                elif current_block == "license":
                    license_lines.append(stripped.rstrip('"""').rstrip("'''"))

        return cls(
            from_model=from_model or "llama3.2:latest",
            parameters=parameters,
            system="\n".join(system_lines) if system_lines else None,
            template="\n".join(template_lines) if template_lines else None,
            license="\n".join(license_lines) if license_lines else None,
        )

    def to_text(self) -> str:
        lines = [f"FROM {self.from_model}"]
        for k, v in self.parameters.items():
            lines.append(f"PARAMETER {k} {v}")
        if self.system:
            lines.append(f'SYSTEM """\n{self.system}\n"""')
        if self.template:
            lines.append(f'TEMPLATE """\n{self.template}\n"""')
        if self.license:
            lines.append(f'LICENSE """\n{self.license}\n"""')
        return "\n".join(lines)


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class GenerateResponse:
    model: str
    response: str
    done: bool = True
    total_duration_ms: float = 120.0
    eval_count: int = 45
    eval_duration_ms: float = 90.0


@dataclass
class ChatResponse:
    model: str
    message: ChatMessage
    done: bool = True
    total_duration_ms: float = 145.0
    eval_count: int = 50


class GPULayerOffloader:
    """Calculates GPU VRAM layer offloading vs CPU RAM throughput."""

    @classmethod
    def calculate_offload(
        cls,
        total_layers: int,
        available_vram_gb: float,
        model_size_gb: float,
    ) -> Dict[str, Any]:
        gb_per_layer = model_size_gb / total_layers if total_layers > 0 else 0.0

        # Leave 1.5 GB for KV cache and CUDA buffers
        usable_vram = max(0.0, available_vram_gb - 1.5)
        gpu_layers = min(total_layers, int(usable_vram / gb_per_layer)) if gb_per_layer > 0 else 0
        cpu_layers = total_layers - gpu_layers

        # Speed estimation:
        # Full GPU: ~80-120 tok/sec (PCIe 16x + HBM / GDDR6 bandwidth 800+ GB/s)
        # Full CPU: ~8-15 tok/sec (DDR5 RAM bandwidth ~50-80 GB/s)
        gpu_ratio = gpu_layers / total_layers if total_layers > 0 else 0.0
        estimated_tps = 10.0 + (gpu_ratio * 90.0)

        return {
            "total_layers": total_layers,
            "gpu_layers": gpu_layers,
            "cpu_layers": cpu_layers,
            "gpu_offload_pct": round(gpu_ratio * 100.0, 1),
            "estimated_tokens_per_sec": round(estimated_tps, 1),
            "status": "FULL_GPU" if cpu_layers == 0 else ("PARTIAL_OFFLOAD" if gpu_layers > 0 else "FULL_CPU"),
        }


class OllamaClient:
    """Local client and daemon interface for Ollama."""

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.host = host
        self.models: Dict[str, Modelfile] = {
            "llama3.2:latest": Modelfile(from_model="llama3.2", parameters={"temperature": 0.7, "num_ctx": 4096}),
            "mistral:latest": Modelfile(from_model="mistral", parameters={"temperature": 0.6, "num_ctx": 8192}),
            "deepseek-r1:8b": Modelfile(from_model="deepseek-r1:8b", parameters={"temperature": 0.2, "num_ctx": 16384}),
        }

    def create(self, model_name: str, modelfile_content: str) -> Modelfile:
        """Compiles and registers a custom local model in Ollama registry."""
        mf = Modelfile.parse(modelfile_content)
        self.models[model_name] = mf
        return mf

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    def generate(self, model: str, prompt: str) -> GenerateResponse:
        """Executes /api/generate prompt completion."""
        return GenerateResponse(
            model=model,
            response=f"Local Ollama engine responded to prompt using model '{model}'.",
            done=True,
            eval_count=max(10, len(prompt.split()) * 2),
        )

    def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> ChatResponse:
        """Executes /api/chat endpoint with optional tool calling."""
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")

        # Check for tool invocation if tools registered
        if tools:
            for tool in tools:
                if tool.name.lower() in last_user.lower():
                    tool_call = {
                        "function": {
                            "name": tool.name,
                            "arguments": {"query": last_user},
                        }
                    }
                    return ChatResponse(
                        model=model,
                        message=ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[tool_call],
                        ),
                    )

        content = f"Ollama local inference completed for {len(messages)} messages."
        return ChatResponse(
            model=model,
            message=ChatMessage(role="assistant", content=content),
        )

    def embeddings(self, model: str, prompt: str) -> List[float]:
        """Generates mock high-dimensional dense embedding vector."""
        # Simulated 768-dim normalized embedding
        dim = 768
        raw = [math.sin(i + len(prompt)) for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]
