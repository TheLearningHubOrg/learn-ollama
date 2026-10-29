"""
Ollama Local Model Systems Masterclass: CLI & Demonstrator.

Demonstrates:
1. 7-Stage Curriculum progress tracking.
2. Modelfile compilation and custom model registration.
3. GGUF Quantization and VRAM footprint calculations across Llama 3 models.
4. Multi-GPU layer offloading simulation.
5. Local /api/chat with grammar-constrained tool calling.
6. Local embeddings generation with zero external network dependencies.
"""

from src.learn_ollama.engine import (
    ChatMessage,
    GGUFCalculator,
    GGUFQuantization,
    GPULayerOffloader,
    OllamaClient,
    ToolDefinition,
)
from src.learn_ollama.roadmap import OllamaRoadmap


def run_roadmap_showcase() -> None:
    roadmap = OllamaRoadmap()
    summary = roadmap.get_summary()

    print("=" * 80)
    print("OLLAMA LOCAL LLM SYSTEMS MASTERCLASS: 7-STAGE CURRICULUM")
    print("=" * 80)
    print(f"Curriculum Completion: {summary['overall_progress_pct']}% | Stages: {summary['completed_stages']}/{summary['total_stages']} Completed\n")

    for s in summary["stages"]:
        status_symbol = "[x]" if s["status"] == "COMPLETED" else "[ ]"
        print(f"  {status_symbol} Stage {s['stage']}: {s['title']} ({s['topics_count']} topics, {s['progress_pct']}%)")
    print("=" * 80 + "\n")


def run_modelfile_demo(client: OllamaClient) -> None:
    print("-" * 80)
    print("STAGE 2: MODELFILE COMPILATION & CUSTOM MODEL ENGINEERING")
    print("-" * 80)

    modelfile_content = """
FROM llama3.2:latest
PARAMETER temperature 0.2
PARAMETER num_ctx 8192
PARAMETER stop "<|eot_id|>"
SYSTEM \"\"\"
You are a staff infrastructure engineer operating high-concurrency systems.
Provide direct, concise, production-ready code.
\"\"\"
"""
    custom_model_name = "sre-agent:latest"
    compiled_mf = client.create(custom_model_name, modelfile_content)

    print(f"Compiled Modelfile for '{custom_model_name}':")
    print(f"  Base Model:   {compiled_mf.from_model}")
    print(f"  Parameters:   {compiled_mf.parameters}")
    print(f"  System Text:  \"{compiled_mf.system}\"")
    print(f"Registry Total: {len(client.list_models())} models registered locally.\n")


def run_gguf_vram_demo() -> None:
    print("-" * 80)
    print("STAGE 3: GGUF QUANTIZATION & VRAM SIZING (LLAMA 3.1 8B & 70B)")
    print("-" * 80)

    models = [
        ("Llama 3.1 8B", 8.0),
        ("Llama 3.1 70B", 70.0),
    ]

    for model_name, params in models:
        print(f"Model: {model_name} ({params}B Parameters, Context: 4,096 tokens):")
        print(f"  {'Quant Scheme':<12} | {'Bits/Weight':<12} | {'Weights GB':<12} | {'Total VRAM GB':<15} | {'Compression':<12}")
        print("  " + "-" * 70)

        for quant in [GGUFQuantization.FP16, GGUFQuantization.Q8_0, GGUFQuantization.Q5_K_M, GGUFQuantization.Q4_K_M, GGUFQuantization.Q2_K]:
            calc = GGUFCalculator.calculate_vram(params, quant, context_length=4096)
            print(f"  {quant.value:<12} | {calc['bits_per_weight']:<12.1f} | {calc['weight_gb']:<12.2f} | {calc['total_vram_gb']:<15.2f} | {calc['compression_ratio']:<4.2f}x ({calc['savings_pct']}%)")
        print()


def run_gpu_offload_demo() -> None:
    print("-" * 80)
    print("STAGE 6: MULTI-GPU LAYER OFFLOADING & MEMORY BANDWIDTH")
    print("-" * 80)

    # 32 transformer layers, 16 GB model size on an 8 GB consumer GPU (RTX 3070 / 4060)
    total_layers = 32
    available_vram = 8.0
    model_size = 14.0

    offload = GPULayerOffloader.calculate_offload(
        total_layers=total_layers,
        available_vram_gb=available_vram,
        model_size_gb=model_size,
    )

    print(f"Hardware Scenario: {available_vram} GB GPU VRAM vs {model_size} GB Model ({total_layers} layers)")
    print(f"  GPU Offloaded Layers: {offload['gpu_layers']}/{offload['total_layers']} ({offload['gpu_offload_pct']}%)")
    print(f"  CPU Offloaded Layers: {offload['cpu_layers']}/{offload['total_layers']}")
    print(f"  Offload Status:       {offload['status']}")
    print(f"  Estimated Generation: {offload['estimated_tokens_per_sec']} tokens/sec (Limited by DDR5 RAM bandwidth)\n")


def run_chat_demo(client: OllamaClient) -> None:
    print("-" * 80)
    print("STAGE 4 & 5: LOCAL /api/chat & TOOL CALLING")
    print("-" * 80)

    kubectl_tool = ToolDefinition(
        name="kubectl_get_nodes",
        description="Lists all nodes and their status in the local Kubernetes cluster.",
        parameters={
            "type": "object",
            "properties": {"namespace": {"type": "string"}},
        },
    )

    messages = [
        ChatMessage(role="user", content="Please call kubectl_get_nodes to check cluster health.")
    ]

    response = client.chat(
        model="llama3.2:latest",
        messages=messages,
        tools=[kubectl_tool],
    )

    print(f"Chat Response from 'llama3.2:latest':")
    if response.message.tool_calls:
        tc = response.message.tool_calls[0]["function"]
        print(f"  Tool Call Dispatched: {tc['name']}({tc['arguments']})")
    else:
        print(f"  Content: {response.message.content}")

    # Embeddings test
    embed_vec = client.embeddings(model="nomic-embed-text", prompt="Search query for vector database.")
    print(f"  Generated Local Embedding: D={len(embed_vec)} dimensions (Zero Cloud Egress)\n")


def main() -> None:
    client = OllamaClient()
    run_roadmap_showcase()
    run_modelfile_demo(client)
    run_gguf_vram_demo()
    run_gpu_offload_demo()
    run_chat_demo(client)


if __name__ == "__main__":
    main()
