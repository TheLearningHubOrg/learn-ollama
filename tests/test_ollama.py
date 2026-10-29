"""
Comprehensive Unit Tests for Ollama Local Model Engine.

Tests:
- Modelfile parsing, parameter validation, and text serialization.
- GGUF Quantization VRAM calculations and compression ratios.
- Multi-GPU layer offloading mechanics.
- Local generation, chat completions, and tool calling.
- Local dense embedding generation.
- Pedagogical roadmap stage completion.
"""

import unittest

from src.learn_ollama.engine import (
    ChatMessage,
    GGUFCalculator,
    GGUFQuantization,
    GPULayerOffloader,
    Modelfile,
    OllamaClient,
    ToolDefinition,
)
from src.learn_ollama.roadmap import OllamaRoadmap


class TestOllamaEngine(unittest.TestCase):

    def setUp(self) -> None:
        self.client = OllamaClient()

    def test_client_initialization(self) -> None:
        self.assertIn("llama3.2:latest", self.client.models)
        self.assertIn("mistral:latest", self.client.models)
        self.assertTrue(len(self.client.list_models()) >= 3)

    def test_modelfile_parse_and_serialize(self) -> None:
        raw_text = """
FROM mistral:7b
PARAMETER temperature 0.3
PARAMETER top_p 0.9
SYSTEM \"\"\"
You are a senior site reliability engineer.
\"\"\"
"""
        mf = Modelfile.parse(raw_text)
        self.assertEqual(mf.from_model, "mistral:7b")
        self.assertEqual(mf.parameters["temperature"], 0.3)
        self.assertEqual(mf.parameters["top_p"], 0.9)
        self.assertIn("reliability engineer", mf.system or "")

        serialized = mf.to_text()
        self.assertIn("FROM mistral:7b", serialized)
        self.assertIn("PARAMETER temperature 0.3", serialized)

    def test_gguf_vram_calculator(self) -> None:
        # Llama 3 8B with Q4_K_M (4.5 bits/weight)
        calc = GGUFCalculator.calculate_vram(8.0, GGUFQuantization.Q4_K_M, context_length=4096)
        self.assertEqual(calc["param_billions"], 8.0)
        self.assertEqual(calc["quantization"], "Q4_K_M")
        # 8B * 4.5 / 8 = 4.5 * 10^9 bytes = ~4.19 GB
        self.assertTrue(3.5 < calc["weight_gb"] < 5.0)
        self.assertTrue(calc["total_vram_gb"] < 8.0)  # Fits in 8GB VRAM GPU!
        self.assertTrue(calc["compression_ratio"] > 3.0)  # >3x compression vs FP16

    def test_gpu_layer_offloader(self) -> None:
        # 32 layers, 14 GB model on 8 GB GPU (6.5 GB usable -> ~14 layers fit)
        offload = GPULayerOffloader.calculate_offload(
            total_layers=32,
            available_vram_gb=8.0,
            model_size_gb=14.0,
        )
        self.assertEqual(offload["total_layers"], 32)
        self.assertTrue(offload["gpu_layers"] > 0)
        self.assertTrue(offload["cpu_layers"] > 0)
        self.assertEqual(offload["status"], "PARTIAL_OFFLOAD")

        # Full GPU scenario: 24 GB GPU for 14 GB model
        full_offload = GPULayerOffloader.calculate_offload(
            total_layers=32,
            available_vram_gb=24.0,
            model_size_gb=14.0,
        )
        self.assertEqual(full_offload["gpu_layers"], 32)
        self.assertEqual(full_offload["cpu_layers"], 0)
        self.assertEqual(full_offload["status"], "FULL_GPU")

    def test_generate_and_chat(self) -> None:
        gen = self.client.generate("llama3.2:latest", "Hello local AI")
        self.assertEqual(gen.model, "llama3.2:latest")
        self.assertTrue(gen.done)
        self.assertIn("Ollama", gen.response)

        # Standard chat
        chat = self.client.chat("llama3.2:latest", [ChatMessage(role="user", content="Hi")])
        self.assertEqual(chat.message.role, "assistant")
        self.assertTrue(len(chat.message.content) > 0)

        # Tool calling chat
        tool = ToolDefinition(
            name="system_reboot",
            description="Reboots the server node.",
            parameters={"type": "object", "properties": {"node": {"type": "string"}}},
        )
        chat_tool = self.client.chat(
            "llama3.2:latest",
            [ChatMessage(role="user", content="Please call system_reboot immediately.")],
            tools=[tool],
        )
        self.assertIsNotNone(chat_tool.message.tool_calls)
        self.assertEqual(chat_tool.message.tool_calls[0]["function"]["name"], "system_reboot")

    def test_local_embeddings(self) -> None:
        vec = self.client.embeddings("nomic-embed-text", "Embed this text locally.")
        self.assertEqual(len(vec), 768)
        # Vector must be normalized (norm ~ 1.0)
        norm = sum(x * x for x in vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_roadmap_progress(self) -> None:
        roadmap = OllamaRoadmap()
        self.assertEqual(len(roadmap.stages), 7)
        self.assertEqual(roadmap.total_progress(), 100.0)
        summary = roadmap.get_summary()
        self.assertEqual(summary["total_stages"], 7)
        self.assertEqual(summary["completed_stages"], 7)
        self.assertEqual(summary["overall_progress_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
