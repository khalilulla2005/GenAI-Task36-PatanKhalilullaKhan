from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.hf_models import create_huggingface_models
from src.prompts import (
    DEMO_PROMPTS,
    PromptResult,
    run_chat_prompt,
    run_normal_prompt,
    run_prompt_comparison,
    write_markdown_report,
)


class FakeChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "prompt-comparison-test-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = str(messages[-1].content)
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=f"Mock response: {content}"))
            ]
        )


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.fake_model = FakeChatModel()

    def test_normal_and_chat_prompts_generate_responses(self):
        normal = run_normal_prompt(self.fake_model, "Explain gravity.")
        chat = run_chat_prompt(self.fake_model, "Explain gravity.")

        self.assertIn("Explain gravity.", normal)
        self.assertIn("Explain gravity.", chat)
        self.assertTrue(normal.startswith("Mock response:"))
        self.assertTrue(chat.startswith("Mock response:"))

    def test_multiple_prompts_compare_both_templates_with_same_model(self):
        results = run_prompt_comparison(
            self.fake_model,
            self.fake_model,
            questions=DEMO_PROMPTS,
        )

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.normal_output for result in results))
        self.assertTrue(all(result.chat_output for result in results))

    def test_empty_prompts_are_rejected(self):
        with self.assertRaises(ValueError):
            run_normal_prompt(self.fake_model, "  ")
        with self.assertRaises(ValueError):
            run_prompt_comparison(self.fake_model, self.fake_model, questions=[])
        with self.assertRaises(ValueError):
            run_prompt_comparison(self.fake_model, self.fake_model, questions=[""])

    def test_markdown_report_has_question_and_both_outputs(self):
        results = [
            PromptResult(
                question="What is LangChain?",
                normal_output="Plain result.",
                chat_output="Chat result.",
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            report_path = write_markdown_report(
                results,
                "example/model",
                Path(directory) / "reports" / "comparison.md",
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("example/model", report)
        self.assertIn("What is LangChain?", report)
        self.assertIn("Plain result.", report)
        self.assertIn("Chat result.", report)

    def test_model_factory_requires_hugging_face_token(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "HUGGINGFACEHUB_API_TOKEN"):
                create_huggingface_models()

    def test_model_factory_wraps_one_hugging_face_endpoint_as_chat_model(self):
        created = {}

        class FakeEndpoint:
            def __init__(self, **kwargs):
                created.update(kwargs)

        class FakeChatHuggingFace:
            def __init__(self, llm):
                self.llm = llm

        fake_package = types.ModuleType("langchain_huggingface")
        fake_package.HuggingFaceEndpoint = FakeEndpoint
        fake_package.ChatHuggingFace = FakeChatHuggingFace
        environment = {
            "HUGGINGFACEHUB_API_TOKEN": "test-token",
            "HF_MODEL": "example/model",
            "HF_PROVIDER": "auto",
            "HF_MAX_NEW_TOKENS": "128",
        }
        with patch.dict(sys.modules, {"langchain_huggingface": fake_package}):
            with patch.dict(os.environ, environment, clear=True):
                endpoint, chat_model = create_huggingface_models()

        self.assertIsInstance(endpoint, FakeEndpoint)
        self.assertIsInstance(chat_model, FakeChatHuggingFace)
        self.assertIs(chat_model.llm, endpoint)
        self.assertEqual(created["repo_id"], "example/model")
        self.assertEqual(created["task"], "text-generation")
        self.assertEqual(created["max_new_tokens"], 128)
        self.assertEqual(created["huggingfacehub_api_token"], "test-token")


if __name__ == "__main__":
    unittest.main()
