from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from src.hf_models import DEFAULT_MODEL, create_huggingface_models
from src.prompts import (
    DEMO_PROMPTS,
    PromptResult,
    run_chat_prompt,
    run_normal_prompt,
    run_prompt_comparison,
    write_markdown_report,
)


def _print_results(results: Sequence[PromptResult]) -> None:
    for index, result in enumerate(results, start=1):
        print(f"\n{'=' * 72}\nExample {index}\n{'=' * 72}")
        print(f"Question: {result.question}")
        if result.normal_output is not None:
            print(f"\nPlain Prompt:\n{result.normal_output}")
        if result.chat_output is not None:
            print(f"\nChat Prompt:\n{result.chat_output}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Compare Hugging Face plain and chat prompt chains with LangChain."
    )
    parser.add_argument(
        "--mode",
        choices=("normal", "chat", "compare"),
        default="compare",
        help="Run a plain PromptTemplate, a chat prompt, or both for comparison.",
    )
    parser.add_argument(
        "--prompt",
        help="Run one custom question. Without it, compare several built-in prompts.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Hugging Face model ID (defaults to HF_MODEL or {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output",
        help="Optional Markdown report path. For example: outputs/prompt_comparison.md",
    )
    args = parser.parse_args()

    model_id = args.model or os.getenv("HF_MODEL", DEFAULT_MODEL)
    llm, chat_model = create_huggingface_models(model_id=model_id)
    questions = [args.prompt] if args.prompt else list(DEMO_PROMPTS)

    if args.mode == "compare":
        results = run_prompt_comparison(llm, chat_model, questions)
    elif args.mode == "normal":
        results = [
            PromptResult(question=question, normal_output=run_normal_prompt(llm, question))
            for question in questions
        ]
    else:
        results = [
            PromptResult(question=question, chat_output=run_chat_prompt(chat_model, question))
            for question in questions
        ]

    _print_results(results)
    if args.output:
        report_path = write_markdown_report(results, model_id, Path(args.output))
        print(f"\nMarkdown report saved to: {report_path}")


if __name__ == "__main__":
    main()
