from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

NORMAL_PROMPT_TEMPLATE = PromptTemplate.from_template(
    "Answer the following question accurately and clearly. "
    "Use concise language and explain unfamiliar terms.\n\n"
    "Question: {question}\n\nAnswer:"
)

CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, accurate assistant. Explain answers clearly, "
            "use concise language, and follow the user's requested format.",
        ),
        ("human", "{question}"),
    ]
)

DEMO_PROMPTS = (
    "Explain photosynthesis in two sentences for a middle-school student.",
    "Give three practical applications of machine learning.",
    "State one benefit and one limitation of large language models.",
)


@dataclass(frozen=True)
class PromptResult:
    question: str
    normal_output: str | None = None
    chat_output: str | None = None


def _output_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if text_parts:
            return "\n".join(text_parts).strip()
    return str(content).strip()


def run_normal_prompt(llm, question: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    result = (NORMAL_PROMPT_TEMPLATE | llm).invoke({"question": question.strip()})
    return _output_text(result)


def run_chat_prompt(chat_model, question: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    result = (CHAT_PROMPT_TEMPLATE | chat_model).invoke({"question": question.strip()})
    return _output_text(result)


def run_prompt_comparison(
    llm,
    chat_model,
    questions: Sequence[str] = DEMO_PROMPTS,
) -> list[PromptResult]:
    if not questions:
        raise ValueError("Provide at least one question to compare.")
    results = []
    for question in questions:
        if not question.strip():
            raise ValueError("Questions cannot be empty.")
        results.append(
            PromptResult(
                question=question.strip(),
                normal_output=run_normal_prompt(llm, question),
                chat_output=run_chat_prompt(chat_model, question),
            )
        )
    return results


def write_markdown_report(
    results: Sequence[PromptResult],
    model_id: str,
    output_path: str | Path,
) -> Path:
    if not results:
        raise ValueError("Cannot write a comparison report without results.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        "# Hugging Face prompt comparison",
        "",
        f"- **Model:** `{model_id}`",
        f"- **Questions compared:** {len(results)}",
        "",
        "Each question was sent to the same Hugging Face model using a plain `PromptTemplate` "
        "and a `ChatPromptTemplate` with separate system and human messages.",
    ]
    for index, result in enumerate(results, start=1):
        sections.extend(
            [
                "",
                f"## Example {index}",
                "",
                f"**Question:** {result.question}",
                "",
                "### Plain prompt",
                "",
                result.normal_output or "_Not run._",
                "",
                "### Chat prompt",
                "",
                result.chat_output or "_Not run._",
            ]
        )
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return path
