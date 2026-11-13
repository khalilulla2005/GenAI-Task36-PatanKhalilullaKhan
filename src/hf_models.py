from __future__ import annotations

import os

from dotenv import load_dotenv

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def create_huggingface_models(
    model_id: str | None = None,
    token: str | None = None,
):
    """Create plain-text and chat LangChain wrappers over one HF model endpoint."""
    load_dotenv()
    access_token = token or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not access_token:
        raise ValueError(
            "Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) in the environment or .env file."
        )

    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    selected_model = model_id or os.getenv("HF_MODEL", DEFAULT_MODEL)
    endpoint = HuggingFaceEndpoint(
        repo_id=selected_model,
        task="text-generation",
        max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "256")),
        do_sample=False,
        provider=os.getenv("HF_PROVIDER", "auto"),
        huggingfacehub_api_token=access_token,
    )
    return endpoint, ChatHuggingFace(llm=endpoint)
