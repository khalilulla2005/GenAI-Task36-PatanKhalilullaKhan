# Assignment 36 — Hugging Face + LangChain Prompt Integration

This project replaces a closed-model API with a Hugging Face model served through LangChain. It compares a plain `PromptTemplate` with a role-based `ChatPromptTemplate` using the same model and several questions.

## Requirements covered

1. **Load a Hugging Face model and generate responses.** `src/hf_models.py` creates a `HuggingFaceEndpoint` and a `ChatHuggingFace` wrapper.
2. **Integrate Hugging Face with LangChain.** Both prompt styles are runnable LangChain chains; no OpenAI model is used.
3. **Use system and human messages.** The chat chain is built with `ChatPromptTemplate.from_messages`.
4. **Compare prompt styles on multiple prompts.** The default comparison uses three questions and can save the results to Markdown.

## Setup (Windows PowerShell)

Use Python 3.10 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Create a Hugging Face access token with Inference Providers access and set it in `.env`:

```text
HUGGINGFACEHUB_API_TOKEN=your-token
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3
HF_PROVIDER=auto
HF_MAX_NEW_TOKENS=256
```

Keep `.env` private; it is excluded from Git. The model account/token must be permitted to use the selected inference provider. You can override the model with `--model`; for chat-template comparisons choose an instruction/chat model supported by Hugging Face Inference Providers.

## Run a plain prompt chain

```powershell
python main.py --mode normal --prompt "Explain photosynthesis in two sentences."
```

## Run a chat prompt chain

```powershell
python main.py --mode chat --prompt "Explain photosynthesis in two sentences."
```

## Compare templates on multiple prompts

Without `--prompt`, comparison mode runs three built-in questions with the same model:

```powershell
python main.py --mode compare --output outputs/prompt_comparison.md
```

Run a single custom question through both styles:

```powershell
python main.py --mode compare --prompt "What is the difference between training and inference?"
```

## How the prompt comparison works

- **Plain prompt:** A `PromptTemplate` formats one text prompt, then passes it directly to the Hugging Face text-generation endpoint.
- **Chat prompt:** A `ChatPromptTemplate` separates a system instruction from the human question, then passes those messages through `ChatHuggingFace`.
- **Fair comparison:** Both chains use the same Hugging Face model and the same question. The chat form can provide stronger role and instruction separation; the plain form is a simple text baseline. Output quality can vary by model, provider, and sampling.
- **Model choice:** The default is `mistralai/Mistral-7B-Instruct-v0.3`. `google/flan-t5-base` is also an example model for text-generation, but the chat comparison requires a model that supports chat/instruction formatting.

The CLI prints generated outputs. Use `--output` to keep a Markdown record of the actual responses. Tests use a deterministic fake chat model and do not call Hugging Face.

## Tests

```powershell
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

The offline tests verify both prompt chains, multi-question comparison, report output, and missing-token validation. Live generation requires a Hugging Face token and provider access.

## Project structure

```text
.
├── src/
│   ├── hf_models.py
│   └── prompts.py
├── tests/
│   └── test_prompts.py
├── .env.example
├── .gitignore
├── main.py
├── requirements-test.txt
└── requirements.txt
```
