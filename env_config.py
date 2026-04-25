from __future__ import annotations

from pathlib import Path
import os


def load_local_env(env_path: str = ".env") -> None:
    """
    Minimal .env loader so local runs work without requiring python-dotenv.
    Existing environment variables win over file values.
    """
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def llm_status_label() -> str:
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"):
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
        model = os.getenv("OPENAI_MODEL", "unknown-model")
        return f"LLM mode: configured ({model} via {base})"
    if os.getenv("HUGGINGFACE_API_TOKEN"):
        model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-70B-Instruct")
        return f"LLM mode: configured ({model} via Hugging Face Inference)"
    return "LLM mode: fallback"
