from __future__ import annotations

import tiktoken

EMBEDDING_MODEL_LIMITS: dict[str, int] = {
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-ada-002": 8191,
    "bge-small-en-v1.5": 512,
    "bge-base-en-v1.5": 512,
    "bge-large-en-v1.5": 512,
    "bge-m3": 8192,
    "e5-small-v2": 512,
    "e5-base-v2": 512,
    "e5-large-v2": 512,
    "nomic-embed-text-v1": 8192,
    "nomic-embed-text-v1.5": 8192,
    "jina-embeddings-v2-base-en": 8192,
    "voyage-3": 32000,
    "voyage-3-lite": 32000,
    "cohere-embed-english-v3.0": 512,
    "cohere-embed-multilingual-v3.0": 512,
}


def load_tokenizer(name: str = "o200k_base") -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(name)
    except ValueError as exc:
        raise ValueError(
            f"Unknown tiktoken encoding '{name}'. "
            f"Common encodings: o200k_base, cl100k_base"
        ) from exc


def infer_max_tokens(model_name: str | None, fallback: int = 8192) -> int:
    """Look up the token limit for a known embedding model.

    Returns *fallback* if the model is unknown or None.
    """
    if model_name is None:
        return fallback
    return EMBEDDING_MODEL_LIMITS.get(model_name, fallback)
