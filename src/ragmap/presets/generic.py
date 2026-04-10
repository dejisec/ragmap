from __future__ import annotations

import json
from typing import Any

from ragmap.models import Score, Source
from ragmap.presets.base import Preset


def resolve_path(data: Any, path: str) -> Any:
    """Resolve a dot-notation path with [] array iteration.

    Examples:
        resolve_path(d, "answer")            -> d["answer"]
        resolve_path(d, "data.response")     -> d["data"]["response"]
        resolve_path(d, "sources[].title")   -> [s["title"] for s in d["sources"]]
    """
    if not path:
        return data
    if data is None:
        return None

    # Split on first dot, handling [] markers
    if "." in path:
        first, rest = path.split(".", 1)
    else:
        first, rest = path, ""

    # Handle array iteration marker
    if first.endswith("[]"):
        key = first[:-2]
        value = data.get(key) if isinstance(data, dict) else None
        if not isinstance(value, list):
            return None
        if not rest:
            return value
        return [resolve_path(item, rest) for item in value]

    # Simple key access
    if isinstance(data, dict):
        return resolve_path(data.get(first), rest)

    return None


_SCORE_FALLBACKS = ("combined_score", "relevance_score", "_score")


class GenericPreset(Preset):
    def __init__(
        self,
        query_field: str = "query",
        body_template: str | None = None,
        source_path: str = "sources[]",
        answer_path: str = "answer",
        title_path: str = "title",
        chunk_id_path: str = "chunk_id",
        text_path: str = "text",
        score_path: str = "score",
        retrieval_time_path: str = "retrieval_time_ms",
    ):
        self.query_field = query_field
        self.body_template = body_template
        self.source_path = source_path
        self.answer_path = answer_path
        self.title_path = title_path
        self.chunk_id_path = chunk_id_path
        self.text_path = text_path
        self.score_path = score_path
        self.retrieval_time_path = retrieval_time_path

        if self.body_template is not None:
            if "{query}" not in self.body_template:
                raise ValueError("body_template must contain '{query}' placeholder")
            try:
                json.loads(self.body_template.replace("{query}", "test"))
            except json.JSONDecodeError as e:
                raise ValueError(f"body_template is not valid JSON: {e}") from e

    def build_request_body(self, query: str) -> dict:
        if self.body_template:
            encoded = json.dumps(query)[
                1:-1
            ]  # JSON-safe string (no surrounding quotes)
            return json.loads(self.body_template.replace("{query}", encoded))
        return {self.query_field: query}

    def extract_sources(self, response: dict) -> list[Source]:
        raw_sources = resolve_path(response, self.source_path)
        if not isinstance(raw_sources, list):
            return []
        sources = []
        for item in raw_sources:
            if isinstance(item, str):
                if item:
                    sources.append(Source(title=item))
                continue
            if not isinstance(item, dict):
                continue
            title = resolve_path(item, self.title_path)
            if not title:
                continue
            sources.append(
                Source(
                    title=str(title),
                    chunk_id=resolve_path(item, self.chunk_id_path),
                    text=resolve_path(item, self.text_path),
                    metadata=item,
                )
            )
        return sources

    def extract_scores(self, response: dict) -> list[Score]:
        raw_sources = resolve_path(response, self.source_path)
        if not isinstance(raw_sources, list):
            return []
        scores = []
        for item in raw_sources:
            if not isinstance(item, dict):
                continue
            title = resolve_path(item, self.title_path)
            score_val = resolve_path(item, self.score_path)
            if score_val is None:
                for fallback in _SCORE_FALLBACKS:
                    if fallback != self.score_path:
                        score_val = item.get(fallback)
                        if score_val is not None:
                            break
            if title and score_val is not None:
                try:
                    combined_score = float(score_val)
                except (TypeError, ValueError):
                    continue
                scores.append(
                    Score(
                        source_title=str(title),
                        combined_score=combined_score,
                    )
                )
        return scores

    def extract_answer(self, response: dict) -> str:
        result = resolve_path(response, self.answer_path)
        return str(result) if result is not None else ""

    def extract_retrieval_time(self, response: dict) -> float | None:
        result = resolve_path(response, self.retrieval_time_path)
        if result is not None:
            try:
                return float(result)
            except (TypeError, ValueError):
                return None
        return None
