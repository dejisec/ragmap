from __future__ import annotations

from ragmap.models import Score, Source
from ragmap.presets.base import Preset


class LlamaIndexPreset(Preset):
    def build_request_body(self, query: str) -> dict:
        return {"query": query}

    def extract_sources(self, response: dict) -> list[Source]:
        nodes = response.get("source_nodes", [])
        sources = []
        for entry in nodes:
            if not isinstance(entry, dict):
                continue
            node = entry.get("node", {})
            metadata = node.get("metadata", {})
            title = metadata.get("file_name", "")
            if not title:
                continue
            sources.append(
                Source(
                    title=title,
                    chunk_id=metadata.get("chunk_id"),
                    text=node.get("text"),
                    metadata=metadata,
                )
            )
        return sources

    def extract_scores(self, response: dict) -> list[Score]:
        nodes = response.get("source_nodes", [])
        scores = []
        for entry in nodes:
            if not isinstance(entry, dict):
                continue
            node = entry.get("node", {})
            metadata = node.get("metadata", {})
            title = metadata.get("file_name", "")
            score_val = entry.get("score")
            if title and score_val is not None:
                try:
                    combined_score = float(score_val)
                except (TypeError, ValueError):
                    continue
                scores.append(Score(source_title=title, combined_score=combined_score))
        return scores

    def extract_answer(self, response: dict) -> str:
        return str(response.get("response", ""))

    def extract_retrieval_time(self, response: dict) -> float | None:
        val = response.get("retrieval_time_ms")
        return float(val) if val is not None else None
