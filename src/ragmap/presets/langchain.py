from __future__ import annotations

from ragmap.models import Score, Source
from ragmap.presets.base import Preset


class LangChainPreset(Preset):
    def build_request_body(self, query: str) -> dict:
        return {"query": query}

    def extract_sources(self, response: dict) -> list[Source]:
        docs = response.get("source_documents", [])
        sources = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            metadata = doc.get("metadata", {})
            title = metadata.get("source", "")
            if not title:
                continue
            sources.append(
                Source(
                    title=title,
                    chunk_id=metadata.get("chunk_id"),
                    text=doc.get("page_content"),
                    metadata=metadata,
                )
            )
        return sources

    def extract_scores(self, response: dict) -> list[Score]:
        docs = response.get("source_documents", [])
        scores = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            metadata = doc.get("metadata", {})
            title = metadata.get("source", "")
            score_val = metadata.get("score")
            if title and score_val is not None:
                try:
                    combined_score = float(score_val)
                except (TypeError, ValueError):
                    continue
                scores.append(Score(source_title=title, combined_score=combined_score))
        return scores

    def extract_answer(self, response: dict) -> str:
        return str(response.get("result", ""))

    def extract_retrieval_time(self, response: dict) -> float | None:
        val = response.get("retrieval_time_ms")
        return float(val) if val is not None else None
