from __future__ import annotations

from ragmap.models import Score, Source
from ragmap.presets.base import Preset


class HaystackPreset(Preset):
    def build_request_body(self, query: str) -> dict:
        return {"query": query}

    def extract_sources(self, response: dict) -> list[Source]:
        docs = response.get("documents", [])
        sources = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            meta = doc.get("meta", {})
            title = meta.get("name", "")
            if not title:
                continue
            sources.append(
                Source(
                    title=title,
                    chunk_id=meta.get("chunk_id"),
                    text=doc.get("content"),
                    metadata=meta,
                )
            )
        return sources

    def extract_scores(self, response: dict) -> list[Score]:
        docs = response.get("documents", [])
        scores = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            meta = doc.get("meta", {})
            title = meta.get("name", "")
            score_val = doc.get("score")
            if title and score_val is not None:
                try:
                    combined_score = float(score_val)
                except (TypeError, ValueError):
                    continue
                scores.append(Score(source_title=title, combined_score=combined_score))
        return scores

    def extract_answer(self, response: dict) -> str:
        answers = response.get("answers", [])
        if answers and isinstance(answers[0], dict):
            return str(answers[0].get("answer", ""))
        return ""

    def extract_retrieval_time(self, response: dict) -> float | None:
        val = response.get("retrieval_time_ms")
        return float(val) if val is not None else None
