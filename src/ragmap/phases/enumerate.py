from __future__ import annotations

from collections.abc import Callable

from ragmap.config import Target
from ragmap.models import Document, EnumerationResult, Source
from ragmap.presets.base import Preset
from ragmap.session import Sender, Session

DEFAULT_TOPICS = [
    "HR policies",
    "technical documentation",
    "system architecture",
    "security procedures",
    "infrastructure",
    "API documentation",
    "employee onboarding",
    "compliance requirements",
    "financial information",
    "internal tools",
]


def _default_query(topic: str) -> str:
    return f"What does the documentation say about {topic}?"


async def run_enumerate(
    session: Sender,
    target: Target,
    preset: Preset,
    topics: list[str] | None = None,
    exhaustive: bool = False,
    max_queries: int = 50,
    query_fn: Callable[[str], str | list[str]] | None = None,
) -> EnumerationResult:
    topics = topics or DEFAULT_TOPICS
    build_query = query_fn or _default_query
    documents: dict[str, Document] = {}
    total_queries = 0
    consecutive_empty = 0
    collected_sources: list[Source] = []

    for topic in topics:
        if total_queries >= max_queries:
            break
        if not exhaustive and consecutive_empty >= 3:
            break

        raw = build_query(topic)
        queries = raw if isinstance(raw, list) else [raw]

        topic_found_new = False

        for query in queries:
            if total_queries >= max_queries:
                break

            body = preset.build_request_body(query)
            resp = await session.send(target.method, target.url, json=body)
            resp.raise_for_status()
            data = Session.parse_json(resp)
            total_queries += 1

            sources = preset.extract_sources(data)
            collected_sources.extend(sources)

            for source in sources:
                if source.title in documents:
                    documents[source.title].hit_count += 1
                else:
                    documents[source.title] = Document(
                        title=source.title,
                        hit_count=1,
                        topics=[],
                        text_snippets=[],
                    )
                    topic_found_new = True

                doc = documents[source.title]
                if topic not in doc.topics:
                    doc.topics.append(topic)
                if source.text and source.text not in doc.text_snippets:
                    doc.text_snippets.append(source.text)

        if topic_found_new:
            consecutive_empty = 0
        else:
            consecutive_empty += 1

    doc_list = sorted(documents.values(), key=lambda d: d.hit_count, reverse=True)

    return EnumerationResult(
        documents=doc_list,
        unique_documents=len(doc_list),
        total_queries=total_queries,
        sources=collected_sources,
    )
