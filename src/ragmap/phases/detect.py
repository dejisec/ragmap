from __future__ import annotations

from ragmap.config import Target
from ragmap.models import DetectionResult, Source
from ragmap.presets.base import Preset
from ragmap.session import Sender, Session


async def run_detect(
    session: Sender,
    target: Target,
    preset: Preset,
    control_query: str = "What is 2+2?",
    domain_query: str = "What are the company policies?",
) -> DetectionResult:
    control_body = preset.build_request_body(control_query)
    control_resp = await session.send(target.method, target.url, json=control_body)
    control_resp.raise_for_status()
    control_data = Session.parse_json(control_resp)

    domain_body = preset.build_request_body(domain_query)
    domain_resp = await session.send(target.method, target.url, json=domain_body)
    domain_resp.raise_for_status()
    domain_data = Session.parse_json(domain_resp)

    control_sources = preset.extract_sources(control_data)
    domain_sources = preset.extract_sources(domain_data)
    domain_scores = preset.extract_scores(domain_data)
    retrieval_time = preset.extract_retrieval_time(domain_data)

    has_rag = len(domain_sources) > 0 or len(control_sources) > 0
    all_sources = domain_sources + control_sources
    exposure_level = classify_exposure(all_sources, domain_scores)

    return DetectionResult(
        has_rag=has_rag,
        exposure_level=exposure_level,
        retrieval_time_ms=retrieval_time,
        source_count=len(domain_sources),
        control_source_count=len(control_sources),
        control_response=control_data,
        domain_response=domain_data,
    )


def classify_exposure(sources: list[Source], scores: list) -> str:
    if not sources:
        return "none"
    has_chunk_ids = any(s.chunk_id for s in sources)
    has_text = any(s.text for s in sources)
    has_scores = len(scores) > 0

    if has_chunk_ids or has_scores:
        return "detailed"
    if has_text:
        return "moderate"
    return "minimal"
