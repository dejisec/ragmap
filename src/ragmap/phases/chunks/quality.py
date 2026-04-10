from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher

from ragmap.models import ChunkMetric, ChunkQuality, DuplicateGroup, Source


def compute_quality(
    metrics: list[ChunkMetric],
    sources: list[Source],
) -> ChunkQuality:
    total = len(metrics)
    if total == 0:
        return _empty_quality()

    empty_count = sum(1 for m in metrics if m.is_empty)
    oversized_count = sum(1 for m in metrics if m.is_oversized)
    boilerplate_count = sum(1 for m in metrics if m.is_low_density)
    opaque_blob_count = sum(1 for m in metrics if m.is_opaque_blob)
    hard_total = empty_count + oversized_count + boilerplate_count + opaque_blob_count
    hard_pct = round((hard_total / total) * 100, 1)

    short_count = sum(1 for m in metrics if m.is_short)
    mid_start_count = sum(1 for m in metrics if m.mid_sentence_start)
    mid_end_count = sum(1 for m in metrics if m.mid_sentence_end)
    orphan_count = sum(1 for m in metrics if m.is_orphaned_continuation)
    near_dup_count = _count_near_duplicates(metrics, sources)
    corpus_dup_count, dup_groups = _find_corpus_duplicates(metrics, sources)

    soft_total = (
        short_count
        + mid_start_count
        + mid_end_count
        + orphan_count
        + near_dup_count
        + corpus_dup_count
    )
    soft_pct = round((soft_total / total) * 100, 1)

    soft_samples = _collect_samples(metrics)

    return ChunkQuality(
        empty_count=empty_count,
        oversized_count=oversized_count,
        boilerplate_count=boilerplate_count,
        opaque_blob_count=opaque_blob_count,
        hard_total=hard_total,
        hard_pct=hard_pct,
        short_count=short_count,
        mid_sentence_start_count=mid_start_count,
        mid_sentence_end_count=mid_end_count,
        orphaned_continuation_count=orphan_count,
        near_duplicate_count=near_dup_count,
        corpus_duplicate_count=corpus_dup_count,
        soft_total=soft_total,
        soft_pct=soft_pct,
        soft_samples=soft_samples,
        duplicate_groups=dup_groups,
    )


def _empty_quality() -> ChunkQuality:
    return ChunkQuality(
        empty_count=0,
        oversized_count=0,
        boilerplate_count=0,
        opaque_blob_count=0,
        hard_total=0,
        hard_pct=0.0,
        short_count=0,
        mid_sentence_start_count=0,
        mid_sentence_end_count=0,
        orphaned_continuation_count=0,
        near_duplicate_count=0,
        corpus_duplicate_count=0,
        soft_total=0,
        soft_pct=0.0,
        soft_samples={},
        duplicate_groups=[],
    )


def _count_near_duplicates(metrics: list[ChunkMetric], sources: list[Source]) -> int:
    text_lookup: dict[tuple[str, str], str] = {}
    for s in sources:
        if s.chunk_id and s.text:
            text_lookup[(s.title, s.chunk_id)] = s.text

    if not text_lookup:
        return 0

    by_doc: dict[str, list[ChunkMetric]] = defaultdict(list)
    for m in metrics:
        by_doc[m.doc_title].append(m)

    count = 0
    for _title, doc_metrics in by_doc.items():
        sorted_metrics = sorted(
            doc_metrics,
            key=lambda m: m.trailing_number if m.trailing_number is not None else 0,
        )
        for i in range(len(sorted_metrics) - 1):
            m_a = sorted_metrics[i]
            m_b = sorted_metrics[i + 1]
            if m_a.text_hash == m_b.text_hash:
                continue
            text_a = text_lookup.get((m_a.doc_title, m_a.chunk_id))
            text_b = text_lookup.get((m_b.doc_title, m_b.chunk_id))
            if text_a and text_b:
                ratio = SequenceMatcher(None, text_a, text_b).ratio()
                if ratio > 0.9:
                    count += 1
    return count


def _find_corpus_duplicates(
    metrics: list[ChunkMetric],
    sources: list[Source],
) -> tuple[int, list[DuplicateGroup]]:
    hash_to_metrics: dict[str, list[ChunkMetric]] = defaultdict(list)
    for m in metrics:
        hash_to_metrics[m.text_hash].append(m)

    text_lookup: dict[tuple[str, str], str] = {}
    for s in sources:
        if s.chunk_id and s.text:
            text_lookup[(s.title, s.chunk_id)] = s.text

    groups: list[DuplicateGroup] = []
    total_dup_chunks = 0

    for text_hash, group_metrics in hash_to_metrics.items():
        doc_titles = {m.doc_title for m in group_metrics}
        if len(doc_titles) < 2:
            continue
        total_dup_chunks += len(group_metrics)

        first = group_metrics[0]
        preview = text_lookup.get((first.doc_title, first.chunk_id), "")[:80]
        groups.append(
            DuplicateGroup(
                text_preview=preview,
                count=len(group_metrics),
                doc_titles=sorted(doc_titles),
            )
        )

    groups.sort(key=lambda g: g.count, reverse=True)
    return total_dup_chunks, groups


def _collect_samples(metrics: list[ChunkMetric]) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = {}
    issue_checks = [
        ("short", lambda m: m.is_short),
        ("mid_sentence_start", lambda m: m.mid_sentence_start),
        ("mid_sentence_end", lambda m: m.mid_sentence_end),
        ("orphaned_continuation", lambda m: m.is_orphaned_continuation),
    ]
    for issue_name, check in issue_checks:
        ids = [m.chunk_id for m in metrics if check(m)][:5]
        if ids:
            samples[issue_name] = ids
    return samples
