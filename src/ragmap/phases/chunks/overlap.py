from __future__ import annotations

import statistics
from collections import defaultdict

from ragmap.models import ChunkMetric, DocOverlap, OverlapStats, Source


def compute_overlap_stats(
    sources: list[Source],
    metrics: list[ChunkMetric],
) -> OverlapStats:
    """Exhaustive per-doc overlap detection with aggregate stats."""
    by_doc: dict[str, list[Source]] = defaultdict(list)
    for s in sources:
        if s.chunk_id and s.text:
            by_doc[s.title].append(s)

    trailing_nums: dict[tuple[str, str], int] = {}
    for m in metrics:
        if m.trailing_number is not None:
            trailing_nums[(m.doc_title, m.chunk_id)] = m.trailing_number

    if not by_doc:
        return OverlapStats(has_overlap=None)

    per_doc: dict[str, DocOverlap] = {}
    all_overlaps: list[int] = []
    has_any_pairs = False

    for title, doc_sources in by_doc.items():
        if len(doc_sources) < 2:
            continue

        has_any_pairs = True
        sorted_sources = sorted(
            doc_sources,
            key=lambda s: trailing_nums.get((s.title, s.chunk_id), 0),
        )

        doc_char_counts = [len(s.text) for s in sorted_sources if s.text]
        doc_median_chars = statistics.median(doc_char_counts) if doc_char_counts else 1

        overlaps: list[int] = []
        pairs_checked = 0
        for i in range(len(sorted_sources) - 1):
            text_a = sorted_sources[i].text or ""
            text_b = sorted_sources[i + 1].text or ""
            pairs_checked += 1
            ov = _find_overlap(text_a, text_b)
            if ov > 10:
                overlaps.append(ov)

        all_overlaps.extend(overlaps)

        if overlaps:
            per_doc[title] = DocOverlap(
                pairs_checked=pairs_checked,
                overlapping_pairs=len(overlaps),
                min_overlap_chars=min(overlaps),
                max_overlap_chars=max(overlaps),
                median_overlap_chars=statistics.median(overlaps),
                overlap_pct=round(
                    (statistics.median(overlaps) / doc_median_chars) * 100, 1
                )
                if doc_median_chars
                else 0.0,
            )
        else:
            per_doc[title] = DocOverlap(
                pairs_checked=pairs_checked,
                overlapping_pairs=0,
                min_overlap_chars=0,
                max_overlap_chars=0,
                median_overlap_chars=0.0,
                overlap_pct=0.0,
            )

    if not has_any_pairs:
        return OverlapStats(has_overlap=None)

    has_overlap = len(all_overlaps) > 0

    agg_min = min(all_overlaps) if all_overlaps else None
    agg_max = max(all_overlaps) if all_overlaps else None
    agg_median = statistics.median(all_overlaps) if all_overlaps else None

    all_char_counts = [m.char_count for m in metrics if m.char_count > 0]
    corpus_median_chars = statistics.median(all_char_counts) if all_char_counts else 1
    agg_pct = (
        round((agg_median / corpus_median_chars) * 100, 1)
        if agg_median and corpus_median_chars
        else None
    )

    return OverlapStats(
        has_overlap=has_overlap,
        per_doc=per_doc,
        aggregate_min=agg_min,
        aggregate_max=agg_max,
        aggregate_median=agg_median,
        aggregate_pct_of_chunk_median=agg_pct,
    )


def _find_overlap(text_a: str, text_b: str) -> int:
    """Find the longest suffix of text_a that matches a prefix of text_b."""
    max_overlap = min(len(text_a), len(text_b))
    for length in range(max_overlap, 0, -1):
        if text_a[-length:] == text_b[:length]:
            return length
    return 0
