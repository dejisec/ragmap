from __future__ import annotations

import re
from collections import Counter, defaultdict

from ragmap.models import ChunkMetric, ContiguityReport, IDAnalysis

_HIERARCHICAL_RE = re.compile(r"^.+_\d+_.+_\d+$")
_SEQUENTIAL_RE = re.compile(r"^[a-zA-Z_]+_?\d+$")


def compute_id_analysis(metrics: list[ChunkMetric]) -> IDAnalysis:
    chunk_ids = [m.chunk_id for m in metrics]
    id_pattern = _detect_pattern(chunk_ids)

    by_doc: dict[str, list[ChunkMetric]] = defaultdict(list)
    for m in metrics:
        by_doc[m.doc_title].append(m)

    scoping = _detect_scoping(by_doc)
    start_index_mode, start_index_outliers = _detect_start_index(by_doc)
    format_consistent, format_issues = _check_format_consistency(metrics)
    contiguity = _compute_contiguity(by_doc)
    intra_doc_duplicates = _find_intra_doc_duplicates(by_doc)
    cross_doc_collisions = (
        _find_cross_doc_collisions(metrics) if scoping == "global" else None
    )

    return IDAnalysis(
        id_pattern=id_pattern,
        scoping=scoping,
        start_index_mode=start_index_mode,
        start_index_outliers=start_index_outliers,
        format_consistent=format_consistent,
        format_issues=format_issues,
        contiguity=contiguity,
        intra_doc_duplicates=intra_doc_duplicates,
        cross_doc_collisions=cross_doc_collisions,
    )


def _detect_pattern(chunk_ids: list[str]) -> str:
    hierarchical_count = 0
    sequential_count = 0
    for cid in chunk_ids:
        if _HIERARCHICAL_RE.match(cid):
            hierarchical_count += 1
        elif _SEQUENTIAL_RE.match(cid):
            sequential_count += 1
    if hierarchical_count > sequential_count:
        return "hierarchical (e.g., doc_N_chunk_N)"
    if sequential_count > 0:
        return "sequential (e.g., chunk_NNN)"
    return "custom"


def _detect_scoping(by_doc: dict[str, list[ChunkMetric]]) -> str:
    if len(by_doc) < 2:
        return "doc-scoped"

    prefixes_per_doc: dict[str, set[str]] = {}
    starts: dict[str, int | None] = {}
    for title, doc_metrics in by_doc.items():
        doc_prefixes = {m.id_prefix for m in doc_metrics if m.id_prefix is not None}
        prefixes_per_doc[title] = doc_prefixes
        nums = [m.trailing_number for m in doc_metrics if m.trailing_number is not None]
        starts[title] = min(nums) if nums else None

    all_prefix_sets = [ps for ps in prefixes_per_doc.values() if ps]
    if len(all_prefix_sets) >= 2:
        flattened = [p for ps in all_prefix_sets for p in ps]
        unique_prefixes = set(flattened)
        if len(unique_prefixes) >= len(all_prefix_sets):
            prefix_per_doc = {}
            for title, ps in prefixes_per_doc.items():
                if len(ps) == 1:
                    prefix_per_doc[title] = next(iter(ps))
            if len(prefix_per_doc) >= 2 and len(set(prefix_per_doc.values())) == len(
                prefix_per_doc
            ):
                return "global"

    start_values = [v for v in starts.values() if v is not None]
    if start_values:
        most_common = Counter(start_values).most_common(1)[0][0]
        shared_count = sum(1 for v in start_values if v == most_common)
        if shared_count / len(start_values) > 0.5:
            return "doc-scoped"

    return "doc-scoped"


def _detect_start_index(
    by_doc: dict[str, list[ChunkMetric]],
) -> tuple[int, dict[str, int]]:
    starts: dict[str, int] = {}
    for title, doc_metrics in by_doc.items():
        nums = [m.trailing_number for m in doc_metrics if m.trailing_number is not None]
        if nums:
            starts[title] = min(nums)

    if not starts:
        return 0, {}

    start_values = list(starts.values())
    mode = Counter(start_values).most_common(1)[0][0]
    outliers = {title: idx for title, idx in starts.items() if idx != mode}
    return mode, outliers


def _check_format_consistency(metrics: list[ChunkMetric]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    prefixes = {m.id_prefix for m in metrics if m.id_prefix is not None}
    if len(prefixes) > 1:
        issues.append(f"Mixed prefixes: {', '.join(sorted(prefixes))}")

    pad_widths = {m.pad_width for m in metrics if m.pad_width is not None}
    has_padded = any(m.zero_padded for m in metrics)
    has_unpadded = any(
        not m.zero_padded and m.trailing_number is not None for m in metrics
    )
    if has_padded and has_unpadded:
        issues.append("Mixed padding: some IDs zero-padded, some not")
    elif len(pad_widths) > 1:
        issues.append(f"Inconsistent pad widths: {sorted(pad_widths)}")

    return len(issues) == 0, issues


def _compute_contiguity(
    by_doc: dict[str, list[ChunkMetric]],
) -> dict[str, ContiguityReport]:
    reports: dict[str, ContiguityReport] = {}
    for title, doc_metrics in by_doc.items():
        nums = sorted(
            {m.trailing_number for m in doc_metrics if m.trailing_number is not None}
        )
        if not nums:
            continue
        expected = set(range(min(nums), max(nums) + 1))
        actual = set(nums)
        missing = sorted(expected - actual)

        has_trailing = False
        interior_count = len(missing)

        reports[title] = ContiguityReport(
            total_ids=len(nums),
            missing_count=len(missing),
            gap_locations=missing,
            has_trailing_gap=has_trailing,
            interior_gap_count=interior_count,
        )
    return reports


def _find_intra_doc_duplicates(
    by_doc: dict[str, list[ChunkMetric]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for title, doc_metrics in by_doc.items():
        counts = Counter(m.chunk_id for m in doc_metrics)
        dupes = [cid for cid, count in counts.items() if count > 1]
        if dupes:
            result[title] = dupes
    return result


def _find_cross_doc_collisions(metrics: list[ChunkMetric]) -> list[str]:
    id_to_docs: dict[str, set[str]] = defaultdict(set)
    for m in metrics:
        id_to_docs[m.chunk_id].add(m.doc_title)
    return sorted(cid for cid, docs in id_to_docs.items() if len(docs) > 1)
