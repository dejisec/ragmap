from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from ragmap.models import ScanResult


def render_scan_result(result: ScanResult, console: Console | None = None) -> None:
    console = console or Console()

    if result.detection:
        _render_detection(result, console)
    if result.enumeration:
        _render_enumeration(result, console)
    if result.thresholds:
        _render_thresholds(result, console)
    if result.chunks:
        _render_chunks(result, console)

    _render_summary(result, console)


def _render_detection(result: ScanResult, console: Console) -> None:
    det = result.detection
    if not det:
        return
    rag_status = "Yes" if det.has_rag else "No"
    lines = [
        f"RAG Detected:    {rag_status}",
        f"Exposure Level:  {det.exposure_level}",
    ]
    if det.retrieval_time_ms is not None:
        lines.append(f"Retrieval Time:  {det.retrieval_time_ms:.0f}ms")
    total_sources = det.source_count + det.control_source_count
    lines.append(f"Sources Found:   {total_sources}")
    console.print(Panel("\n".join(lines), title="RAG Detection"))


def _render_enumeration(result: ScanResult, console: Console) -> None:
    enum = result.enumeration
    if not enum:
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Document")
    table.add_column("Hits", justify="right")
    table.add_column("Topics")

    for doc in enum.documents:
        table.add_row(doc.title, str(doc.hit_count), ", ".join(doc.topics))

    console.print(
        Panel(
            table,
            title=f"Document Inventory ({enum.unique_documents} unique documents)",
        )
    )


def _render_thresholds(result: ScanResult, console: Console) -> None:
    thresh = result.thresholds
    if not thresh:
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Query Variant")
    table.add_column("Sources", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Status")

    for v in thresh.variants:
        score_str = f"{v.avg_score:.3f}" if v.avg_score is not None else "-"
        if v.variant == "off-topic":
            status = "No retrieval"
        elif v.retrieved:
            status = "Retrieved"
        else:
            status = "Blocked"
        table.add_row(
            v.variant.replace("_", " "), str(v.source_count), score_str, status
        )

    console.print(Panel(table, title="Retrieval Thresholds"))
    console.print(f"  Threshold: {thresh.threshold_boundary}")
    console.print()


def _render_chunks(result: ScanResult, console: Console) -> None:
    ch = result.chunks
    if not ch:
        return
    lines: list[str] = []
    if ch.size:
        lines.extend(_fmt_size(ch.size))
    if ch.overlap:
        lines.extend(_fmt_overlap(ch.overlap))
    if ch.ids:
        lines.extend(_fmt_ids(ch.ids))
    if ch.quality:
        lines.extend(_fmt_quality(ch.quality))
    if lines:
        console.print(Panel("\n".join(lines), title="Chunk Analysis"))


def _fmt_size(s) -> list[str]:
    lines = [
        "\u2500\u2500 Size Distribution \u2500\u2500",
        f"Tokens:  min {s.token_min} | median {s.token_median:.0f} | p95 {s.token_p95:.0f} | p99 {s.token_p99:.0f} | max {s.token_max}",
        f"Chars:   min {s.char_min} | median {s.char_median:.0f} | p95 {s.char_p95:.0f} | max {s.char_max}",
        f"Samples: {s.sample_count} chunks",
    ]
    if s.oversized_count > 0:
        lines.append(
            f"\u26a0 Oversized: {s.oversized_count} chunks exceed {s.max_chunk_tokens} token limit"
        )
    else:
        lines.append(f"Oversized: 0 chunks exceed {s.max_chunk_tokens} token limit")
    lines.append(f"\u2139 Token counts via {s.tokenizer}; \u00b115% on CJK/code")
    lines.append("")
    return lines


def _fmt_overlap(o) -> list[str]:
    lines = ["\u2500\u2500 Overlap \u2500\u2500"]
    for doc_title, doc_ov in o.per_doc.items():
        if doc_ov.overlapping_pairs > 0:
            lines.append(
                f"{doc_title}   {doc_ov.overlapping_pairs}/{doc_ov.pairs_checked} pairs "
                f"| ~{doc_ov.median_overlap_chars:.0f} chars median ({doc_ov.overlap_pct:.0f}%)"
            )
        else:
            lines.append(
                f"{doc_title}   {doc_ov.overlapping_pairs}/{doc_ov.pairs_checked} pairs | no overlap"
            )
    if o.aggregate_median is not None:
        lines.append(
            f"Aggregate: ~{o.aggregate_median:.0f} chars median overlap "
            f"({o.aggregate_pct_of_chunk_median:.0f}% of median chunk)"
        )
    elif o.has_overlap is False:
        lines.append("Aggregate: no overlap detected")
    lines.append("")
    return lines


def _fmt_ids(ids) -> list[str]:
    lines = [
        "\u2500\u2500 Chunk IDs \u2500\u2500",
        f"Pattern:     {ids.id_pattern}",
        f"Scoping:     {ids.scoping}",
    ]
    if ids.start_index_outliers:
        outlier_str = ", ".join(
            f"{doc}={idx}" for doc, idx in ids.start_index_outliers.items()
        )
        lines.append(f"Start Index: {ids.start_index_mode} (outliers: {outlier_str})")
    else:
        lines.append(f"Start Index: {ids.start_index_mode} (consistent)")

    if ids.format_consistent:
        lines.append("Format:      consistent")
    else:
        for issue in ids.format_issues:
            lines.append(f"\u26a0 {issue}")

    any_gaps = any(r.missing_count > 0 for r in ids.contiguity.values())
    if not any_gaps and ids.contiguity:
        lines.append("Contiguity:  all documents contiguous, no gaps")
    else:
        lines.append("Contiguity:")
        for doc_title, report in ids.contiguity.items():
            if report.missing_count > 0:
                gap_preview = ", ".join(str(g) for g in report.gap_locations[:10])
                if len(report.gap_locations) > 10:
                    gap_preview += ", ..."
                label = "trailing" if report.has_trailing_gap else "interior"
                lines.append(
                    f"  {doc_title}  {report.total_ids} IDs, {report.missing_count} {label} gaps [{gap_preview}]"
                )
            else:
                lines.append(f"  {doc_title}  {report.total_ids} IDs, 0 gaps")

    for doc_title, dupes in ids.intra_doc_duplicates.items():
        lines.append(f"\u26a0 {len(dupes)} intra-doc duplicate IDs in {doc_title}")

    if ids.cross_doc_collisions:
        lines.append(f"\u26a0 {len(ids.cross_doc_collisions)} cross-doc ID collisions")

    lines.append("")
    return lines


def _fmt_quality(q) -> list[str]:
    lines = ["\u2500\u2500 Quality \u2500\u2500"]

    if q.hard_total > 0:
        parts = []
        if q.empty_count:
            parts.append(f"{q.empty_count} empty")
        if q.oversized_count:
            parts.append(f"{q.oversized_count} oversized")
        if q.boilerplate_count:
            parts.append(f"{q.boilerplate_count} boilerplate")
        if q.opaque_blob_count:
            parts.append(f"{q.opaque_blob_count} opaque blob")
        lines.append(
            f"Hard degenerate: {q.hard_total} ({q.hard_pct}%) \u2014 {', '.join(parts)}"
        )
    else:
        lines.append("Hard degenerate: 0")

    if q.soft_total > 0:
        parts = []
        if q.mid_sentence_start_count or q.mid_sentence_end_count:
            mid_total = q.mid_sentence_start_count + q.mid_sentence_end_count
            parts.append(f"{mid_total} mid-sentence split")
        if q.short_count:
            parts.append(f"{q.short_count} short")
        if q.near_duplicate_count:
            parts.append(f"{q.near_duplicate_count} near-duplicate")
        if q.orphaned_continuation_count:
            parts.append(f"{q.orphaned_continuation_count} orphaned")
        if q.corpus_duplicate_count:
            parts.append(f"{q.corpus_duplicate_count} corpus duplicate")
        lines.append(
            f"Soft suspicious: {q.soft_total} ({q.soft_pct}%) \u2014 {', '.join(parts)}"
        )

        all_samples = []
        for _issue, ids in q.soft_samples.items():
            all_samples.extend(ids)
        if all_samples:
            lines.append(f"  samples: {', '.join(all_samples[:5])}")
    else:
        lines.append("Soft suspicious: 0")

    if q.duplicate_groups:
        lines.append(f"Corpus duplicates: {len(q.duplicate_groups)} groups")
        for group in q.duplicate_groups[:3]:
            preview = (
                group.text_preview[:50] + "..."
                if len(group.text_preview) > 50
                else group.text_preview
            )
            lines.append(
                f'  "{preview}" \u00d7 {group.count} across {len(group.doc_titles)} docs'
            )

    return lines


def _render_summary(result: ScanResult, console: Console) -> None:
    meta = result.meta
    parts = [
        f"Queries sent: {meta.queries_sent}",
        f"Duration: {meta.duration_seconds * 1000:.0f}ms"
        if meta.duration_seconds < 1
        else f"Duration: {meta.duration_seconds:.1f}s",
    ]
    if result.enumeration:
        parts.append(f"Unique docs: {result.enumeration.unique_documents}")
    console.print(Rule("Summary"))
    console.print("  " + " | ".join(parts))
    console.print()
