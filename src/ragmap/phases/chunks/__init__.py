from __future__ import annotations

from ragmap.models import ChunkResult, Source
from ragmap.phases.chunks.ids import compute_id_analysis
from ragmap.phases.chunks.metrics import compute_metrics
from ragmap.phases.chunks.overlap import compute_overlap_stats
from ragmap.phases.chunks.quality import compute_quality
from ragmap.phases.chunks.size import compute_size_stats
from ragmap.phases.chunks.tokenizer import load_tokenizer


def run_chunks(
    sources: list[Source],
    tokenizer_name: str = "o200k_base",
    max_chunk_tokens: int = 8192,
) -> ChunkResult:
    """Run two-pass chunk analysis pipeline.

    Pass 1: compute per-chunk metrics (tokens, hashes, quality flags).
    Pass 2: aggregate into corpus-wide stats.
    """
    tokenizer = load_tokenizer(tokenizer_name)
    metrics = compute_metrics(sources, tokenizer, max_chunk_tokens)

    if not metrics:
        return ChunkResult()

    size = compute_size_stats(metrics, tokenizer_name, max_chunk_tokens)
    overlap = compute_overlap_stats(sources, metrics)
    ids = compute_id_analysis(metrics)
    quality = compute_quality(metrics, sources)

    return ChunkResult(
        size=size,
        overlap=overlap,
        ids=ids,
        quality=quality,
        chunk_metrics=metrics,
    )
