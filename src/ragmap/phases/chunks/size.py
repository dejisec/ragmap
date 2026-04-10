from __future__ import annotations

import math
import statistics

from ragmap.models import ChunkMetric, SizeStats


def compute_size_stats(
    metrics: list[ChunkMetric],
    tokenizer_name: str,
    max_chunk_tokens: int,
) -> SizeStats:
    char_counts = [m.char_count for m in metrics]
    token_counts = [m.token_count for m in metrics]

    return SizeStats(
        char_min=min(char_counts),
        char_max=max(char_counts),
        char_median=statistics.median(char_counts),
        char_p95=_percentile(char_counts, 95),
        char_stddev=statistics.pstdev(char_counts),
        token_min=min(token_counts),
        token_max=max(token_counts),
        token_median=statistics.median(token_counts),
        token_p95=_percentile(token_counts, 95),
        token_p99=_percentile(token_counts, 99),
        token_stddev=statistics.pstdev(token_counts),
        sample_count=len(metrics),
        tokenizer=tokenizer_name,
        max_chunk_tokens=max_chunk_tokens,
        oversized_count=sum(1 for m in metrics if m.is_oversized),
    )


def _percentile(data: list[int | float], pct: float) -> float:
    """Compute the given percentile using linear interpolation."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return float(sorted_data[0])
    k = (pct / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_data[f])
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
