from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import tiktoken

from ragmap.models import ChunkMetric, Source

_HEADING_RE = re.compile(r"^#{1,6}\s")
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z\s]{2,}$")
_LIST_ITEM_RE = re.compile(r"^[-*\d]+[.)]\s")
_ORPHAN_RE = re.compile(
    r"^(\.{3}|however,?\s|but\s|and\s|therefore\s|furthermore\s|additionally\s|moreover\s|[\]\)\}])",
    re.IGNORECASE,
)
_TRAILING_NUM_RE = re.compile(r"(\d+)$")
_TERMINAL_PUNCT = set(".!?:;)]\"'>}")


def compute_metrics(
    sources: list[Source],
    tokenizer: tiktoken.Encoding,
    max_chunk_tokens: int,
) -> list[ChunkMetric]:
    metrics: list[ChunkMetric] = []
    for s in sources:
        if s.chunk_id is None:
            continue
        text = s.text or ""
        char_count = len(text)
        token_count = len(tokenizer.encode(text)) if text else 0
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        trailing_number, id_prefix, zero_padded, pad_width = _parse_chunk_id(s.chunk_id)

        stripped = text.strip()
        is_empty = stripped == ""
        is_short = (
            token_count < 10 and not is_empty and not _is_structural_boundary(stripped)
        )
        is_oversized = token_count > max_chunk_tokens
        is_low_density = _check_low_density(text)
        is_opaque_blob = _check_opaque_blob(text)
        is_orphaned_continuation = (
            bool(_ORPHAN_RE.match(stripped)) if stripped else False
        )
        mid_start = _check_mid_sentence_start(stripped)
        mid_end = _check_mid_sentence_end(stripped, char_count)

        metrics.append(
            ChunkMetric(
                doc_title=s.title,
                chunk_id=s.chunk_id,
                char_count=char_count,
                token_count=token_count,
                text_hash=text_hash,
                is_empty=is_empty,
                is_short=is_short,
                is_oversized=is_oversized,
                is_low_density=is_low_density,
                is_opaque_blob=is_opaque_blob,
                is_orphaned_continuation=is_orphaned_continuation,
                mid_sentence_start=mid_start,
                mid_sentence_end=mid_end,
                trailing_number=trailing_number,
                id_prefix=id_prefix,
                zero_padded=zero_padded,
                pad_width=pad_width,
            )
        )
    return metrics


def _parse_chunk_id(chunk_id: str) -> tuple[int | None, str | None, bool, int | None]:
    match = _TRAILING_NUM_RE.search(chunk_id)
    if not match:
        return None, None, False, None
    num_str = match.group(1)
    num = int(num_str)
    prefix = chunk_id[: match.start()]
    zero_padded = len(num_str) > 1 and num_str[0] == "0"
    pad_width = len(num_str) if zero_padded else None
    return num, prefix, zero_padded, pad_width


def _is_structural_boundary(text: str) -> bool:
    return bool(
        _HEADING_RE.match(text) or _ALLCAPS_RE.match(text) or _LIST_ITEM_RE.match(text)
    )


def _check_low_density(text: str) -> bool:
    if len(text) < 20:
        return False
    alnum = sum(1 for c in text if c.isalnum())
    return (alnum / len(text)) < 0.5


def _check_opaque_blob(text: str) -> bool:
    if len(text) < 50:
        return False
    max_run = max((len(seg) for seg in text.split() if seg), default=0)
    if max_run <= 100:
        return False
    data = text.encode("utf-8")
    counts = Counter(data)
    length = len(data)
    entropy = -sum((c / length) * math.log2(c / length) for c in counts.values())
    return entropy > 4.5


def _check_mid_sentence_start(stripped: str) -> bool:
    if not stripped:
        return False
    first = stripped[0]
    if not first.isalpha():
        return False
    return first.islower()


def _check_mid_sentence_end(stripped: str, char_count: int) -> bool:
    if char_count <= 50 or not stripped:
        return False
    last = stripped[-1]
    return last not in _TERMINAL_PUNCT
