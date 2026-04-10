from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    title: str
    chunk_id: str | None = None
    text: str | None = None
    metadata: dict = Field(default_factory=dict)


class Score(BaseModel):
    source_title: str
    vector_score: float | None = None
    bm25_score: float | None = None
    combined_score: float | None = None


class DetectionResult(BaseModel):
    has_rag: bool
    exposure_level: str
    retrieval_time_ms: float | None = None
    source_count: int = 0
    control_source_count: int = 0
    control_response: dict = Field(default_factory=dict)
    domain_response: dict = Field(default_factory=dict)


class Document(BaseModel):
    title: str
    hit_count: int = 1
    topics: list[str] = Field(default_factory=list)
    text_snippets: list[str] = Field(default_factory=list)


class EnumerationResult(BaseModel):
    documents: list[Document]
    unique_documents: int
    total_queries: int
    sources: list[Source] = Field(default_factory=list)


class ThresholdVariant(BaseModel):
    variant: str
    query: str
    source_count: int
    avg_score: float | None = None
    retrieved: bool


class ThresholdResult(BaseModel):
    variants: list[ThresholdVariant]
    threshold_boundary: str


class ChunkMetric(BaseModel):
    doc_title: str
    chunk_id: str
    char_count: int
    token_count: int
    text_hash: str

    is_empty: bool
    is_short: bool
    is_oversized: bool
    is_low_density: bool
    is_opaque_blob: bool
    is_orphaned_continuation: bool
    mid_sentence_start: bool
    mid_sentence_end: bool

    trailing_number: int | None
    id_prefix: str | None
    zero_padded: bool
    pad_width: int | None


class SizeStats(BaseModel):
    char_min: int
    char_max: int
    char_median: float
    char_p95: float
    char_stddev: float
    token_min: int
    token_max: int
    token_median: float
    token_p95: float
    token_p99: float
    token_stddev: float
    sample_count: int
    tokenizer: str
    max_chunk_tokens: int
    oversized_count: int


class DocOverlap(BaseModel):
    pairs_checked: int
    overlapping_pairs: int
    min_overlap_chars: int
    max_overlap_chars: int
    median_overlap_chars: float
    overlap_pct: float


class OverlapStats(BaseModel):
    has_overlap: bool | None
    per_doc: dict[str, DocOverlap] = Field(default_factory=dict)
    aggregate_min: int | None = None
    aggregate_max: int | None = None
    aggregate_median: float | None = None
    aggregate_pct_of_chunk_median: float | None = None


class ContiguityReport(BaseModel):
    total_ids: int
    missing_count: int
    gap_locations: list[int] = Field(default_factory=list)
    has_trailing_gap: bool
    interior_gap_count: int


class IDAnalysis(BaseModel):
    id_pattern: str
    scoping: str
    start_index_mode: int
    start_index_outliers: dict[str, int] = Field(default_factory=dict)
    format_consistent: bool
    format_issues: list[str] = Field(default_factory=list)
    contiguity: dict[str, ContiguityReport] = Field(default_factory=dict)
    intra_doc_duplicates: dict[str, list[str]] = Field(default_factory=dict)
    cross_doc_collisions: list[str] | None = None


class DuplicateGroup(BaseModel):
    text_preview: str
    count: int
    doc_titles: list[str] = Field(default_factory=list)


class ChunkQuality(BaseModel):
    empty_count: int
    oversized_count: int
    boilerplate_count: int
    opaque_blob_count: int
    hard_total: int
    hard_pct: float
    short_count: int
    mid_sentence_start_count: int
    mid_sentence_end_count: int
    orphaned_continuation_count: int
    near_duplicate_count: int
    corpus_duplicate_count: int
    soft_total: int
    soft_pct: float
    soft_samples: dict[str, list[str]] = Field(default_factory=dict)
    duplicate_groups: list[DuplicateGroup] = Field(default_factory=list)


class ChunkResult(BaseModel):
    size: SizeStats | None = None
    overlap: OverlapStats | None = None
    ids: IDAnalysis | None = None
    quality: ChunkQuality | None = None
    chunk_metrics: list[ChunkMetric] = Field(default_factory=list)


class ScanMeta(BaseModel):
    target: str
    preset: str
    queries_sent: int
    duration_seconds: float


class ScanResult(BaseModel):
    detection: DetectionResult | None = None
    enumeration: EnumerationResult | None = None
    thresholds: ThresholdResult | None = None
    chunks: ChunkResult | None = None
    meta: ScanMeta
