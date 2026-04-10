from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import httpx
import typer

from ragmap.config import Target, parse_cookie, parse_header
from ragmap.models import Document, ScanMeta, ScanResult, Source
from ragmap.output import render_scan_result
from ragmap.phases.chunks import run_chunks
from ragmap.phases.chunks.tokenizer import infer_max_tokens
from ragmap.phases.detect import classify_exposure, run_detect
from ragmap.phases.enumerate import run_enumerate
from ragmap.phases.threshold import run_threshold
from ragmap.presets import get_preset
from ragmap.session import Session
from ragmap.stealth.keywords import KeywordScanner
from ragmap.stealth.queries import StealthQueries
from ragmap.stealth.rotator import SessionRotator

app = typer.Typer(name="ragmap", help="RAG Pipeline Reconnaissance Tool")

# Reusable option types
HeaderOpt = Annotated[
    Optional[list[str]],
    typer.Option("-H", "--header", help="HTTP header 'Key: Value' (repeatable)"),
]
CookieOpt = Annotated[
    Optional[list[str]],
    typer.Option("-b", "--cookie", help="Cookie 'name=value' (repeatable)"),
]
ProxyOpt = Annotated[
    Optional[str],
    typer.Option("--proxy", help="Proxy URL (e.g., http://127.0.0.1:8080)"),
]
PresetOpt = Annotated[
    str,
    typer.Option("--preset", help="Preset: generic, langchain, llamaindex, haystack"),
]
DelayOpt = Annotated[float, typer.Option("--delay", help="Seconds between requests")]
JitterOpt = Annotated[
    float, typer.Option("--jitter", help="Random +/- jitter added to delay")
]
InsecureOpt = Annotated[bool, typer.Option("--insecure", help="Skip TLS verification")]
JsonOpt = Annotated[bool, typer.Option("--json", help="JSON output")]
OutputOpt = Annotated[
    Optional[str], typer.Option("-o", "--output", help="Write output to file")
]
VerboseOpt = Annotated[
    bool, typer.Option("-v", "--verbose", help="Show request/response details")
]
SourcePathOpt = Annotated[
    Optional[str],
    typer.Option("--source-path", help="Generic preset: dot-notation path to sources"),
]
AnswerPathOpt = Annotated[
    Optional[str],
    typer.Option("--answer-path", help="Generic preset: dot-notation path to answer"),
]
ScorePathOpt = Annotated[
    Optional[str],
    typer.Option("--score-path", help="Generic preset: dot-notation path to score"),
]
BodyTemplateOpt = Annotated[
    Optional[str],
    typer.Option(
        "--body-template", help="Generic preset: JSON body template with {query}"
    ),
]
QueryFieldOpt = Annotated[
    Optional[str],
    typer.Option(
        "--query-field",
        help="Generic preset: JSON field name for query (default: 'query')",
    ),
]
TitlePathOpt = Annotated[
    Optional[str],
    typer.Option("--title-path", help="Generic preset: field name for document title"),
]
ChunkIdPathOpt = Annotated[
    Optional[str],
    typer.Option("--chunk-id-path", help="Generic preset: field name for chunk ID"),
]
TextPathOpt = Annotated[
    Optional[str],
    typer.Option("--text-path", help="Generic preset: field name for chunk text"),
]
RetrievalTimePathOpt = Annotated[
    Optional[str],
    typer.Option(
        "--retrieval-time-path", help="Generic preset: path to retrieval time"
    ),
]
StealthOpt = Annotated[
    bool, typer.Option("--stealth", help="Activate stealth evasion mode")
]
EvasionRulesOpt = Annotated[
    Optional[str], typer.Option("--evasion-rules", help="Custom evasion rules YAML")
]
RotateEveryOpt = Annotated[
    int, typer.Option("--rotate-every", help="Rotate session ID every N requests")
]
SessionFieldOpt = Annotated[
    Optional[str],
    typer.Option("--session-field", help="Request body field for session ID"),
]
TimeoutOpt = Annotated[
    float, typer.Option("--timeout", help="HTTP request timeout in seconds")
]
TokenizerOpt = Annotated[
    str, typer.Option("--tokenizer", help="Tiktoken encoding (default: o200k_base)")
]
MaxChunkTokensOpt = Annotated[
    Optional[int],
    typer.Option(
        "--max-chunk-tokens", help="Embedding model token limit for oversized detection"
    ),
]
EmbeddingModelOpt = Annotated[
    Optional[str],
    typer.Option(
        "--embedding-model", help="Embedding model name (auto-infers token limit)"
    ),
]


def _build_target(url, header, cookie, proxy, preset, insecure):
    headers = dict(parse_header(h) for h in (header or []))
    cookies = dict(parse_cookie(c) for c in (cookie or []))
    return Target(
        url=url,
        headers=headers,
        cookies=cookies,
        proxy=proxy,
        preset_name=preset,
        insecure=insecure,
    )


def _build_preset(preset_name, **kwargs):
    try:
        if preset_name == "generic":
            filtered = {k: v for k, v in kwargs.items() if v is not None}
            return get_preset(preset_name, **filtered)
        return get_preset(preset_name)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def _build_stealth(stealth, evasion_rules, rotate_every, session_field):
    if not stealth:
        return None, None, None
    if rotate_every < 1:
        typer.echo("Error: --rotate-every must be >= 1", err=True)
        raise typer.Exit(1)
    overlay = Path(evasion_rules) if evasion_rules else None
    try:
        scanner = KeywordScanner(overlay_path=overlay)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    queries = StealthQueries()
    return (
        scanner,
        queries,
        {"rotate_every": rotate_every, "session_field": session_field or "session_id"},
    )


def _resolve_chunk_token_limit(
    max_chunk_tokens: int | None,
    embedding_model: str | None,
) -> int:
    """Resolve the chunk token limit from explicit flag, model lookup, or default."""
    if max_chunk_tokens is not None:
        return max_chunk_tokens
    if embedding_model is not None:
        limit = infer_max_tokens(embedding_model)
        if limit == 8192 and embedding_model not in (
            "text-embedding-3-small",
            "text-embedding-3-large",
        ):
            typer.echo(
                f"Warning: Unknown embedding model '{embedding_model}', "
                f"using default limit 8192. Override with --max-chunk-tokens.",
                err=True,
            )
        return limit
    return 8192


def _handle_run(coro_fn):
    try:
        return asyncio.run(coro_fn())
    except httpx.ConnectError as e:
        typer.echo(f"Error: Connection failed — {e}", err=True)
        raise typer.Exit(1)
    except httpx.TimeoutException as e:
        typer.echo(f"Error: Request timed out — {e}", err=True)
        raise typer.Exit(1)
    except httpx.HTTPStatusError as e:
        typer.echo(f"Error: HTTP {e.response.status_code} from target", err=True)
        raise typer.Exit(1)
    except httpx.RequestError as e:
        typer.echo(f"Error: Request failed — {e}", err=True)
        raise typer.Exit(1)
    except RuntimeError as e:
        if "non-JSON" in str(e):
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        raise
    except (OverflowError, ValueError) as e:
        typer.echo(f"Error: Invalid configuration — {e}", err=True)
        raise typer.Exit(1)
    except BaseException as e:
        # ExceptionGroup from asyncio.TaskGroup wraps the real exception on Python 3.11+
        if isinstance(e, ExceptionGroup):
            config_errs = [
                ex for ex in e.exceptions if isinstance(ex, (OverflowError, ValueError))
            ]
            if config_errs:
                typer.echo(f"Error: Invalid configuration — {config_errs[0]}", err=True)
                raise typer.Exit(1)
        raise


def _validate_delay(delay: float) -> None:
    if delay < 0:
        typer.echo("Error: --delay must be >= 0", err=True)
        raise typer.Exit(1)


@app.command()
def detect(
    url: Annotated[str, typer.Argument(help="Target RAG API URL")],
    header: HeaderOpt = None,
    cookie: CookieOpt = None,
    proxy: ProxyOpt = None,
    preset: PresetOpt = "generic",
    delay: DelayOpt = 1.0,
    jitter: JitterOpt = 0.0,
    insecure: InsecureOpt = False,
    json_output: JsonOpt = False,
    output: OutputOpt = None,
    verbose: VerboseOpt = False,
    control_query: Annotated[
        str, typer.Option("--control-query", help="Control query (no RAG expected)")
    ] = "What is 2+2?",
    domain_query: Annotated[
        str, typer.Option("--domain-query", help="Domain query (RAG expected)")
    ] = "What are the company policies?",
    source_path: SourcePathOpt = None,
    answer_path: AnswerPathOpt = None,
    score_path: ScorePathOpt = None,
    body_template: BodyTemplateOpt = None,
    query_field: QueryFieldOpt = None,
    title_path: TitlePathOpt = None,
    chunk_id_path: ChunkIdPathOpt = None,
    text_path: TextPathOpt = None,
    retrieval_time_path: RetrievalTimePathOpt = None,
    timeout: TimeoutOpt = 30.0,
    stealth: StealthOpt = False,
    evasion_rules: EvasionRulesOpt = None,
    rotate_every: RotateEveryOpt = 3,
    session_field: SessionFieldOpt = None,
):
    """Phase 1: Detect RAG usage and assess metadata exposure."""
    _validate_delay(delay)
    target = _build_target(url, header, cookie, proxy, preset, insecure)
    p = _build_preset(
        preset,
        source_path=source_path,
        answer_path=answer_path,
        score_path=score_path,
        body_template=body_template,
        query_field=query_field,
        title_path=title_path,
        chunk_id_path=chunk_id_path,
        text_path=text_path,
        retrieval_time_path=retrieval_time_path,
    )

    scanner, sq, rot_opts = _build_stealth(
        stealth, evasion_rules, rotate_every, session_field
    )
    if stealth:
        if control_query == "What is 2+2?":
            control_query = sq.control_query()
        else:
            scanner.warn(control_query)
        if domain_query == "What are the company policies?":
            domain_query = sq.domain_query()
        else:
            scanner.warn(domain_query)
        if delay == 1.0:
            delay = 5.0
        if jitter == 0.0:
            jitter = 2.0

    async def _run():
        async with Session(
            target, delay=delay, jitter=jitter, verbose=verbose, timeout=timeout
        ) as session:
            sender = SessionRotator(session, **rot_opts) if rot_opts else session
            det = await run_detect(sender, target, p, control_query, domain_query)
            return det, session.elapsed, session.request_count

    result, elapsed, req_count = _handle_run(_run)

    scan = ScanResult(
        detection=result,
        meta=ScanMeta(
            target=url, preset=preset, queries_sent=req_count, duration_seconds=elapsed
        ),
    )
    if output:
        Path(output).write_text(scan.model_dump_json(indent=2))
    elif json_output:
        typer.echo(scan.model_dump_json(indent=2))
    else:
        render_scan_result(scan)


@app.command()
def enumerate(
    url: Annotated[str, typer.Argument(help="Target RAG API URL")],
    header: HeaderOpt = None,
    cookie: CookieOpt = None,
    proxy: ProxyOpt = None,
    preset: PresetOpt = "generic",
    delay: DelayOpt = 1.0,
    jitter: JitterOpt = 0.0,
    insecure: InsecureOpt = False,
    json_output: JsonOpt = False,
    output: OutputOpt = None,
    verbose: VerboseOpt = False,
    topics: Annotated[
        Optional[str], typer.Option("--topics", help="Topic file (one per line)")
    ] = None,
    exhaustive: Annotated[
        bool, typer.Option("--exhaustive", help="Don't stop early")
    ] = False,
    max_queries: Annotated[
        int, typer.Option("--max-queries", help="Max query count")
    ] = 50,
    source_path: SourcePathOpt = None,
    answer_path: AnswerPathOpt = None,
    score_path: ScorePathOpt = None,
    body_template: BodyTemplateOpt = None,
    query_field: QueryFieldOpt = None,
    title_path: TitlePathOpt = None,
    chunk_id_path: ChunkIdPathOpt = None,
    text_path: TextPathOpt = None,
    retrieval_time_path: RetrievalTimePathOpt = None,
    timeout: TimeoutOpt = 30.0,
    stealth: StealthOpt = False,
    evasion_rules: EvasionRulesOpt = None,
    rotate_every: RotateEveryOpt = 3,
    session_field: SessionFieldOpt = None,
):
    """Phase 2: Enumerate knowledge base documents."""
    _validate_delay(delay)
    target = _build_target(url, header, cookie, proxy, preset, insecure)
    p = _build_preset(
        preset,
        source_path=source_path,
        answer_path=answer_path,
        score_path=score_path,
        body_template=body_template,
        query_field=query_field,
        title_path=title_path,
        chunk_id_path=chunk_id_path,
        text_path=text_path,
        retrieval_time_path=retrieval_time_path,
    )

    scanner, sq, rot_opts = _build_stealth(
        stealth, evasion_rules, rotate_every, session_field
    )
    if stealth:
        if delay == 1.0:
            delay = 5.0
        if jitter == 0.0:
            jitter = 2.0

    topic_list = None
    if topics:
        topic_path = Path(topics)
        if not topic_path.exists():
            typer.echo(f"Error: topics file not found: {topics}", err=True)
            raise typer.Exit(1)
        topic_list = [
            line.strip() for line in topic_path.read_text().splitlines() if line.strip()
        ]

    enum_query_fn = sq.enumerate_queries if sq else None

    async def _run():
        async with Session(
            target, delay=delay, jitter=jitter, verbose=verbose, timeout=timeout
        ) as session:
            sender = SessionRotator(session, **rot_opts) if rot_opts else session
            enum = await run_enumerate(
                sender,
                target,
                p,
                topics=topic_list,
                exhaustive=exhaustive,
                max_queries=max_queries,
                query_fn=enum_query_fn,
            )
            return enum, session.elapsed

    result, elapsed = _handle_run(_run)

    scan = ScanResult(
        enumeration=result,
        meta=ScanMeta(
            target=url,
            preset=preset,
            queries_sent=result.total_queries,
            duration_seconds=elapsed,
        ),
    )
    if output:
        Path(output).write_text(scan.model_dump_json(indent=2))
    elif json_output:
        typer.echo(scan.model_dump_json(indent=2))
    else:
        render_scan_result(scan)


@app.command()
def threshold(
    url: Annotated[str, typer.Argument(help="Target RAG API URL")],
    header: HeaderOpt = None,
    cookie: CookieOpt = None,
    proxy: ProxyOpt = None,
    preset: PresetOpt = "generic",
    delay: DelayOpt = 1.0,
    jitter: JitterOpt = 0.0,
    insecure: InsecureOpt = False,
    json_output: JsonOpt = False,
    output: OutputOpt = None,
    verbose: VerboseOpt = False,
    test_query: Annotated[
        str, typer.Option("--test-query", help="Query to degrade")
    ] = "What is the company policy?",
    source_path: SourcePathOpt = None,
    answer_path: AnswerPathOpt = None,
    score_path: ScorePathOpt = None,
    body_template: BodyTemplateOpt = None,
    query_field: QueryFieldOpt = None,
    title_path: TitlePathOpt = None,
    chunk_id_path: ChunkIdPathOpt = None,
    text_path: TextPathOpt = None,
    retrieval_time_path: RetrievalTimePathOpt = None,
    timeout: TimeoutOpt = 30.0,
    stealth: StealthOpt = False,
    evasion_rules: EvasionRulesOpt = None,
    rotate_every: RotateEveryOpt = 3,
    session_field: SessionFieldOpt = None,
):
    """Phase 3: Map retrieval threshold boundaries."""
    _validate_delay(delay)
    target = _build_target(url, header, cookie, proxy, preset, insecure)
    p = _build_preset(
        preset,
        source_path=source_path,
        answer_path=answer_path,
        score_path=score_path,
        body_template=body_template,
        query_field=query_field,
        title_path=title_path,
        chunk_id_path=chunk_id_path,
        text_path=text_path,
        retrieval_time_path=retrieval_time_path,
    )

    scanner, sq, rot_opts = _build_stealth(
        stealth, evasion_rules, rotate_every, session_field
    )
    if stealth:
        if delay == 1.0:
            delay = 5.0
        if jitter == 0.0:
            jitter = 2.0
        if test_query == "What is the company policy?":
            test_query = sq.domain_query()
        else:
            scanner.warn(test_query)

    async def _run():
        async with Session(
            target, delay=delay, jitter=jitter, verbose=verbose, timeout=timeout
        ) as session:
            sender = SessionRotator(session, **rot_opts) if rot_opts else session
            thresh = await run_threshold(sender, target, p, test_query=test_query)
            return thresh, session.elapsed, session.request_count

    result, elapsed, req_count = _handle_run(_run)

    scan = ScanResult(
        thresholds=result,
        meta=ScanMeta(
            target=url, preset=preset, queries_sent=req_count, duration_seconds=elapsed
        ),
    )
    if output:
        Path(output).write_text(scan.model_dump_json(indent=2))
    elif json_output:
        typer.echo(scan.model_dump_json(indent=2))
    else:
        render_scan_result(scan)


@app.command()
def chunks(
    url: Annotated[
        Optional[str], typer.Argument(help="Unused (retained for compatibility)")
    ] = None,
    input_file: Annotated[
        Optional[str], typer.Option("--input", help="Prior scan JSON to analyze")
    ] = None,
    json_output: JsonOpt = False,
    output: OutputOpt = None,
    tokenizer: TokenizerOpt = "o200k_base",
    max_chunk_tokens: MaxChunkTokensOpt = None,
    embedding_model: EmbeddingModelOpt = None,
):
    """Phase 4: Extract chunk parameters from prior scan data."""
    if not input_file:
        typer.echo("Error: --input is required for standalone chunk analysis", err=True)
        raise typer.Exit(1)

    input_path = Path(input_file)
    if not input_path.exists():
        typer.echo(f"Error: file not found: {input_file}", err=True)
        raise typer.Exit(1)

    scan_data = ScanResult.model_validate_json(input_path.read_text())
    all_sources: list[Source] = []

    if scan_data.detection:
        p = get_preset(scan_data.meta.preset)
        if scan_data.detection.domain_response:
            all_sources.extend(p.extract_sources(scan_data.detection.domain_response))
        if scan_data.detection.control_response:
            all_sources.extend(p.extract_sources(scan_data.detection.control_response))

    if scan_data.enumeration:
        if scan_data.enumeration.sources:
            all_sources.extend(scan_data.enumeration.sources)
        else:
            # Fallback for older scan data without sources field
            for doc in scan_data.enumeration.documents:
                for snippet in doc.text_snippets:
                    all_sources.append(Source(title=doc.title, text=snippet))

    resolved_limit = _resolve_chunk_token_limit(max_chunk_tokens, embedding_model)
    result = run_chunks(
        all_sources, tokenizer_name=tokenizer, max_chunk_tokens=resolved_limit
    )

    scan = ScanResult(chunks=result, meta=scan_data.meta)
    if output:
        Path(output).write_text(scan.model_dump_json(indent=2))
    elif json_output:
        typer.echo(scan.model_dump_json(indent=2))
    else:
        render_scan_result(scan)


@app.command()
def scan(
    url: Annotated[str, typer.Argument(help="Target RAG API URL")],
    header: HeaderOpt = None,
    cookie: CookieOpt = None,
    proxy: ProxyOpt = None,
    preset: PresetOpt = "generic",
    delay: DelayOpt = 1.0,
    jitter: JitterOpt = 0.0,
    insecure: InsecureOpt = False,
    json_output: JsonOpt = False,
    output: OutputOpt = None,
    verbose: VerboseOpt = False,
    control_query: Annotated[str, typer.Option("--control-query")] = "What is 2+2?",
    domain_query: Annotated[
        str, typer.Option("--domain-query")
    ] = "What are the company policies?",
    topics: Annotated[Optional[str], typer.Option("--topics")] = None,
    exhaustive: Annotated[bool, typer.Option("--exhaustive")] = False,
    max_queries: Annotated[int, typer.Option("--max-queries")] = 50,
    test_query: Annotated[Optional[str], typer.Option("--test-query")] = None,
    skip_phase: Annotated[Optional[list[str]], typer.Option("--skip-phase")] = None,
    source_path: SourcePathOpt = None,
    answer_path: AnswerPathOpt = None,
    score_path: ScorePathOpt = None,
    body_template: BodyTemplateOpt = None,
    query_field: QueryFieldOpt = None,
    title_path: TitlePathOpt = None,
    chunk_id_path: ChunkIdPathOpt = None,
    text_path: TextPathOpt = None,
    retrieval_time_path: RetrievalTimePathOpt = None,
    timeout: TimeoutOpt = 30.0,
    stealth: StealthOpt = False,
    evasion_rules: EvasionRulesOpt = None,
    rotate_every: RotateEveryOpt = 3,
    session_field: SessionFieldOpt = None,
    tokenizer: TokenizerOpt = "o200k_base",
    max_chunk_tokens: MaxChunkTokensOpt = None,
    embedding_model: EmbeddingModelOpt = None,
):
    """Run all four reconnaissance phases sequentially."""
    _validate_delay(delay)
    target = _build_target(url, header, cookie, proxy, preset, insecure)
    p = _build_preset(
        preset,
        source_path=source_path,
        answer_path=answer_path,
        score_path=score_path,
        body_template=body_template,
        query_field=query_field,
        title_path=title_path,
        chunk_id_path=chunk_id_path,
        text_path=text_path,
        retrieval_time_path=retrieval_time_path,
    )
    skips = set(skip_phase or [])
    valid_phases = {"detect", "enumerate", "threshold", "chunks"}
    invalid = skips - valid_phases
    if invalid:
        typer.echo(
            f"Error: Unknown phase(s): {', '.join(sorted(invalid))}. Valid: {', '.join(sorted(valid_phases))}",
            err=True,
        )
        raise typer.Exit(1)

    scanner, sq, rot_opts = _build_stealth(
        stealth, evasion_rules, rotate_every, session_field
    )
    if stealth:
        if control_query == "What is 2+2?":
            control_query = sq.control_query()
        else:
            scanner.warn(control_query)
        if domain_query == "What are the company policies?":
            domain_query = sq.domain_query()
        else:
            scanner.warn(domain_query)
        if delay == 1.0:
            delay = 5.0
        if jitter == 0.0:
            jitter = 2.0

    topic_list = None
    if topics:
        topic_path = Path(topics)
        if not topic_path.exists():
            typer.echo(f"Error: topics file not found: {topics}", err=True)
            raise typer.Exit(1)
        topic_list = [
            line.strip() for line in topic_path.read_text().splitlines() if line.strip()
        ]

    enum_query_fn = sq.enumerate_queries if sq else None

    async def _run():
        async with Session(
            target, delay=delay, jitter=jitter, verbose=verbose, timeout=timeout
        ) as session:
            sender = SessionRotator(session, **rot_opts) if rot_opts else session
            detection = None
            enumeration = None
            thresholds = None
            chunk_result = None
            all_sources: list[Source] = []

            if "detect" not in skips:
                detection = await run_detect(
                    sender, target, p, control_query, domain_query
                )
                if detection.domain_response:
                    all_sources.extend(p.extract_sources(detection.domain_response))
                if detection.control_response:
                    all_sources.extend(p.extract_sources(detection.control_response))

            if "enumerate" not in skips:
                enumeration = await run_enumerate(
                    sender,
                    target,
                    p,
                    topics=topic_list,
                    exhaustive=exhaustive,
                    max_queries=max_queries,
                    query_fn=enum_query_fn,
                )
                all_sources.extend(enumeration.sources)

            if "threshold" not in skips:
                tq = test_query
                if tq is None and enumeration and enumeration.documents:
                    top_doc = enumeration.documents[0]
                    if sq:
                        # Stealth: use a stealth topic query for the top doc's topic
                        topic = (
                            top_doc.topics[0] if top_doc.topics else "company policies"
                        )
                        tq = sq.enumerate_query(topic)
                    else:
                        # Normal: use the topic that successfully retrieved the document
                        topic = top_doc.topics[0] if top_doc.topics else top_doc.title
                        tq = f"What does the documentation say about {topic}?"
                tq = tq or "What is the company policy?"
                if sq and tq == "What is the company policy?":
                    tq = sq.domain_query()
                elif scanner and not sq:
                    scanner.warn(tq)
                thresholds = await run_threshold(sender, target, p, test_query=tq)

            # Merge non-enumeration sources into the document inventory
            if enumeration is not None:
                detection_sources: list[Source] = []
                if detection:
                    if detection.domain_response:
                        detection_sources.extend(
                            p.extract_sources(detection.domain_response)
                        )
                    if detection.control_response:
                        detection_sources.extend(
                            p.extract_sources(detection.control_response)
                        )

                existing_titles = {d.title for d in enumeration.documents}
                for source in detection_sources:
                    if source.title not in existing_titles:
                        enumeration.documents.append(
                            Document(
                                title=source.title,
                                hit_count=1,
                                topics=["(detection)"],
                                text_snippets=[source.text] if source.text else [],
                            )
                        )
                        existing_titles.add(source.title)
                enumeration.unique_documents = len(enumeration.documents)

            # Correct detection false negative: if detect said no RAG
            # but enumeration found documents, update the detection result
            if (
                detection
                and not detection.has_rag
                and enumeration
                and enumeration.unique_documents > 0
            ):
                enum_sources = (
                    enumeration.sources
                    if enumeration.sources
                    else [
                        Source(
                            title=d.title,
                            text=d.text_snippets[0] if d.text_snippets else None,
                        )
                        for d in enumeration.documents
                    ]
                )
                detection.has_rag = True
                detection.exposure_level = classify_exposure(enum_sources, [])
                detection.source_count = len(enum_sources)

            if "chunks" not in skips:
                resolved_limit = _resolve_chunk_token_limit(
                    max_chunk_tokens, embedding_model
                )
                chunk_result = run_chunks(
                    all_sources,
                    tokenizer_name=tokenizer,
                    max_chunk_tokens=resolved_limit,
                )

            return ScanResult(
                detection=detection,
                enumeration=enumeration,
                thresholds=thresholds,
                chunks=chunk_result,
                meta=ScanMeta(
                    target=url,
                    preset=preset,
                    queries_sent=session.request_count,
                    duration_seconds=session.elapsed,
                ),
            )

    result = _handle_run(_run)

    if output:
        Path(output).write_text(result.model_dump_json(indent=2))
    elif json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        render_scan_result(result)
