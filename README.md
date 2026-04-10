# ragmap

Recon tool for RAG pipelines. Point it at a RAG-enabled API endpoint and it'll figure out if there's retrieval happening, what documents are in the knowledge base, where the retrieval thresholds fall off, and how the chunks are structured.

Built for AI red team engagements.

## Get started

```bash
git clone https://github.com/dejisec/ragmap.git
uv sync
uv run ragmap --help
```

```bash
# Full scan -- runs all 4 phases
ragmap scan http://target/api/chat --preset langchain

# With auth
ragmap scan http://target/api/chat \
  -H "Authorization: Bearer tok_xxx" \
  --preset llamaindex

# JSON output
ragmap scan http://target/api/chat --preset haystack --json -o results.json

# Through a proxy
ragmap scan http://target/api/chat \
  --proxy http://127.0.0.1:8080 \
  --preset langchain
```

## How it works

ragmap runs four phases. You can run them individually or use `scan` to run all four sequentially.

### `scan` -- run everything

```bash
ragmap scan <URL> [OPTIONS]
```

### `detect`

Sends a control query and a domain query, compares the responses, and tells you whether retrieval is in play. Also classifies how much metadata the endpoint leaks (none / minimal / moderate / detailed).

```bash
ragmap detect <URL> [OPTIONS]
  --control-query TEXT    General knowledge query (default: "What is 2+2?")
  --domain-query TEXT     Domain-specific query (default: "What are the company policies?")
```

### `enumerate` -- what's in the knowledge base?

Iterates through topics, sends natural-sounding queries, and builds an inventory of documents. Deduplicates by title and tracks how often each doc gets hit.

```bash
ragmap enumerate <URL> [OPTIONS]
  --topics FILE           Custom topic list (one per line)
  --exhaustive            Don't stop on diminishing returns
  --max-queries INT       Query limit (default: 50)
```

### `threshold` -- where does retrieval break?

Takes a reference query and progressively degrades it -- synonym swaps, misspellings, off-topic drift -- to find where the retriever stops returning results.

```bash
ragmap threshold <URL> [OPTIONS]
  --test-query TEXT       Query to degrade (default: "What is the company policy?")
```

### `chunks` -- how is the data chunked?

Offline analysis only (no HTTP requests). Feeds in sources from a prior scan and looks for chunk ID patterns, estimates chunk sizes, and detects text overlap between chunks.

```bash
ragmap chunks --input results.json [OPTIONS]
```

## Common options

These work with `scan`, `detect`, `enumerate`, and `threshold`:

| Flag | What it does |
|------|-------------|
| `-H, --header` | HTTP header `Key: Value` (repeatable) |
| `-b, --cookie` | Cookie `name=value` (repeatable) |
| `--proxy` | Proxy URL (e.g. `http://127.0.0.1:8080`) |
| `--preset` | `generic`, `langchain`, `llamaindex`, `haystack` (default: `generic`) |
| `--delay` | Seconds between requests (default: 1.0) |
| `--jitter` | Random +/- jitter on delay (default: 0.0) |
| `--insecure` | Skip TLS cert verification |
| `--timeout` | Request timeout in seconds (default: 30.0) |
| `--json` | JSON output |
| `-o, --output` | Write output to file |
| `-v, --verbose` | Show request/response details |

## Presets

Presets tell ragmap how to build requests and parse responses for common RAG frameworks. If your target uses something custom, see `generic` below.

### langchain

```bash
ragmap scan http://target/api/chat --preset langchain
```

Sends `{"query": "..."}`, parses `source_documents[].page_content` and `source_documents[].metadata.source`.

### llamaindex

```bash
ragmap scan http://target/api/chat --preset llamaindex
```

Sends `{"query": "..."}`, parses `source_nodes[].node.text` and `source_nodes[].node.metadata.file_name`.

### haystack

```bash
ragmap scan http://target/api/chat --preset haystack
```

Sends `{"query": "..."}`, parses `documents[].content` and `documents[].meta.name`.

### generic

For anything else. You tell ragmap where things are using dot-notation paths:

```bash
ragmap scan http://target/api/chat \
  --preset generic \
  --body-template '{"input": "{query}", "mode": "search"}' \
  --source-path "results[].doc" \
  --answer-path "output.text" \
  --score-path "results[].relevance" \
  --title-path "name" \
  --chunk-id-path "id" \
  --text-path "content"
```

| Flag | Default | What it does |
|------|---------|-------------|
| `--body-template` | `{"query": "..."}` | JSON body with `{query}` placeholder |
| `--query-field` | `query` | Field name for query (when no template) |
| `--source-path` | `sources[]` | Path to the sources array |
| `--answer-path` | `answer` | Path to the LLM answer |
| `--score-path` | `score` | Similarity score field |
| `--title-path` | `title` | Document title field |
| `--chunk-id-path` | `chunk_id` | Chunk ID field |
| `--text-path` | `text` | Chunk text field |
| `--retrieval-time-path` | `retrieval_time_ms` | Retrieval time field |

## Stealth mode

Reduces the chance of tripping detection rules.

```bash
ragmap scan http://target/api/chat \
  --preset langchain \
  --stealth
```

What changes with `--stealth`:

- Default queries swap to non-obvious alternatives that still get the job done
- Delay bumps to 5s (from 1s), jitter to 2s (from 0s) -- unless you override
- Session IDs rotate every N requests to break burst correlation
- User-supplied queries get checked against known detection patterns (you'll get a warning)

| Flag | Default | What it does |
|------|---------|-------------|
| `--stealth` | off | Enable stealth mode |
| `--evasion-rules` | built-in | Custom evasion rules YAML |
| `--rotate-every` | 3 | Rotate session ID every N requests |
| `--session-field` | `session_id` | Request body field for session ID |

### Custom evasion rules

```yaml
categories:
  document_enumeration:
    severity: high
    evasion_tip: "Ask contextual questions that force source citation"
    triggers:
      - "what * documents"
      - "list * sources"
      - "enumerate *"
    burst_threshold: 3
```

Custom rules merge with the defaults:

```bash
ragmap scan http://target/api/chat --stealth --evasion-rules custom.yml
```
