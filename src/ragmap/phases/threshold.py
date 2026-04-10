from __future__ import annotations

import random

from ragmap.config import Target
from ragmap.models import ThresholdResult, ThresholdVariant
from ragmap.presets.base import Preset
from ragmap.session import Sender, Session


SYNONYM_MAP: dict[str, list[str]] = {
    "policy": ["rules", "guidelines", "regulations"],
    "employee": ["staff", "worker", "personnel"],
    "company": ["organization", "firm", "business"],
    "security": ["safety", "protection", "defense"],
    "architecture": ["design", "structure", "framework"],
    "documentation": ["docs", "manuals", "guides"],
    "internal": ["private", "in-house", "proprietary"],
    "api": ["interface", "endpoint", "service"],
    "infrastructure": ["systems", "platform", "environment"],
    "access": ["permission", "authorization", "privilege"],
    "pto": ["vacation", "leave", "days away"],
    "data": ["information", "records", "content"],
    "password": ["credential", "passphrase", "secret"],
    "database": ["datastore", "storage", "repository"],
    "network": ["connectivity", "communications", "topology"],
    "question": ["inquiry", "query", "concern"],
    "process": ["procedure", "method", "workflow"],
    "requesting": ["asking for", "applying for", "seeking"],
    "days": ["time", "leave", "period"],
    "review": ["check", "go over", "look at"],
    "steps": ["procedure", "instructions", "stages"],
    "onboarding": ["orientation", "induction", "training"],
    "connect": ["access", "reach", "link to"],
    "set": ["configure", "establish", "arrange"],
    "tools": ["utilities", "applications", "software"],
    "reporting": ["filing", "submitting", "logging"],
    "concern": ["issue", "problem", "matter"],
    "monitoring": ["tracking", "observing", "watching"],
}

OFF_TOPIC_QUERIES = [
    "What is the speed of light in a vacuum?",
    "How many legs does a spider have?",
    "What year was the Eiffel Tower built?",
]


def apply_synonyms(text: str) -> str:
    words = text.split()
    result = []
    for word in words:
        clean = word.lower().strip("?.,!;:")
        if clean in SYNONYM_MAP:
            replacement = SYNONYM_MAP[clean][0]
            # Find where the clean word starts and ends within the original
            lower_word = word.lower()
            start = lower_word.find(clean)
            end = start + len(clean)
            prefix = word[:start]
            suffix = word[end:]
            if start < len(word) and word[start].isupper():
                replacement = replacement.capitalize()
            result.append(prefix + replacement + suffix)
        else:
            result.append(word)
    return " ".join(result)


def apply_light_misspelling(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    words = text.split()
    result = []
    for word in words:
        if len(word) <= 2:
            result.append(word)
            continue
        chars = list(word)
        i = rng.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        result.append("".join(chars))
    return " ".join(result)


def apply_heavy_misspelling(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    words = text.split()
    result = []
    for word in words:
        if len(word) <= 2:
            result.append(word)
            continue
        chars = list(word)
        for _ in range(min(3, len(chars) - 1)):
            mutation = rng.choice(["swap", "double", "drop"])
            i = rng.randint(0, max(0, len(chars) - 2))
            if mutation == "swap" and i < len(chars) - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif mutation == "double":
                chars.insert(i, chars[i])
            elif mutation == "drop" and len(chars) > 2:
                chars.pop(i)
        result.append("".join(chars))
    return " ".join(result)


async def run_threshold(
    session: Sender,
    target: Target,
    preset: Preset,
    test_query: str = "What is the company policy?",
) -> ThresholdResult:
    variants_config = [
        ("exact", test_query),
        ("synonym", apply_synonyms(test_query)),
        ("light_misspelling", apply_light_misspelling(test_query, seed=42)),
        ("heavy_misspelling", apply_heavy_misspelling(test_query, seed=42)),
        ("off-topic", OFF_TOPIC_QUERIES[0]),
    ]

    variants: list[ThresholdVariant] = []

    for variant_name, query in variants_config:
        body = preset.build_request_body(query)
        resp = await session.send(target.method, target.url, json=body)
        resp.raise_for_status()
        data = Session.parse_json(resp)

        sources = preset.extract_sources(data)
        scores = preset.extract_scores(data)

        source_count = len(sources)
        avg_score = None
        if scores:
            score_values = [
                s.combined_score for s in scores if s.combined_score is not None
            ]
            if score_values:
                avg_score = sum(score_values) / len(score_values)

        retrieved = source_count > 0

        variants.append(
            ThresholdVariant(
                variant=variant_name,
                query=query,
                source_count=source_count,
                avg_score=avg_score,
                retrieved=retrieved,
            )
        )

    boundary = _determine_boundary(variants)
    return ThresholdResult(variants=variants, threshold_boundary=boundary)


def _determine_boundary(variants: list[ThresholdVariant]) -> str:
    ordered_names = [
        "exact",
        "synonym",
        "light_misspelling",
        "heavy_misspelling",
        "off-topic",
    ]
    for i, name in enumerate(ordered_names):
        variant = next((v for v in variants if v.variant == name), None)
        if variant and not variant.retrieved:
            if i == 0:
                return "retrieval fails even on exact match"
            return f"retrieval fails at {name.replace('_', ' ')} level"
    return "retrieval survives all degradation levels"
