from __future__ import annotations

import importlib.resources
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from rich.console import Console


@dataclass
class Hit:
    category: str
    trigger: str
    severity: str
    evasion_tip: str


def _trigger_matches(text: str, pattern: str) -> bool:
    """Check if *pattern* matches anywhere inside *text*.

    Trigger patterns use ``*`` as a wildcard that matches zero or more
    characters (like shell globs).  The match is performed as a
    *substring* search — the pattern does not need to cover the entire
    text.  Splitting the pattern on ``*`` gives a list of literal
    fragments; each fragment must appear in order inside *text*.
    Fragments are stripped of surrounding whitespace so that patterns
    like ``"what * documents"`` match ``"what documents do you have"``.
    """
    parts = pattern.split("*")
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return True
    idx = 0
    for part in parts:
        found = text.find(part, idx)
        if found == -1:
            return False
        idx = found + len(part)
    return True


class KeywordScanner:
    def __init__(
        self,
        rules_path: Path | None = None,
        overlay_path: Path | None = None,
    ):
        if rules_path is None:
            ref = importlib.resources.files("ragmap.rules").joinpath("default.yml")
            with importlib.resources.as_file(ref) as p:
                rules_path = p
                self._rules = yaml.safe_load(rules_path.read_text())
        else:
            self._rules = yaml.safe_load(rules_path.read_text())

        self.categories: dict[str, dict] = self._rules.get("categories", {})

        if overlay_path is not None:
            if not overlay_path.exists():
                raise FileNotFoundError(f"Evasion rules file not found: {overlay_path}")
            overlay = yaml.safe_load(overlay_path.read_text())
            for name, cat in overlay.get("categories", {}).items():
                if name in self.categories:
                    if "triggers" in cat:
                        self.categories[name]["triggers"].extend(cat["triggers"])
                    if "severity" in cat:
                        self.categories[name]["severity"] = cat["severity"]
                    if "evasion_tip" in cat:
                        self.categories[name]["evasion_tip"] = cat["evasion_tip"]
                    if "burst_threshold" in cat:
                        self.categories[name]["burst_threshold"] = cat[
                            "burst_threshold"
                        ]
                else:
                    self.categories[name] = cat

        self._burst_counts: dict[str, list[float]] = {}
        self._burst_cooldown = 60.0

    def check(self, query: str) -> list[Hit]:
        query_lower = query.lower()
        now = time.monotonic()
        hits: list[Hit] = []

        for cat_name, cat in self.categories.items():
            burst_threshold = cat.get("burst_threshold")
            matched_trigger: str | None = None

            for trigger in cat.get("triggers", []):
                if _trigger_matches(query_lower, trigger.lower()):
                    matched_trigger = trigger
                    break

            if matched_trigger is None:
                continue

            if burst_threshold is not None:
                timestamps = self._burst_counts.setdefault(cat_name, [])
                timestamps.append(now)
                timestamps[:] = [
                    t for t in timestamps if now - t < self._burst_cooldown
                ]
                if len(timestamps) < burst_threshold:
                    continue

            hits.append(
                Hit(
                    category=cat_name,
                    trigger=matched_trigger,
                    severity=cat.get("severity", "medium"),
                    evasion_tip=cat.get("evasion_tip", ""),
                )
            )

        return hits

    def warn(self, query: str) -> None:
        hits = self.check(query)
        if not hits:
            return
        # Build a fresh console each call so pytest capsys can capture stderr
        console = Console(file=sys.stderr)
        for hit in hits:
            console.print(
                f"[bold yellow][!] OPSEC WARNING: Query may trigger detection[/bold yellow]\n"
                f"    Category: {hit.category} ({hit.severity.upper()})\n"
                f'    Matched:  "{hit.trigger}"\n'
                f'    Query:    "{query}"\n'
                f"    Tip:      {hit.evasion_tip}"
            )
