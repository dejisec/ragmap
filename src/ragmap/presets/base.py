from __future__ import annotations

from abc import ABC, abstractmethod

from ragmap.models import Score, Source


class Preset(ABC):
    @abstractmethod
    def build_request_body(self, query: str) -> dict: ...

    @abstractmethod
    def extract_sources(self, response: dict) -> list[Source]: ...

    @abstractmethod
    def extract_scores(self, response: dict) -> list[Score]: ...

    @abstractmethod
    def extract_answer(self, response: dict) -> str: ...

    @abstractmethod
    def extract_retrieval_time(self, response: dict) -> float | None: ...
