from __future__ import annotations

from pydantic import BaseModel, Field


class Target(BaseModel):
    url: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    cookies: dict[str, str] = Field(default_factory=dict)
    proxy: str | None = None
    preset_name: str = "generic"
    insecure: bool = False


def parse_header(raw: str) -> tuple[str, str]:
    key, _, value = raw.partition(":")
    return key.strip(), value.strip()


def parse_cookie(raw: str) -> tuple[str, str]:
    key, _, value = raw.partition("=")
    return key.strip(), value.strip()
