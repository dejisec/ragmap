from __future__ import annotations

import asyncio
import json
import random
import time
from types import TracebackType
from typing import Protocol, runtime_checkable

import httpx
from rich.console import Console

from ragmap.config import Target

_console = Console(stderr=True)


@runtime_checkable
class Sender(Protocol):
    @property
    def request_count(self) -> int: ...

    @property
    def elapsed(self) -> float: ...

    async def send(self, method: str, url: str, **kwargs) -> httpx.Response: ...


class Session:
    def __init__(
        self,
        target: Target,
        delay: float = 1.0,
        jitter: float = 0.0,
        verbose: bool = False,
        timeout: float = 30.0,
    ):
        self._delay = delay
        self._jitter = jitter
        self._verbose = verbose
        self.request_count = 0
        self._start_time = time.monotonic()
        self._client = httpx.AsyncClient(
            headers=target.headers,
            cookies=target.cookies,
            proxy=target.proxy,
            verify=not target.insecure,
            timeout=httpx.Timeout(timeout),
        )

    async def send(self, method: str, url: str, **kwargs) -> httpx.Response:
        if self.request_count > 0 and self._delay > 0:
            wait = self._delay + random.uniform(-self._jitter, self._jitter)
            await asyncio.sleep(max(0.0, wait))
        self.request_count += 1

        if self._verbose:
            _console.print(f"[dim]>>> {method} {url}[/dim]")
            if "json" in kwargs:
                _console.print(f"[dim]{json.dumps(kwargs['json'], indent=2)}[/dim]")

        resp = await self._client.request(method, url, **kwargs)

        if self._verbose:
            _console.print(f"[dim]<<< {resp.status_code}[/dim]")
            try:
                body = resp.json()
                _console.print(f"[dim]{json.dumps(body, indent=2, default=str)}[/dim]")
            except Exception:
                _console.print(f"[dim]{resp.text[:500]}[/dim]")

        return resp

    @staticmethod
    def parse_json(resp: httpx.Response) -> dict:
        """Parse JSON from response with a user-friendly error on failure."""
        try:
            return resp.json()
        except Exception:
            preview = resp.text[:200] if resp.text else "(empty body)"
            raise RuntimeError(
                f"Target returned non-JSON response (HTTP {resp.status_code}). "
                f"Preview: {preview}"
            )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start_time

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Session:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
