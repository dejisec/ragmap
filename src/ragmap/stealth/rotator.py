from __future__ import annotations

from types import TracebackType
from uuid import uuid4

import httpx

from ragmap.session import Session


class SessionRotator:
    def __init__(
        self,
        session: Session,
        rotate_every: int = 3,
        session_field: str = "session_id",
    ):
        if rotate_every < 1:
            raise ValueError("rotate_every must be >= 1")
        self._session = session
        self.rotate_every = rotate_every
        self.session_field = session_field
        self._rotation_count = 0
        self._session_id = uuid4().hex

    async def send(self, method: str, url: str, **kwargs) -> httpx.Response:
        self._inject_session_id(kwargs)
        self._rotation_count += 1
        response = await self._session.send(method, url, **kwargs)
        if self._rotation_count % self.rotate_every == 0:
            self._session_id = uuid4().hex
        return response

    def _inject_session_id(self, kwargs: dict) -> None:
        if "json" in kwargs and isinstance(kwargs["json"], dict):
            kwargs["json"][self.session_field] = self._session_id

    @property
    def request_count(self) -> int:
        return self._session.request_count

    @property
    def elapsed(self) -> float:
        return self._session.elapsed

    async def close(self) -> None:
        await self._session.close()

    async def __aenter__(self) -> SessionRotator:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
