# admission control for TTS / speech (optional)
from __future__ import annotations

import asyncio
import functools
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, TypeVar

from config import config_manager

logger = logging.getLogger(__name__)

_sem: Optional[asyncio.Semaphore] = None
_sem_n: int = 0
T = TypeVar("T")


def _get_semaphore() -> Optional[asyncio.Semaphore]:
    global _sem, _sem_n
    n = max(0, config_manager.get_int("server.max_concurrent_tts_requests", 0))
    if n <= 0:
        return None
    if _sem is None or _sem_n != n:
        _sem = asyncio.Semaphore(n)
        _sem_n = n
        logger.info("TTS admission: max_concurrent_tts_requests=%s", n)
    return _sem


@asynccontextmanager
async def tts_concurrency_slot():
    """If ``server.max_concurrent_tts_requests > 0``, limit concurrent TTS and speech work."""
    sem = _get_semaphore()
    if sem is None:
        yield
    else:
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()


def limit_tts_concurrency(fn):
    """Decorator: wrap an async route handler in ``tts_concurrency_slot``."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        async with tts_concurrency_slot():
            return await fn(*args, **kwargs)

    return wrapper


async def limit_tts_concurrency_stream(stream: AsyncIterator[T]) -> AsyncIterator[T]:
    """Hold the optional TTS admission slot until a streaming response is exhausted."""
    async with tts_concurrency_slot():
        async for item in stream:
            yield item
