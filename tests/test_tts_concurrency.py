from __future__ import annotations

import asyncio

import tts_concurrency


def test_limit_tts_concurrency_stream_holds_slot_until_stream_exhausted(monkeypatch):
    monkeypatch.setattr(
        "tts_concurrency.config_manager.get_int",
        lambda key, default=0: 1
        if key == "server.max_concurrent_tts_requests"
        else default,
    )
    tts_concurrency._sem = None
    tts_concurrency._sem_n = 0

    async def scenario():
        release_first = asyncio.Event()
        started = []

        async def source(name):
            started.append(name)
            yield name.encode()
            await release_first.wait()
            yield b"done"

        async def consume(name):
            chunks = []
            async for item in tts_concurrency.limit_tts_concurrency_stream(
                source(name)
            ):
                chunks.append(item)
            return chunks

        first = asyncio.create_task(consume("first"))
        await asyncio.sleep(0)
        second = asyncio.create_task(consume("second"))
        await asyncio.sleep(0.05)

        assert started == ["first"]

        release_first.set()
        first_chunks, second_chunks = await asyncio.gather(first, second)

        assert started == ["first", "second"]
        assert first_chunks == [b"first", b"done"]
        assert second_chunks == [b"second", b"done"]

    try:
        asyncio.run(scenario())
    finally:
        tts_concurrency._sem = None
        tts_concurrency._sem_n = 0
