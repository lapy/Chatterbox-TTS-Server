#!/usr/bin/env python3
"""
CUDA / single-request latency benchmark matrix (manual / CI helper).

Measures wall time and optional streaming time-to-first-byte (TTFB) against a
running server. Requires: pip install httpx

Example:
  export CHATTERBOX_BENCH_BASE=http://127.0.0.1:8004
  export CHATTERBOX_BENCH_VOICE=default_sample.wav
  python scripts/benchmark_cuda_latency.py

Enable server.performance_cuda_sync + server.enable_performance_monitor for
GPU-synchronized per-stage logs on the server.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

SHORT_TEXT = "Hello, this is a short latency test."
LONG_TEXT = (
    "This is sentence one for a long form test. "
    "This is sentence two to force chunking. "
    "This is sentence three for more audio. "
    "This is sentence four to keep the GPU busy. "
    "This is sentence five for benchmarking. "
    "This is sentence six before we wrap up. "
    "This is sentence seven still speaking. "
    "This is sentence eight near the end. "
    "This is sentence nine almost done. "
    "This is sentence ten to finish the test."
)


def bench_openai_speech(
    client: Any,
    base: str,
    voice: str,
    text: str,
    response_format: str,
    stream: bool,
) -> Tuple[float, Optional[float]]:
    """Return (total_ms, ttfb_ms or None)."""
    url = f"{base.rstrip('/')}/v1/audio/speech"
    payload: Dict[str, Any] = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": response_format,
    }
    if stream and response_format in ("opus", "mp3"):
        payload["stream_format"] = "audio"

    t0 = time.perf_counter()
    ttfb: Optional[float] = None
    with client.stream("POST", url, json=payload, timeout=600.0) as resp:
        resp.raise_for_status()
        n = 0
        for chunk in resp.iter_bytes(chunk_size=4096):
            if ttfb is None and chunk:
                ttfb = (time.perf_counter() - t0) * 1000.0
            n += len(chunk)
        if n < 100:
            raise RuntimeError(f"short response: {n} bytes")
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, ttfb


def bench_tts(
    client: Any,
    base: str,
    voice: str,
    text: str,
    output_format: str,
    stream: bool,
    split_text: bool,
    chunk_size: int,
) -> Tuple[float, Optional[float]]:
    url = f"{base.rstrip('/')}/tts"
    data = {
        "text": text,
        "voice_mode": "predefined",
        "predefined_voice_id": voice,
        "output_format": output_format,
        "split_text": split_text,
        "chunk_size": chunk_size,
        "stream": stream,
        "temperature": 0.8,
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "seed": 0,
        "speed_factor": 1.0,
    }
    t0 = time.perf_counter()
    ttfb: Optional[float] = None
    with client.stream("POST", url, json=data, timeout=600.0) as resp:
        resp.raise_for_status()
        n = 0
        for chunk in resp.iter_bytes(chunk_size=4096):
            if ttfb is None and chunk:
                ttfb = (time.perf_counter() - t0) * 1000.0
            n += len(chunk)
        if n < 100:
            raise RuntimeError(f"short response: {n} bytes")
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, ttfb


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatterbox TTS latency benchmark")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CHATTERBOX_BENCH_BASE", "http://127.0.0.1:8004"),
    )
    parser.add_argument(
        "--voice",
        default=os.environ.get("CHATTERBOX_BENCH_VOICE", "default_sample.wav"),
    )
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    try:
        import httpx
    except ImportError as e:
        raise SystemExit("Install httpx: pip install httpx") from e

    base = args.base_url
    voice = args.voice
    runs = max(1, args.runs)

    scenarios: List[Tuple[str, callable]] = [
        ("openai short wav", lambda c: bench_openai_speech(c, base, voice, SHORT_TEXT, "wav", False)),
        ("openai long wav (chunked)", lambda c: bench_openai_speech(c, base, voice, LONG_TEXT, "wav", False)),
        ("openai long mp3 stream", lambda c: bench_openai_speech(c, base, voice, LONG_TEXT, "mp3", True)),
        ("tts short wav", lambda c: bench_tts(c, base, voice, SHORT_TEXT, "wav", False, False, 120)),
        ("tts long wav chunked", lambda c: bench_tts(c, base, voice, LONG_TEXT, "wav", False, True, 120)),
        ("tts long mp3 stream", lambda c: bench_tts(c, base, voice, LONG_TEXT, "mp3", True, True, 120)),
    ]

    print(f"Base URL: {base}  voice={voice}  runs={runs}\n")
    with httpx.Client() as client:
        for name, fn in scenarios:
            totals: List[float] = []
            ttfbs: List[float] = []
            for _ in range(runs):
                total_ms, ttfb = fn(client)
                totals.append(total_ms)
                if ttfb is not None:
                    ttfbs.append(ttfb)
            ttfb_str = (
                f"  TTFB p50={statistics.median(ttfbs):.1f} ms"
                if ttfbs
                else "  TTFB n/a"
            )
            print(
                f"{name}:\n"
                f"  total p50={statistics.median(totals):.1f} ms "
                f"(min={min(totals):.1f} max={max(totals):.1f})\n"
                f"{ttfb_str}\n"
            )


if __name__ == "__main__":
    main()
