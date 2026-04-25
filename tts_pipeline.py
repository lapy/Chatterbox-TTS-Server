"""Shared generation pipeline for custom and OpenAI-compatible TTS routes."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np
import torch

import utils
from audio_output import (
    AudioOutputPolicy,
    EncodedAudio,
    EndpointKind,
    encode_audio_with_policy,
    write_encoded_audio_if_enabled,
)
from audio_pipeline import _finalize_stitched_tts_audio
from config import config_manager, get_output_path
from tts_orchestration import ResolvedSynthesisParams, synthesize_text_chunks_async

logger = logging.getLogger(__name__)


CancellationCheck = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class GenerationRequestContext:
    endpoint: EndpointKind
    text_chunks: List[str]
    audio_prompt_path: Optional[Path]
    params: ResolvedSynthesisParams
    output_policy: AudioOutputPolicy
    perf_monitor: Any
    log_prefix: str
    cancellation_check: Optional[CancellationCheck] = None
    download_filename: Optional[str] = None
    save_filename: Optional[str] = None


@dataclass(frozen=True)
class BufferedGenerationResult:
    encoded: EncodedAudio
    headers: Dict[str, str]
    filename: Optional[str]


def build_locked_synthesis_payload(ctx: GenerationRequestContext) -> Dict[str, Any]:
    return {
        "audio_prompt_path": str(ctx.audio_prompt_path) if ctx.audio_prompt_path else None,
        "temperature": ctx.params.temperature,
        "exaggeration": ctx.params.exaggeration,
        "cfg_weight": ctx.params.cfg_weight,
        "seed": ctx.params.seed,
        "language": ctx.params.language,
        "speed_factor": ctx.params.speed_factor,
    }


async def _check_cancel(ctx: GenerationRequestContext) -> None:
    if ctx.cancellation_check is not None:
        await ctx.cancellation_check()


async def synthesize_segments(ctx: GenerationRequestContext) -> tuple[List[np.ndarray], int]:
    await _check_cancel(ctx)
    segments, engine_sr = await synthesize_text_chunks_async(
        ctx.text_chunks,
        str(ctx.audio_prompt_path) if ctx.audio_prompt_path else None,
        ctx.params,
        perf_monitor=ctx.perf_monitor,
        log_prefix=ctx.log_prefix,
        cancellation_check=ctx.cancellation_check,
    )
    await _check_cancel(ctx)
    if not segments:
        raise RuntimeError("Audio generation resulted in no output.")
    return segments, engine_sr


async def postprocess_segments(
    ctx: GenerationRequestContext, segments: List[np.ndarray], engine_sr: int
) -> tuple[np.ndarray, int]:
    await _check_cancel(ctx)
    final_audio_np = _finalize_stitched_tts_audio(
        segments,
        engine_sr,
        perf_monitor=ctx.perf_monitor,
        log_prefix=ctx.log_prefix,
    )
    if ctx.params.speed_factor != 1.0:
        sped_t = torch.from_numpy(final_audio_np.astype(np.float32, copy=False))
        sped_t, engine_sr = utils.apply_speed_factor(
            sped_t, engine_sr, ctx.params.speed_factor
        )
        final_audio_np = sped_t.cpu().numpy().squeeze().astype(np.float32)
        ctx.perf_monitor.record(f"{ctx.log_prefix} speed_factor applied (post-stitch)")
    await _check_cancel(ctx)
    return final_audio_np, engine_sr


async def synthesize_buffered_audio(ctx: GenerationRequestContext) -> EncodedAudio:
    segments, engine_sr = await synthesize_segments(ctx)
    final_audio_np, engine_sr = await postprocess_segments(ctx, segments, engine_sr)
    await _check_cancel(ctx)
    encoded = encode_audio_with_policy(final_audio_np, engine_sr, ctx.output_policy)
    ctx.perf_monitor.record(
        f"{ctx.log_prefix} encoded to {ctx.output_policy.output_format} "
        f"(target SR: {ctx.output_policy.target_sample_rate}Hz from engine SR: {engine_sr}Hz)"
    )
    if len(encoded.data) < 100:
        raise RuntimeError(
            f"Encoded {ctx.output_policy.output_format} output is too small "
            f"({len(encoded.data)} bytes)."
        )
    return encoded


async def synthesize_buffered_response(
    ctx: GenerationRequestContext,
) -> BufferedGenerationResult:
    encoded = await synthesize_buffered_audio(ctx)
    headers: Dict[str, str] = {}
    if ctx.download_filename:
        headers["Content-Disposition"] = f'attachment; filename="{ctx.download_filename}"'

    if config_manager.get_bool("audio_output.save_to_disk", False):
        await _check_cancel(ctx)
        output_dir = get_output_path(ensure_absolute=True)
        filename = (
            ctx.save_filename
            or ctx.download_filename
            or f"tts_output.{ctx.output_policy.output_format}"
        )
        try:
            output_file_path = write_encoded_audio_if_enabled(
                encoded.data, output_dir, filename
            )
            logger.info("Audio saved to disk: %s", output_file_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to save audio file: {exc}") from exc

    if ctx.perf_monitor.enabled:
        logger.info(ctx.perf_monitor.report(log_level=logging.INFO))
    return BufferedGenerationResult(
        encoded=encoded,
        headers=headers,
        filename=ctx.download_filename,
    )


def disconnected_check(request: Any, log_message: str) -> CancellationCheck:
    async def _raise_if_disconnected() -> None:
        if await request.is_disconnected():
            logger.info(log_message)
            raise asyncio.CancelledError()

    return _raise_if_disconnected
