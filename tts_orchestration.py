"""Shared TTS orchestration for /tts and OpenAI-compatible speech routes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import numpy as np
from starlette.concurrency import run_in_threadpool

import asr_validation
import engine
import utils as _utils_quality
from config import (
    config_manager,
    get_gen_default_cfg_weight,
    get_gen_default_exaggeration,
    get_gen_default_language,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_gen_default_temperature,
)
from models import CustomTTSRequest, OpenAISpeechRequest

logger = logging.getLogger(__name__)


def _to_float32_chunk_numpy(audio_tensor: Any) -> np.ndarray:
    """Convert engine output tensor to float32 numpy with minimal copies."""
    if hasattr(audio_tensor, "device") and getattr(audio_tensor.device, "type", "") != "cpu":
        audio_tensor = audio_tensor.cpu()
    return audio_tensor.numpy().squeeze().astype(np.float32)


@dataclass(frozen=True)
class ResolvedSynthesisParams:
    temperature: float
    exaggeration: float
    cfg_weight: float
    seed: int
    language: str
    speed_factor: float


def resolve_synthesis_params_custom(request: CustomTTSRequest) -> ResolvedSynthesisParams:
    return ResolvedSynthesisParams(
        temperature=(
            request.temperature
            if request.temperature is not None
            else get_gen_default_temperature()
        ),
        exaggeration=(
            request.exaggeration
            if request.exaggeration is not None
            else get_gen_default_exaggeration()
        ),
        cfg_weight=(
            request.cfg_weight
            if request.cfg_weight is not None
            else get_gen_default_cfg_weight()
        ),
        seed=request.seed if request.seed is not None else get_gen_default_seed(),
        language=(
            request.language
            if request.language is not None
            else get_gen_default_language()
        ),
        speed_factor=(
            request.speed_factor
            if request.speed_factor is not None
            else get_gen_default_speed_factor()
        ),
    )


def resolve_synthesis_params_openai(request: OpenAISpeechRequest) -> ResolvedSynthesisParams:
    """
    Map an OpenAI speech request to engine parameters.
    temperature, exaggeration, cfg_weight, and language are not in the OpenAI schema;
    they are read from ui_state (last_*), falling back to generation_defaults.
    """
    seed = request.seed if request.seed is not None else get_gen_default_seed()
    speed = request.speed * get_gen_default_speed_factor()
    speed = max(0.25, min(4.0, speed))
    temperature = config_manager.get_float(
        "ui_state.last_temperature", get_gen_default_temperature()
    )
    exaggeration = config_manager.get_float(
        "ui_state.last_exaggeration", get_gen_default_exaggeration()
    )
    cfg_weight = config_manager.get_float(
        "ui_state.last_cfg_weight", get_gen_default_cfg_weight()
    )
    lang_raw = config_manager.get("ui_state.last_language")
    if isinstance(lang_raw, str) and lang_raw.strip():
        language = lang_raw.strip()
    else:
        language = get_gen_default_language()
    return ResolvedSynthesisParams(
        temperature=temperature,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        seed=seed,
        language=language,
        speed_factor=speed,
    )


def build_text_chunks(
    text: str,
    *,
    split_enabled: bool,
    chunk_size: int,
    chunk_size_min: int = 50,
    chunk_size_max: int = 1000,
) -> List[str]:
    import utils as _utils

    normalized_text = _utils.normalize_markdown_for_tts(text)
    chunk_size_clamped = max(chunk_size_min, min(chunk_size_max, int(chunk_size)))
    threshold = chunk_size_clamped * 1.5
    if split_enabled and len(normalized_text) > threshold:
        return _utils.chunk_text_by_sentences(
            normalized_text, chunk_size_clamped, text_is_normalized=True
        )
    return [normalized_text] if normalized_text else []


async def _chunk_quality_failure_reason(
    segment: np.ndarray,
    sample_rate: int,
    chunk_text: str,
) -> Optional[str]:
    glitch = _utils_quality.detect_chunk_audio_glitch(
        segment, sample_rate, chunk_text
    )
    if glitch:
        return glitch
    if not config_manager.get_bool("asr.enabled", False):
        return None
    return await run_in_threadpool(
        asr_validation.transcription_mismatch_reason,
        segment,
        sample_rate,
        chunk_text,
    )


async def synthesize_text_chunks_async(
    text_chunks: List[str],
    audio_prompt_path_str: Optional[str],
    params: ResolvedSynthesisParams,
    *,
    perf_monitor: Optional[Any] = None,
    log_prefix: str = "TTS",
    cancellation_check: Optional[Callable[[], Awaitable[None]]] = None,
) -> Tuple[List[np.ndarray], int]:
    """
    Run chunked synthesis. Speed adjustment is applied once on the stitched waveform
    in the HTTP layer (lower CPU overhead than per-chunk stretching).
    """
    from audio_pipeline import _ensure_mono_waveform_1d

    batch_cfg = config_manager.get_int("tts_engine.chunk_batch_size", 0)
    chunks_count = len(text_chunks)
    # 0 or negative => single batch (one threadpool hop) for lowest orchestration overhead.
    if batch_cfg <= 0:
        batch_size = chunks_count
    else:
        batch_size = max(1, batch_cfg)
    logger.debug(
        f"{log_prefix}: sequential chunk synthesis batch_size={batch_size} "
        f"(cfg={batch_cfg}), chunks={chunks_count}"
    )
    segments: List[np.ndarray] = []
    engine_sr: Optional[int] = None
    for batch_start in range(0, chunks_count, batch_size):
        if cancellation_check is not None:
            await cancellation_check()
        batch_end = min(batch_start + batch_size, chunks_count)
        jobs = []
        for i in range(batch_start, batch_end):
            global_idx = i + 1
            logger.debug(
                f"{log_prefix}: queueing chunk {global_idx}/{chunks_count}..."
            )
            # First chunk in each lock-held batch prepares reference conditionals.
            # Later chunks in that batch reuse self.conds while synthesize_batch holds the lock.
            if audio_prompt_path_str and i == batch_start:
                chunk_prompt = audio_prompt_path_str
            else:
                chunk_prompt = None

            jobs.append(
                {
                    "text": text_chunks[i],
                    "audio_prompt_path": chunk_prompt,
                    "temperature": params.temperature,
                    "exaggeration": params.exaggeration,
                    "cfg_weight": params.cfg_weight,
                    "seed": params.seed,
                    "language": params.language,
                    "chunk_index": global_idx,
                    "chunk_total": chunks_count,
                }
            )
        batch_results = await run_in_threadpool(
            engine.synthesize_batch,
            jobs,
            perf_monitor=perf_monitor,
            log_prefix=log_prefix,
        )
        if cancellation_check is not None:
            await cancellation_check()
        for batch_offset, (audio_tensor, sr) in enumerate(batch_results):
            if cancellation_check is not None:
                await cancellation_check()
            chunk_index = batch_start + batch_offset + 1
            if audio_tensor is None or sr is None:
                raise RuntimeError(
                    f"{log_prefix}: engine failed to synthesize chunk {chunk_index}."
                )
            if perf_monitor:
                perf_monitor.record(
                    f"{log_prefix} postprocess chunk {chunk_index}/{chunks_count} (numpy)"
                )
            if engine_sr is None:
                engine_sr = sr
            elif engine_sr != sr:
                logger.warning(
                    f"{log_prefix}: inconsistent sample rate on chunk {chunk_index} ({sr} Hz vs "
                    f"{engine_sr} Hz); continuing with first chunk rate."
                )
            chunk_np = _to_float32_chunk_numpy(audio_tensor)
            segments.append(_ensure_mono_waveform_1d(chunk_np))
    if engine_sr is None:
        raise RuntimeError(f"{log_prefix}: could not determine engine sample rate.")

    max_quality_retries = max(
        0, config_manager.get_int("tts_engine.chunk_quality_max_retries", 0)
    )
    if max_quality_retries > 0 and chunks_count > 0:
        for i in range(chunks_count):
            if cancellation_check is not None:
                await cancellation_check()
            chunk_text = text_chunks[i]
            chunk_idx = i + 1
            for attempt in range(max_quality_retries + 1):
                if cancellation_check is not None:
                    await cancellation_check()
                reason = await _chunk_quality_failure_reason(
                    segments[i], engine_sr, chunk_text
                )
                if cancellation_check is not None:
                    await cancellation_check()
                if reason is None:
                    break
                if attempt >= max_quality_retries:
                    logger.error(
                        "%schunk %s/%s still flagged after %s retries: %s",
                        log_prefix + ": ",
                        chunk_idx,
                        chunks_count,
                        max_quality_retries,
                        reason,
                    )
                    break
                logger.warning(
                    "%schunk %s/%s quality check '%s' — retry %s/%s",
                    log_prefix + ": ",
                    chunk_idx,
                    chunks_count,
                    reason,
                    attempt + 1,
                    max_quality_retries,
                )
                retry_seed = _utils_quality.derive_chunk_retry_seed(
                    params.seed, chunk_idx, attempt + 1
                )
                # Single-chunk resynthesis: always pass reference when set so conds are deterministic.
                prompt = audio_prompt_path_str if audio_prompt_path_str else None
                audio_tensor, sr = await run_in_threadpool(
                    engine.synthesize,
                    chunk_text,
                    prompt,
                    params.temperature,
                    params.exaggeration,
                    params.cfg_weight,
                    retry_seed,
                    params.language,
                    perf_monitor=None,
                    chunk_index=chunk_idx,
                    chunk_total=chunks_count,
                    log_prefix=f"{log_prefix} retry",
                )
                if cancellation_check is not None:
                    await cancellation_check()
                if audio_tensor is None or sr is None:
                    logger.error(
                        "%sretry failed for chunk %s/%s",
                        log_prefix + ": ",
                        chunk_idx,
                        chunks_count,
                    )
                    break
                chunk_np = _to_float32_chunk_numpy(audio_tensor)
                segments[i] = _ensure_mono_waveform_1d(chunk_np)
                if sr != engine_sr:
                    logger.warning(
                        "%sretry chunk %s sample rate %s vs engine %s — keeping numpy segment",
                        log_prefix + ": ",
                        chunk_idx,
                        sr,
                        engine_sr,
                    )

    return segments, engine_sr
