"""Shared TTS orchestration for /tts and OpenAI-compatible speech routes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

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


def _parallel_job_seed(base_seed: int, chunk_index_one_based: int) -> int:
    """Derive a deterministic per-chunk seed so parallel workers do not all call set_seed identically."""
    if base_seed == 0:
        return 0
    mixed = (int(base_seed) + chunk_index_one_based * 0x9E3779B9) & 0x7FFFFFFF
    return mixed if mixed != 0 else 1


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
    chunk_size_clamped = max(chunk_size_min, min(chunk_size_max, int(chunk_size)))
    threshold = chunk_size_clamped * 1.5
    if split_enabled and len(text) > threshold:
        import utils as _utils

        return _utils.chunk_text_by_sentences(text, chunk_size_clamped)
    return [text]


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
) -> Tuple[List[np.ndarray], int]:
    """
    Run chunked synthesis. Speed adjustment is applied once on the stitched waveform
    in the HTTP layer (lower CPU overhead than per-chunk stretching).
    """
    from audio_pipeline import _ensure_mono_waveform_1d

    batch_cfg = config_manager.get_int("tts_engine.chunk_batch_size", 0)
    parallel_workers_cfg = max(
        1, config_manager.get_int("tts_engine.parallel_chunk_workers", 1)
    )
    chunks_count = len(text_chunks)
    # 0 or negative => single batch (one threadpool hop) for lowest orchestration overhead.
    batch_size = chunks_count if batch_cfg <= 0 else max(1, batch_cfg)
    logger.info(
        f"{log_prefix}: chunk synthesis batch_size={batch_size} (cfg={batch_cfg}), "
        f"parallel_chunk_workers={parallel_workers_cfg}, chunks={chunks_count}"
    )
    segments: List[np.ndarray] = []
    engine_sr: Optional[int] = None
    for batch_start in range(0, chunks_count, batch_size):
        batch_end = min(batch_start + batch_size, chunks_count)
        batch_len = batch_end - batch_start
        use_parallel = parallel_workers_cfg > 1 and batch_len > 1
        jobs = []
        for i in range(batch_start, batch_end):
            global_idx = i + 1
            logger.info(
                f"{log_prefix}: queueing chunk {global_idx}/{chunks_count}..."
            )
            # Sequential mode: only chunk 0 prepares reference; others reuse conds.
            # Parallel mode (Extended-style): pass reference on every chunk so workers
            # do not depend on shared self.conds ordering.
            chunk_prompt: Optional[str]
            if use_parallel and audio_prompt_path_str:
                chunk_prompt = audio_prompt_path_str
            elif audio_prompt_path_str and i == 0:
                chunk_prompt = audio_prompt_path_str
            elif audio_prompt_path_str:
                chunk_prompt = None
            else:
                chunk_prompt = None

            job_seed = (
                _parallel_job_seed(params.seed, global_idx)
                if use_parallel
                else params.seed
            )
            jobs.append(
                {
                    "text": text_chunks[i],
                    "audio_prompt_path": chunk_prompt,
                    "temperature": params.temperature,
                    "exaggeration": params.exaggeration,
                    "cfg_weight": params.cfg_weight,
                    "seed": job_seed,
                    "language": params.language,
                    "chunk_index": global_idx,
                    "chunk_total": chunks_count,
                }
            )
        if use_parallel:
            batch_results = await run_in_threadpool(
                engine.synthesize_batch_parallel,
                jobs,
                max_workers=parallel_workers_cfg,
                perf_monitor=perf_monitor,
                log_prefix=log_prefix,
            )
        else:
            batch_results = await run_in_threadpool(
                engine.synthesize_batch,
                jobs,
                perf_monitor=perf_monitor,
                log_prefix=log_prefix,
            )
        for batch_offset, (audio_tensor, sr) in enumerate(batch_results):
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
            chunk_np = audio_tensor.cpu().numpy().squeeze().astype(np.float32)
            segments.append(_ensure_mono_waveform_1d(chunk_np))
    if engine_sr is None:
        raise RuntimeError(f"{log_prefix}: could not determine engine sample rate.")

    max_quality_retries = max(
        0, config_manager.get_int("tts_engine.chunk_quality_max_retries", 0)
    )
    if max_quality_retries > 0 and chunks_count > 0:
        for i in range(chunks_count):
            chunk_text = text_chunks[i]
            chunk_idx = i + 1
            for attempt in range(max_quality_retries + 1):
                reason = await _chunk_quality_failure_reason(
                    segments[i], engine_sr, chunk_text
                )
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
                if audio_tensor is None or sr is None:
                    logger.error(
                        "%sretry failed for chunk %s/%s",
                        log_prefix + ": ",
                        chunk_idx,
                        chunks_count,
                    )
                    break
                chunk_np = audio_tensor.cpu().numpy().squeeze().astype(np.float32)
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
