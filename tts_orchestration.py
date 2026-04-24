"""Shared TTS orchestration for /tts and OpenAI-compatible speech routes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from starlette.concurrency import run_in_threadpool

import engine
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
    chunks_count = len(text_chunks)
    # 0 or negative => single batch (one threadpool hop) for lowest orchestration overhead.
    batch_size = chunks_count if batch_cfg <= 0 else max(1, batch_cfg)
    logger.info(
        f"{log_prefix}: chunk synthesis batch_size={batch_size} (cfg={batch_cfg}), "
        f"chunks={chunks_count}"
    )
    segments: List[np.ndarray] = []
    engine_sr: Optional[int] = None
    for batch_start in range(0, chunks_count, batch_size):
        batch_end = min(batch_start + batch_size, chunks_count)
        jobs = []
        for i in range(batch_start, batch_end):
            global_idx = i + 1
            logger.info(
                f"{log_prefix}: queueing chunk {global_idx}/{chunks_count}..."
            )
            # Only the first chunk of a request prepares reference audio; later chunks
            # reuse chatterbox_model.conds under the same inference lock.
            chunk_prompt: Optional[str]
            if audio_prompt_path_str and i == 0:
                chunk_prompt = audio_prompt_path_str
            elif audio_prompt_path_str:
                chunk_prompt = None
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
    return segments, engine_sr
