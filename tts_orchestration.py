"""Shared TTS orchestration for /tts and OpenAI-compatible speech routes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from starlette.concurrency import run_in_threadpool

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


def make_synthesize_chunk_partial(
    audio_prompt_path_str: Optional[str],
    params: ResolvedSynthesisParams,
) -> Callable[[str], Tuple[np.ndarray, int]]:
    from audio_pipeline import _synthesize_tts_chunk_sync

    # Bind prompt path by keyword so the first positional slot stays free for chunk_text.
    return partial(
        _synthesize_tts_chunk_sync,
        audio_prompt_path_str=audio_prompt_path_str,
        temperature=params.temperature,
        exaggeration=params.exaggeration,
        cfg_weight=params.cfg_weight,
        seed=params.seed,
        language=params.language,
        speed_factor=params.speed_factor,
    )


async def synthesize_text_chunks_async(
    text_chunks: List[str],
    audio_prompt_path_str: Optional[str],
    params: ResolvedSynthesisParams,
    *,
    perf_monitor: Optional[Any] = None,
    log_prefix: str = "TTS",
) -> Tuple[List[np.ndarray], int]:
    synth = make_synthesize_chunk_partial(audio_prompt_path_str, params)
    segments: List[np.ndarray] = []
    engine_sr: Optional[int] = None
    for i, chunk_text in enumerate(text_chunks):
        logger.info(f"{log_prefix}: synthesizing chunk {i + 1}/{len(text_chunks)}...")
        wave, sr = await run_in_threadpool(synth, chunk_text)
        if perf_monitor:
            perf_monitor.record(f"Engine synthesized chunk {i + 1}")
        if engine_sr is None:
            engine_sr = sr
        elif engine_sr != sr:
            logger.warning(
                f"{log_prefix}: inconsistent sample rate on chunk {i + 1} ({sr} Hz vs "
                f"{engine_sr} Hz); continuing with first chunk rate."
            )
        segments.append(wave)
    if engine_sr is None:
        raise RuntimeError(f"{log_prefix}: could not determine engine sample rate.")
    return segments, engine_sr
