"""Custom POST /tts route."""
import io
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse

import engine
import utils
from audio_output import AudioOutputPolicy, EndpointKind, StreamFormat, streaming_headers
from audio_pipeline import _stream_encoded_audio_from_pcm
from config import (
    config_manager,
    get_audio_output_format,
    get_audio_sample_rate,
    get_predefined_voices_path,
    get_reference_audio_path,
)
from models import CustomTTSRequest, ErrorResponse
from tts_concurrency import limit_tts_concurrency, limit_tts_concurrency_stream
from tts_orchestration import (
    build_text_chunks,
    resolve_synthesis_params_custom,
)
from tts_pipeline import (
    GenerationRequestContext,
    build_locked_synthesis_payload,
    disconnected_check,
    synthesize_buffered_response,
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/tts",
    tags=["TTS Generation"],
    summary="Generate speech with custom parameters",
    responses={
        200: {
            "content": {
                "audio/wav": {},
                "audio/mpeg": {},
                "audio/ogg": {},
            },
            "description": "Successful audio generation.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or input.",
        },
        404: {
            "model": ErrorResponse,
            "description": "Required resource not found (e.g., voice file).",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during generation.",
        },
        503: {
            "model": ErrorResponse,
            "description": "TTS engine not available or model not loaded.",
        },
    },
)
@limit_tts_concurrency
async def custom_tts_endpoint(
    request: CustomTTSRequest, background_tasks: BackgroundTasks, http_request: Request
):
    """
    Generates speech audio from text using specified parameters.
    Handles various voice modes (predefined, clone) and audio processing options.
    Returns audio as a stream (WAV or Opus).
    """
    perf_monitor = utils.PerformanceMonitor(
        enabled=config_manager.get_bool("server.enable_performance_monitor", False),
        request_id=str(uuid.uuid4())[:12],
        cuda_sync=config_manager.get_bool("server.performance_cuda_sync", False),
    )
    perf_monitor.record("TTS request received")
    try:
        utils.validate_request_text_length(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    requested_output_format = request.output_format if request.output_format else get_audio_output_format()
    logger.info(
        "TTS request intake: endpoint=/tts request_id=%s streaming=%s "
        "output_format=%s voice_mode=%s split_text=%s chunk_size=%s text_chars=%s",
        perf_monitor.request_id,
        bool(request.stream),
        requested_output_format,
        request.voice_mode,
        bool(request.split_text),
        request.chunk_size,
        len(request.text or ""),
    )

    if not engine.MODEL_LOADED:
        logger.error("TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.debug(
        f"TTS params: seed={request.seed}, split={request.split_text}, chunk_size={request.chunk_size}"
    )
    logger.debug(f"Input text (first 100 chars): '{request.text[:100]}...'")

    cancellation_check = disconnected_check(
        http_request, "/tts client disconnected; stopping non-streaming generation."
    )

    audio_prompt_path_for_engine: Optional[Path] = None
    if request.voice_mode == "predefined":
        if not request.predefined_voice_id:
            raise HTTPException(
                status_code=400,
                detail="Missing 'predefined_voice_id' for 'predefined' voice mode.",
            )
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        try:
            potential_path = utils.resolve_file_under_directory(
                voices_dir,
                request.predefined_voice_id,
                allowed_suffixes={".wav", ".mp3"},
            )
        except FileNotFoundError:
            logger.error("Predefined voice file not found: %s", request.predefined_voice_id)
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file '{request.predefined_voice_id}' not found.",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        audio_prompt_path_for_engine = potential_path
        logger.info(f"Using predefined voice: {request.predefined_voice_id}")

    elif request.voice_mode == "clone":
        if not request.reference_audio_filename:
            raise HTTPException(
                status_code=400,
                detail="Missing 'reference_audio_filename' for 'clone' voice mode.",
            )
        ref_dir = get_reference_audio_path(ensure_absolute=True)
        try:
            potential_path = utils.resolve_file_under_directory(
                ref_dir,
                request.reference_audio_filename,
                allowed_suffixes={".wav", ".mp3"},
            )
        except FileNotFoundError:
            logger.error(
                "Reference audio file for cloning not found: %s",
                request.reference_audio_filename,
            )
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file '{request.reference_audio_filename}' not found.",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
        is_valid, msg = utils.validate_reference_audio(potential_path, max_dur)
        if not is_valid:
            raise HTTPException(
                status_code=400, detail=f"Invalid reference audio: {msg}"
            )
        audio_prompt_path_for_engine = potential_path
        logger.info(
            f"Using reference audio for cloning: {request.reference_audio_filename}"
        )

    perf_monitor.record("Parameters and voice path resolved")

    final_output_sample_rate = get_audio_sample_rate()
    params = resolve_synthesis_params_custom(request)
    chunk_size_to_use = request.chunk_size if request.chunk_size is not None else 120
    text_chunks = build_text_chunks(
        request.text,
        split_enabled=bool(request.split_text),
        chunk_size=chunk_size_to_use,
        chunk_size_min=50,
        chunk_size_max=500,
    )
    perf_monitor.record(f"Text split into {len(text_chunks)} chunks")

    if not text_chunks:
        raise HTTPException(
            status_code=400, detail="Text processing resulted in no usable chunks."
        )

    output_format_str = requested_output_format
    logger.info(
        "TTS request resolved: endpoint=/tts request_id=%s streaming=%s "
        "output_format=%s chunks=%s voice_mode=%s",
        perf_monitor.request_id,
        bool(request.stream),
        output_format_str,
        len(text_chunks),
        request.voice_mode,
    )
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    param_tag = (
        f"T{params.temperature:.1f}_E{params.exaggeration:.1f}_W{params.cfg_weight:.1f}".replace(
            ".", ""
        )
    )
    suggested_filename_base = f"tts_output_{param_tag}_{timestamp_str}"
    download_filename = utils.sanitize_filename(
        f"{suggested_filename_base}.{output_format_str}"
    )

    if request.stream:
        if output_format_str not in ("opus", "mp3"):
            raise HTTPException(
                status_code=400,
                detail="When 'stream' is true, output_format must be 'opus' or 'mp3'.",
            )
        if shutil.which("ffmpeg") is None:
            raise HTTPException(
                status_code=503,
                detail="ffmpeg is required for streaming opus/mp3 output.",
            )
        output_policy = AudioOutputPolicy(
            output_format=output_format_str,
            target_sample_rate=final_output_sample_rate,
            chunk_count=len(text_chunks),
            stream_format=StreamFormat.AUDIO,
        )
        ctx = GenerationRequestContext(
            endpoint=EndpointKind.CUSTOM,
            text_chunks=text_chunks,
            audio_prompt_path=audio_prompt_path_for_engine,
            params=params,
            output_policy=output_policy,
            perf_monitor=perf_monitor,
            log_prefix="/tts stream",
            cancellation_check=cancellation_check,
            download_filename=download_filename,
        )
        headers = streaming_headers(output_policy, download_filename=download_filename)
        logger.info(
            f"Streaming /tts ({output_format_str}), {len(text_chunks)} text chunk(s)."
        )
        stream_iter = _stream_encoded_audio_from_pcm(
            text_chunks=text_chunks,
            target_sample_rate=final_output_sample_rate,
            output_format=output_format_str,
            sse=False,
            log_prefix="/tts stream",
            locked_synthesis=build_locked_synthesis_payload(ctx),
            perf_monitor=perf_monitor,
            timing_policy=output_policy.timing,
        )
        return StreamingResponse(
            limit_tts_concurrency_stream(stream_iter),
            media_type=output_policy.media_type,
            headers=headers,
        )

    output_policy = AudioOutputPolicy(
        output_format=output_format_str,
        target_sample_rate=final_output_sample_rate,
        chunk_count=len(text_chunks),
        stream_format=StreamFormat.NONE,
    )
    ctx = GenerationRequestContext(
        endpoint=EndpointKind.CUSTOM,
        text_chunks=text_chunks,
        audio_prompt_path=audio_prompt_path_for_engine,
        params=params,
        output_policy=output_policy,
        perf_monitor=perf_monitor,
        log_prefix="/tts",
        cancellation_check=cancellation_check,
        download_filename=download_filename,
    )
    try:
        result = await synthesize_buffered_response(ctx)
    except Exception as e:
        logger.error(f"Error during TTS generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        "Successfully generated audio: %s, %s bytes, type %s.",
        download_filename,
        len(result.encoded.data),
        result.encoded.media_type,
    )

    return StreamingResponse(
        io.BytesIO(result.encoded.data),
        media_type=result.encoded.media_type,
        headers=result.headers,
    )
