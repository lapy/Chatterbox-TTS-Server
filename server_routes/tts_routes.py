"""Custom POST /tts route."""
import io
import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

import torch

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse

import engine
import utils
from audio_pipeline import (
    _finalize_stitched_tts_audio,
    _get_audio_media_type,
    _stream_encoded_audio_from_pcm,
)
from config import (
    config_manager,
    get_audio_output_format,
    get_audio_sample_rate,
    get_output_path,
    get_predefined_voices_path,
    get_reference_audio_path,
)
from models import CustomTTSRequest, ErrorResponse
from tts_concurrency import limit_tts_concurrency, limit_tts_concurrency_stream
from tts_orchestration import (
    build_text_chunks,
    resolve_synthesis_params_custom,
    synthesize_text_chunks_async,
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

    if not engine.MODEL_LOADED:
        logger.error("TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.info(
        f"Received /tts request: mode='{request.voice_mode}', format='{request.output_format}'"
    )
    logger.debug(
        f"TTS params: seed={request.seed}, split={request.split_text}, chunk_size={request.chunk_size}"
    )
    logger.debug(f"Input text (first 100 chars): '{request.text[:100]}...'")

    async def raise_if_disconnected() -> None:
        if await http_request.is_disconnected():
            logger.info("/tts client disconnected; stopping non-streaming generation.")
            raise asyncio.CancelledError()

    audio_prompt_path_for_engine: Optional[Path] = None
    if request.voice_mode == "predefined":
        if not request.predefined_voice_id:
            raise HTTPException(
                status_code=400,
                detail="Missing 'predefined_voice_id' for 'predefined' voice mode.",
            )
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        potential_path = voices_dir / request.predefined_voice_id
        if not potential_path.is_file():
            logger.error(f"Predefined voice file not found: {potential_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file '{request.predefined_voice_id}' not found.",
            )
        audio_prompt_path_for_engine = potential_path
        logger.info(f"Using predefined voice: {request.predefined_voice_id}")

    elif request.voice_mode == "clone":
        if not request.reference_audio_filename:
            raise HTTPException(
                status_code=400,
                detail="Missing 'reference_audio_filename' for 'clone' voice mode.",
            )
        ref_dir = get_reference_audio_path(ensure_absolute=True)
        potential_path = ref_dir / request.reference_audio_filename
        if not potential_path.is_file():
            logger.error(
                f"Reference audio file for cloning not found: {potential_path}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file '{request.reference_audio_filename}' not found.",
            )
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

    output_format_str = (
        request.output_format if request.output_format else get_audio_output_format()
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
        path_for_synth = (
            str(audio_prompt_path_for_engine)
            if audio_prompt_path_for_engine
            else None
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
        headers = {
            "Content-Disposition": f'attachment; filename="{download_filename}"',
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        logger.info(
            f"Streaming /tts ({output_format_str}), {len(text_chunks)} text chunk(s)."
        )
        stream_iter = _stream_encoded_audio_from_pcm(
            text_chunks=text_chunks,
            target_sample_rate=final_output_sample_rate,
            output_format=output_format_str,
            sse=False,
            log_prefix="/tts stream",
            locked_synthesis={
                "audio_prompt_path": path_for_synth,
                "temperature": params.temperature,
                "exaggeration": params.exaggeration,
                "cfg_weight": params.cfg_weight,
                "seed": params.seed,
                "language": params.language,
                "speed_factor": params.speed_factor,
            },
            perf_monitor=perf_monitor,
        )
        return StreamingResponse(
            limit_tts_concurrency_stream(stream_iter),
            media_type=_get_audio_media_type(output_format_str),
            headers=headers,
        )

    path_for_synth = (
        str(audio_prompt_path_for_engine) if audio_prompt_path_for_engine else None
    )
    try:
        all_audio_segments_np, engine_output_sample_rate = (
            await synthesize_text_chunks_async(
                text_chunks,
                path_for_synth,
                params,
                perf_monitor=perf_monitor,
                log_prefix="/tts",
                cancellation_check=raise_if_disconnected,
            )
        )
    except RuntimeError as e:
        logger.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    if not all_audio_segments_np:
        logger.error("No audio segments were successfully generated.")
        raise HTTPException(
            status_code=500, detail="Audio generation resulted in no output."
        )

    try:
        await raise_if_disconnected()
        final_audio_np = _finalize_stitched_tts_audio(
            all_audio_segments_np,
            engine_output_sample_rate,
            perf_monitor=perf_monitor,
            log_prefix="/tts",
        )

        if params.speed_factor != 1.0:
            sped_t = torch.from_numpy(
                final_audio_np.astype(np.float32, copy=False)
            )
            sped_t, engine_output_sample_rate = utils.apply_speed_factor(
                sped_t, engine_output_sample_rate, params.speed_factor
            )
            final_audio_np = sped_t.cpu().numpy().squeeze().astype(np.float32)
            perf_monitor.record("/tts speed_factor applied (post-stitch)")
        await raise_if_disconnected()

    except ValueError as e_concat:
        logger.error(f"Audio concatenation/stitching failed: {e_concat}", exc_info=True)
        for idx, seg in enumerate(all_audio_segments_np):
            logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
        raise HTTPException(
            status_code=500, detail=f"Audio stitching error: {e_concat}"
        )

    await raise_if_disconnected()
    encoded_audio_bytes = utils.encode_audio(
        audio_array=final_audio_np,
        sample_rate=engine_output_sample_rate,
        output_format=output_format_str,
        target_sample_rate=final_output_sample_rate,
    )
    perf_monitor.record(
        f"Final audio encoded to {output_format_str} (target SR: {final_output_sample_rate}Hz from engine SR: {engine_output_sample_rate}Hz)"
    )

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(
            f"Failed to encode final audio to format: {output_format_str} or output is too small ({len(encoded_audio_bytes or b'')} bytes)."
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode audio to {output_format_str} or generated invalid audio.",
        )

    media_type = _get_audio_media_type(output_format_str)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    # Include generation parameters in filename for easy comparison across presets
    param_tag = (
        f"T{params.temperature:.1f}_E{params.exaggeration:.1f}_W{params.cfg_weight:.1f}".replace(
            ".", ""
        )
    )
    suggested_filename_base = f"tts_output_{param_tag}_{timestamp_str}"
    download_filename = utils.sanitize_filename(
        f"{suggested_filename_base}.{output_format_str}"
    )
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

    logger.info(
        f"Successfully generated audio: {download_filename}, {len(encoded_audio_bytes)} bytes, type {media_type}."
    )
    if perf_monitor.enabled:
        logger.info(perf_monitor.report(log_level=logging.INFO))

    # Optional: Save to disk if enabled
    if config_manager.get_bool("audio_output.save_to_disk", False):
        await raise_if_disconnected()
        output_dir = get_output_path(ensure_absolute=True)
        output_file_path = output_dir / download_filename
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "wb") as f:
                f.write(encoded_audio_bytes)
            if not output_file_path.exists() or output_file_path.stat().st_size < 100:
                logger.error(f"File save verification failed for {output_file_path}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save audio file to {output_file_path}",
                )
            logger.info(f"Audio saved to disk: {output_file_path}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to save audio to {output_file_path}: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to save audio file: {e}"
            )

    return StreamingResponse(
        io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers
    )
