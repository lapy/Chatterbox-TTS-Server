"""OpenAI-compatible speech API routes."""
import asyncio
import base64
import io
import json
import logging
import shutil
import time
import uuid
import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

import engine
import utils
from audio_pipeline import (
    _finalize_stitched_tts_audio,
    _get_audio_media_type,
    _stream_encoded_audio_from_pcm,
    async_iter_locked_pcm_s16le,
)
from config import (
    config_manager,
    get_audio_sample_rate,
    get_output_path,
    get_predefined_voices_path,
    get_reference_audio_path,
)
from models import OpenAISpeechRequest
from tts_concurrency import limit_tts_concurrency, limit_tts_concurrency_stream
from tts_orchestration import (
    build_text_chunks,
    resolve_synthesis_params_openai,
    synthesize_text_chunks_async,
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/v1/audio/voices", tags=["llama-swap Compatible"])
# llama-swap, koboldcpp, and probably some more use this
async def openai_voices_endpoint(model: str = ""):
    logger.debug("Request for /v1/audio/voices.")
    try:
        return {"status": "ok", "voices": [voice["filename"] for voice in utils.get_predefined_voices()]}
    except Exception as e:
        logger.error(f"Error getting predefined voices for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve predefined voices list."
        )

@router.post("/v1/audio/speech", tags=["OpenAI Compatible"])
@limit_tts_concurrency
async def openai_speech_endpoint(request: OpenAISpeechRequest, http_request: Request):
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    reference_audio_path = get_reference_audio_path(ensure_absolute=True)
    voice_path_predefined = predefined_voices_path / request.voice
    voice_path_reference = reference_audio_path / request.voice

    if voice_path_predefined.is_file():
        audio_prompt_path = voice_path_predefined
    elif voice_path_reference.is_file():
        audio_prompt_path = voice_path_reference
    else:
        raise HTTPException(
            status_code=404, detail=f"Voice file '{request.voice}' not found."
        )

    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    _STREAMING_RESPONSE_FORMATS = frozenset({"pcm", "opus", "mp3"})
    _COMPRESSED_STREAMING_FORMATS = frozenset({"opus", "mp3"})
    if (
        request.stream_format is not None
        and request.response_format not in _STREAMING_RESPONSE_FORMATS
    ):
        raise HTTPException(
            status_code=400,
            detail="When stream_format is set, response_format must be 'pcm', 'opus', "
            "or 'mp3'.",
        )

    try:
        perf_monitor = utils.PerformanceMonitor(
            enabled=config_manager.get_bool(
                "server.enable_performance_monitor", False
            ),
            request_id=str(uuid.uuid4())[:12],
            cuda_sync=config_manager.get_bool("server.performance_cuda_sync", False),
        )
        perf_monitor.record("OpenAI speech request received")

        params = resolve_synthesis_params_openai(request)

        split_enabled = config_manager.get_bool(
            "ui_state.last_split_text_enabled", True
        )
        chunk_size_cfg = config_manager.get_int("ui_state.last_chunk_size", 120)
        text_chunks = build_text_chunks(
            request.input_,
            split_enabled=split_enabled,
            chunk_size=chunk_size_cfg,
            chunk_size_min=50,
            chunk_size_max=1000,
        )
        perf_monitor.record(
            f"OpenAI speech text split into {len(text_chunks)} chunk(s)"
        )

        if not text_chunks:
            raise HTTPException(
                status_code=400, detail="Text processing resulted in no usable chunks."
            )

        effective_stream_format = request.stream_format
        if (
            effective_stream_format is None
            and request.response_format in _COMPRESSED_STREAMING_FORMATS
            and len(text_chunks) > 1
        ):
            effective_stream_format = "audio"
            logger.info(
                "OpenAI speech: stream_format omitted; using incremental audio streaming "
                f"for {len(text_chunks)} chunk(s) ({request.response_format})."
            )

        if (
            effective_stream_format is not None
            and request.response_format in _COMPRESSED_STREAMING_FORMATS
            and shutil.which("ffmpeg") is None
        ):
            raise HTTPException(
                status_code=503,
                detail="ffmpeg is required for streaming opus/mp3 output.",
            )

        logger.info(
            f"OpenAI speech: processing {len(text_chunks)} chunk(s) for "
            f"{len(request.input_)} chars"
            + (
                f", stream_format={effective_stream_format}"
                if effective_stream_format
                else ""
            )
        )

        target_pcm_sr = get_audio_sample_rate()
        stream_headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        if request.response_format == "pcm":
            stream_headers["X-Sample-Rate"] = str(target_pcm_sr)
        elif request.response_format == "opus":
            stream_headers["X-Stream-Audio-Codec"] = "opus"
            stream_headers["X-Stream-Container"] = "ogg"
        elif request.response_format == "mp3":
            stream_headers["X-Stream-Audio-Codec"] = "mp3"
            stream_headers["X-Stream-Container"] = "mp3"

        locked = {
            "audio_prompt_path": str(audio_prompt_path),
            "temperature": params.temperature,
            "exaggeration": params.exaggeration,
            "cfg_weight": params.cfg_weight,
            "seed": params.seed,
            "language": params.language,
            "speed_factor": params.speed_factor,
        }

        async def raise_if_disconnected() -> None:
            if await http_request.is_disconnected():
                logger.info(
                    "OpenAI speech client disconnected; stopping non-streaming generation."
                )
                raise asyncio.CancelledError()

        if effective_stream_format == "audio":

            async def raw_audio_stream():
                async for pcm in async_iter_locked_pcm_s16le(
                    text_chunks,
                    target_pcm_sr,
                    locked,
                    perf_monitor=perf_monitor,
                    log_prefix="OpenAI speech pcm",
                ):
                    yield pcm

            if request.response_format == "pcm":
                stream_iter = raw_audio_stream()
            else:
                stream_iter = _stream_encoded_audio_from_pcm(
                    text_chunks=text_chunks,
                    target_sample_rate=target_pcm_sr,
                    output_format=request.response_format,
                    sse=False,
                    log_prefix="OpenAI speech",
                    locked_synthesis=locked,
                    perf_monitor=perf_monitor,
                )
            return StreamingResponse(
                limit_tts_concurrency_stream(stream_iter),
                media_type=_get_audio_media_type(request.response_format),
                headers=stream_headers,
            )

        if effective_stream_format == "sse":

            async def sse_audio_stream():
                async for pcm in async_iter_locked_pcm_s16le(
                    text_chunks,
                    target_pcm_sr,
                    locked,
                    perf_monitor=perf_monitor,
                    log_prefix="OpenAI speech sse pcm",
                ):
                    b64 = base64.standard_b64encode(pcm).decode("ascii")
                    payload = json.dumps({"type": "speech.audio.delta", "audio": b64})
                    yield f"data: {payload}\n\n".encode("utf-8")
                done = json.dumps({"type": "speech.audio.done"})
                yield f"data: {done}\n\n".encode("utf-8")

            if request.response_format == "pcm":
                stream_iter = sse_audio_stream()
            else:
                stream_iter = _stream_encoded_audio_from_pcm(
                    text_chunks=text_chunks,
                    target_sample_rate=target_pcm_sr,
                    output_format=request.response_format,
                    sse=True,
                    log_prefix="OpenAI speech",
                    locked_synthesis=locked,
                    perf_monitor=perf_monitor,
                )

            return StreamingResponse(
                limit_tts_concurrency_stream(stream_iter),
                media_type="text/event-stream",
                headers=stream_headers,
            )

        all_audio_segments_np, engine_sr = await synthesize_text_chunks_async(
            text_chunks,
            str(audio_prompt_path),
            params,
            perf_monitor=perf_monitor,
            log_prefix="OpenAI speech",
            cancellation_check=raise_if_disconnected,
        )

        await raise_if_disconnected()
        final_audio_np = _finalize_stitched_tts_audio(
            all_audio_segments_np,
            engine_sr,
            perf_monitor=perf_monitor,
            log_prefix="OpenAI speech",
        )

        if params.speed_factor != 1.0:
            sped_t = torch.from_numpy(
                final_audio_np.astype(np.float32, copy=False)
            )
            sped_t, engine_sr = utils.apply_speed_factor(
                sped_t, engine_sr, params.speed_factor
            )
            final_audio_np = sped_t.cpu().numpy().squeeze().astype(np.float32)
            perf_monitor.record("OpenAI speech speed_factor applied (post-stitch)")

        await raise_if_disconnected()
        encoded_audio = utils.encode_audio(
            audio_array=final_audio_np,
            sample_rate=engine_sr,
            output_format=request.response_format,
            target_sample_rate=get_audio_sample_rate(),
        )

        if encoded_audio is None:
            raise HTTPException(status_code=500, detail="Failed to encode audio.")

        perf_monitor.record(
            f"OpenAI speech encoded ({request.response_format}) "
            f"{len(encoded_audio)} bytes"
        )
        if perf_monitor.enabled:
            logger.info(perf_monitor.report(log_level=logging.INFO))

        media_type = _get_audio_media_type(request.response_format)

        if config_manager.get_bool("audio_output.save_to_disk", False):
            await raise_if_disconnected()
            output_dir = get_output_path(ensure_absolute=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            download_filename = f"openai_tts_{timestamp_str}.{request.response_format}"
            output_file_path = output_dir / download_filename
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_file_path, "wb") as f:
                    f.write(encoded_audio)
                if (
                    not output_file_path.exists()
                    or output_file_path.stat().st_size < 100
                ):
                    logger.error(
                        f"File save verification failed for {output_file_path}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to save audio file to {output_file_path}",
                    )
                logger.info(
                    f"OpenAI-compatible audio saved to disk: {output_file_path}"
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    f"Failed to save audio to {output_file_path}: {e}", exc_info=True
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to save audio file: {e}"
                )

        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in openai_speech_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
