"""OpenAI-compatible speech API routes."""
import io
import logging
import shutil
import time
import uuid
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

import engine
import utils
from audio_output import AudioOutputPolicy, EndpointKind, StreamFormat, streaming_headers
from audio_pipeline import (
    _stream_encoded_audio_from_pcm,
    async_iter_sse_audio_from_pcm,
    async_iter_locked_pcm_s16le,
)
from config import (
    config_manager,
    get_audio_sample_rate,
    get_predefined_voices_path,
    get_reference_audio_path,
)
from models import OpenAISpeechRequest
from tts_concurrency import limit_tts_concurrency, limit_tts_concurrency_stream
from tts_orchestration import (
    build_text_chunks,
    resolve_synthesis_params_openai,
)
from tts_pipeline import (
    GenerationRequestContext,
    build_locked_synthesis_payload,
    disconnected_check,
    synthesize_buffered_response,
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
    logger.info(
        "TTS request intake: endpoint=/v1/audio/speech streaming_requested=%s "
        "stream_format=%s response_format=%s voice=%s model=%s text_chars=%s",
        request.stream_format is not None,
        request.stream_format or "buffered",
        request.response_format,
        request.voice,
        request.model,
        len(request.input_ or ""),
    )
    try:
        utils.validate_request_text_length(request.input_, field_name="input")
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    reference_audio_path = get_reference_audio_path(ensure_absolute=True)
    audio_prompt_path = None

    try:
        voice_path_predefined = utils.resolve_file_under_directory(
            predefined_voices_path,
            request.voice,
            allowed_suffixes={".wav", ".mp3"},
        )
        audio_prompt_path = voice_path_predefined
    except FileNotFoundError:
        try:
            voice_path_reference = utils.resolve_file_under_directory(
                reference_audio_path,
                request.voice,
                allowed_suffixes={".wav", ".mp3"},
            )
            max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
            is_valid, msg = utils.validate_reference_audio(voice_path_reference, max_dur)
            if not is_valid:
                raise HTTPException(
                    status_code=400, detail=f"Invalid reference audio: {msg}"
                )
            audio_prompt_path = voice_path_reference
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Voice file '{request.voice}' not found."
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if audio_prompt_path is None:
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
        logger.info(
            "TTS request resolved: endpoint=/v1/audio/speech request_id=%s "
            "streaming=%s stream_format=%s response_format=%s chunks=%s "
            "split_text=%s chunk_size=%s",
            perf_monitor.request_id,
            effective_stream_format is not None,
            effective_stream_format or "buffered",
            request.response_format,
            len(text_chunks),
            split_enabled,
            chunk_size_cfg,
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
        stream_format = (
            StreamFormat(effective_stream_format)
            if effective_stream_format is not None
            else StreamFormat.NONE
        )
        output_policy = AudioOutputPolicy(
            output_format=request.response_format,
            target_sample_rate=target_pcm_sr,
            chunk_count=len(text_chunks),
            stream_format=stream_format,
        )
        cancellation_check = disconnected_check(
            http_request,
            "OpenAI speech client disconnected; stopping non-streaming generation.",
        )
        ctx = GenerationRequestContext(
            endpoint=EndpointKind.OPENAI,
            text_chunks=text_chunks,
            audio_prompt_path=audio_prompt_path,
            params=params,
            output_policy=output_policy,
            perf_monitor=perf_monitor,
            log_prefix="OpenAI speech",
            cancellation_check=cancellation_check,
            save_filename=f"openai_tts_{time.strftime('%Y%m%d_%H%M%S')}.{utils.format_to_extension(request.response_format)}",
        )
        stream_headers = streaming_headers(output_policy)
        locked = build_locked_synthesis_payload(ctx)

        if effective_stream_format == "audio":
            if request.response_format == "pcm":
                stream_iter = async_iter_locked_pcm_s16le(
                    text_chunks,
                    target_pcm_sr,
                    locked,
                    perf_monitor=perf_monitor,
                    log_prefix="OpenAI speech pcm",
                    inter_chunk_gap_sec=output_policy.timing.inter_chunk_gap_sec,
                )
            else:
                stream_iter = _stream_encoded_audio_from_pcm(
                    text_chunks=text_chunks,
                    target_sample_rate=target_pcm_sr,
                    output_format=request.response_format,
                    sse=False,
                    log_prefix="OpenAI speech",
                    locked_synthesis=locked,
                    perf_monitor=perf_monitor,
                    timing_policy=output_policy.timing,
                )
            return StreamingResponse(
                limit_tts_concurrency_stream(stream_iter),
                media_type=output_policy.media_type,
                headers=stream_headers,
            )

        if effective_stream_format == "sse":
            if request.response_format == "pcm":
                pcm_iter = async_iter_locked_pcm_s16le(
                    text_chunks,
                    target_pcm_sr,
                    locked,
                    perf_monitor=perf_monitor,
                    log_prefix="OpenAI speech sse pcm",
                    inter_chunk_gap_sec=output_policy.timing.inter_chunk_gap_sec,
                )
                stream_iter = async_iter_sse_audio_from_pcm(pcm_iter)
            else:
                stream_iter = _stream_encoded_audio_from_pcm(
                    text_chunks=text_chunks,
                    target_sample_rate=target_pcm_sr,
                    output_format=request.response_format,
                    sse=True,
                    log_prefix="OpenAI speech",
                    locked_synthesis=locked,
                    perf_monitor=perf_monitor,
                    timing_policy=output_policy.timing,
                )

            return StreamingResponse(
                limit_tts_concurrency_stream(stream_iter),
                media_type=output_policy.media_type,
                headers=stream_headers,
            )

        result = await synthesize_buffered_response(ctx)
        return StreamingResponse(
            io.BytesIO(result.encoded.data), media_type=result.encoded.media_type
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in openai_speech_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
