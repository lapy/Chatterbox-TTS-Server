"""OpenAI-compatible speech API routes."""
import base64
import io
import json
import logging
import shutil
import time
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

import engine
import utils
from audio_pipeline import (
    _finalize_stitched_tts_audio,
    _float32_to_pcm_s16le_bytes,
    _get_audio_media_type,
    _stream_encoded_audio_from_pcm,
)
from config import (
    config_manager,
    get_audio_sample_rate,
    get_output_path,
    get_predefined_voices_path,
    get_reference_audio_path,
)
from models import OpenAISpeechRequest
from tts_orchestration import (
    build_text_chunks,
    make_synthesize_chunk_partial,
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
async def openai_speech_endpoint(request: OpenAISpeechRequest):
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
                detail="ffmpeg is required for compressed streaming output (opus/mp3).",
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

        synthesize_fn = make_synthesize_chunk_partial(
            str(audio_prompt_path), params
        )

        if effective_stream_format == "audio":

            async def raw_audio_stream():
                n = len(text_chunks)
                for i, chunk_text in enumerate(text_chunks):
                    wave, sr = await run_in_threadpool(synthesize_fn, chunk_text)
                    pcm = _float32_to_pcm_s16le_bytes(wave, sr, target_pcm_sr)
                    yield pcm
                    if i < n - 1 and n > 1:
                        gap = int(target_pcm_sr * 0.03)
                        yield np.zeros(gap, dtype=np.int16).tobytes()

            if request.response_format == "pcm":
                stream_iter = raw_audio_stream()
            else:
                stream_iter = _stream_encoded_audio_from_pcm(
                    text_chunks=text_chunks,
                    target_sample_rate=target_pcm_sr,
                    output_format=request.response_format,
                    sse=False,
                    log_prefix="OpenAI speech",
                    synthesize_chunk_sync=synthesize_fn,
                )
            return StreamingResponse(
                stream_iter,
                media_type=_get_audio_media_type(request.response_format),
                headers=stream_headers,
            )

        if effective_stream_format == "sse":

            async def sse_audio_stream():
                n = len(text_chunks)
                for i, chunk_text in enumerate(text_chunks):
                    wave, sr = await run_in_threadpool(synthesize_fn, chunk_text)
                    if request.response_format == "pcm":
                        pcm = _float32_to_pcm_s16le_bytes(wave, sr, target_pcm_sr)
                        b64 = base64.standard_b64encode(pcm).decode("ascii")
                        if i < n - 1 and n > 1:
                            gap = int(target_pcm_sr * 0.03)
                            gap_pcm = np.zeros(gap, dtype=np.int16).tobytes()
                    payload = json.dumps({"type": "speech.audio.delta", "audio": b64})
                    yield f"data: {payload}\n\n".encode("utf-8")
                    if request.response_format == "pcm" and i < n - 1 and n > 1:
                        gap_b64 = base64.standard_b64encode(gap_pcm).decode("ascii")
                        gap_payload = json.dumps(
                            {"type": "speech.audio.delta", "audio": gap_b64}
                        )
                        yield f"data: {gap_payload}\n\n".encode("utf-8")
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
                    synthesize_chunk_sync=synthesize_fn,
                )

            return StreamingResponse(
                stream_iter,
                media_type="text/event-stream",
                headers=stream_headers,
            )

        all_audio_segments_np, engine_sr = await synthesize_text_chunks_async(
            text_chunks,
            str(audio_prompt_path),
            params,
            perf_monitor=None,
            log_prefix="OpenAI speech",
        )

        final_audio_np = _finalize_stitched_tts_audio(
            all_audio_segments_np,
            engine_sr,
            perf_monitor=None,
            log_prefix="OpenAI speech",
        )

        encoded_audio = utils.encode_audio(
            audio_array=final_audio_np,
            sample_rate=engine_sr,
            output_format=request.response_format,
            target_sample_rate=get_audio_sample_rate(),
        )

        if encoded_audio is None:
            raise HTTPException(status_code=500, detail="Failed to encode audio.")

        media_type = _get_audio_media_type(request.response_format)

        if config_manager.get_bool("audio_output.save_to_disk", False):
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
