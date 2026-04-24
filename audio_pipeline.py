# Audio stitching, OpenAI streaming helpers, and synthesis utilities for TTS routes.

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shutil
from contextlib import suppress
from typing import AsyncIterator, Callable, List, Optional, Tuple

import librosa
import numpy as np
from starlette.concurrency import run_in_threadpool

import engine
import utils
from config import config_manager

logger = logging.getLogger(__name__)


def _generate_equal_power_curves(n_samples: int):
    """
    Generate equal-power crossfade curves using cos²/sin² functions.
    These curves maintain perceptually constant loudness during transitions.

    Args:
        n_samples: Number of samples in the fade region

    Returns:
        Tuple of (fade_out, fade_in) numpy arrays
    """
    t = np.linspace(0, np.pi / 2, n_samples, dtype=np.float32)
    fade_out = np.cos(t) ** 2  # 1 → 0
    fade_in = np.sin(t) ** 2  # 0 → 1
    return fade_out, fade_in


def _crossfade_with_overlap(
    chunk_a: np.ndarray, chunk_b: np.ndarray, fade_samples: int
) -> np.ndarray:
    """
    Perform true crossfade by overlapping and summing audio regions.

    This creates a seamless transition by:
    1. Taking the tail of chunk_a and head of chunk_b
    2. Applying equal-power fade curves
    3. Summing the overlapped regions

    Result length = len(chunk_a) + len(chunk_b) - fade_samples

    Args:
        chunk_a: First audio chunk (numpy float32 array)
        chunk_b: Second audio chunk (numpy float32 array)
        fade_samples: Number of samples to overlap

    Returns:
        Crossfaded audio as numpy float32 array
    """
    # Handle edge cases
    fade_samples = min(fade_samples, len(chunk_a), len(chunk_b))
    if fade_samples <= 0:
        return np.concatenate([chunk_a, chunk_b])

    fade_out, fade_in = _generate_equal_power_curves(fade_samples)

    # Extract overlap regions
    a_tail = chunk_a[-fade_samples:]
    b_head = chunk_b[:fade_samples]

    # Crossfade: weighted sum of overlapping regions
    crossfaded_region = (a_tail * fade_out) + (b_head * fade_in)

    # Assemble: [chunk_a without tail] + [crossfaded region] + [chunk_b without head]
    return np.concatenate(
        [chunk_a[:-fade_samples], crossfaded_region, chunk_b[fade_samples:]]
    )


def _apply_edge_fades(
    chunk: np.ndarray, fade_samples: int, fade_in: bool = True, fade_out: bool = True
) -> np.ndarray:
    """
    Apply minimal linear edge fades for click protection.

    This is used in fallback mode when full crossfading is disabled.
    Linear fades are acceptable for ultra-short safety fades (2-3ms).

    Args:
        chunk: Audio chunk (numpy array)
        fade_samples: Number of samples to fade
        fade_in: Whether to apply fade-in at start
        fade_out: Whether to apply fade-out at end

    Returns:
        Audio chunk with edge fades applied (numpy float32 array)
    """
    # Skip if chunk is too short for fading
    if len(chunk) < fade_samples * 2:
        return chunk.astype(np.float32, copy=False)

    result = chunk.astype(np.float32, copy=True)

    if fade_in:
        result[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)
    if fade_out:
        result[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)

    return result


def _remove_dc_offset(
    audio: np.ndarray, sample_rate: int, cutoff_hz: float = 15.0
) -> np.ndarray:
    """
    Remove DC offset using a high-pass Butterworth filter.

    DC offset can cause low-frequency thumps when concatenating audio chunks.
    This applies a 2nd-order high-pass filter at the specified cutoff frequency.

    Args:
        audio: Audio data (numpy array)
        sample_rate: Sample rate in Hz
        cutoff_hz: High-pass filter cutoff frequency (default 15 Hz)

    Returns:
        Audio with DC offset removed (numpy float32 array)

    Note:
        Requires scipy. If scipy is not available, returns audio unchanged
        with a warning logged.
    """
    try:
        from scipy.signal import butter, filtfilt

        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_hz / nyquist

        # 2nd-order Butterworth high-pass filter
        b, a = butter(2, normalized_cutoff, btype="high")

        # Zero-phase filtering (no phase distortion)
        return filtfilt(b, a, audio).astype(np.float32)

    except ImportError:
        logger.warning(
            "scipy not available for DC offset removal. "
            "Install scipy to enable this feature: pip install scipy"
        )
        return audio.astype(np.float32, copy=False)
    except Exception as e:
        logger.error(f"DC offset removal failed: {e}")
        return audio.astype(np.float32, copy=False)


def _ensure_mono_waveform_1d(audio: np.ndarray) -> np.ndarray:
    """
    Normalize engine output to 1D float32 mono.

    Crossfading uses len() as the sample count; a (1, n) or (2, n) array would
    make that tiny and produce unintelligible garbage.
    """
    a = np.asarray(audio, dtype=np.float32)
    if a.size == 0:
        return a
    if a.ndim == 1:
        return np.ascontiguousarray(a, dtype=np.float32)
    if a.ndim == 2:
        r, c = a.shape
        if r == 1:
            a = a[0]
        elif c == 1:
            a = a[:, 0]
        elif r <= 8 and c > r:
            a = a[0]
        elif c <= 8 and r > c:
            a = a[:, 0]
        else:
            logger.warning(
                "Ambiguous 2D waveform shape %s; using first row as mono.", a.shape
            )
            a = a[0]
    else:
        a = np.squeeze(a)
        if a.ndim != 1:
            logger.warning(
                "Unexpected waveform shape after squeeze %s; flattening.", a.shape
            )
            a = a.reshape(-1)
    return np.ascontiguousarray(a, dtype=np.float32)


def _finalize_stitched_tts_audio(
    all_audio_segments_np: List[np.ndarray],
    engine_output_sample_rate: int,
    *,
    perf_monitor: Optional[utils.PerformanceMonitor] = None,
    log_prefix: str = "TTS",
) -> np.ndarray:
    """
    Stitch multi-chunk engine output, normalize, and apply optional global post-processing.
    Used by /tts and OpenAI-compatible speech so behavior matches config (audio_processing.*).
    """
    all_audio_segments_np = [
        _ensure_mono_waveform_1d(seg) for seg in all_audio_segments_np
    ]

    SENTENCE_PAUSE_MS = 200
    CROSSFADE_MS = 20
    SAFETY_FADE_MS = 3
    ENABLE_DC_REMOVAL = False
    DC_HIGHPASS_HZ = 15
    PEAK_NORMALIZE_THRESHOLD = 0.99
    PEAK_NORMALIZE_TARGET = 0.95

    enable_smart_stitching = config_manager.get_bool(
        "audio_processing.enable_crossfade", True
    )

    def _perf(label: str) -> None:
        if perf_monitor is not None:
            perf_monitor.record(label)

    if not engine_output_sample_rate or engine_output_sample_rate <= 0:
        logger.error(
            f"{log_prefix}: invalid sample rate {engine_output_sample_rate}, "
            "falling back to raw concatenation"
        )
        final_audio_np = (
            np.concatenate(all_audio_segments_np)
            if len(all_audio_segments_np) > 1
            else all_audio_segments_np[0]
        )

    elif len(all_audio_segments_np) == 1:
        final_audio_np = all_audio_segments_np[0]
        logger.info(f"{log_prefix}: single audio chunk — no stitching")

    elif enable_smart_stitching:
        fade_samples = int(CROSSFADE_MS / 1000 * engine_output_sample_rate)
        desired_silence_samples = int(
            SENTENCE_PAUSE_MS / 1000 * engine_output_sample_rate
        )
        silence_buffer_samples = desired_silence_samples + (fade_samples * 2)

        chunks = []
        for chunk in all_audio_segments_np:
            processed = chunk.astype(np.float32, copy=True)
            if ENABLE_DC_REMOVAL:
                processed = _remove_dc_offset(
                    processed, engine_output_sample_rate, DC_HIGHPASS_HZ
                )
            chunks.append(processed)

        result = chunks[0]
        for i in range(1, len(chunks)):
            silence = np.zeros(silence_buffer_samples, dtype=np.float32)
            result = _crossfade_with_overlap(result, silence, fade_samples)
            result = _crossfade_with_overlap(result, chunks[i], fade_samples)

        final_audio_np = result
        logger.info(
            f"{log_prefix}: smart stitching — {len(chunks)} chunks, "
            f"{CROSSFADE_MS}ms crossfades, {SENTENCE_PAUSE_MS}ms pauses"
        )

    else:
        fade_samples = int(SAFETY_FADE_MS / 1000 * engine_output_sample_rate)
        num_chunks = len(all_audio_segments_np)
        processed_chunks = []
        for i, chunk in enumerate(all_audio_segments_np):
            is_first = i == 0
            is_last = i == num_chunks - 1
            processed = _apply_edge_fades(
                chunk,
                fade_samples,
                fade_in=(not is_first),
                fade_out=(not is_last),
            )
            processed_chunks.append(processed)

        final_audio_np = np.concatenate(processed_chunks)
        logger.info(
            f"{log_prefix}: safety edge fades — {num_chunks} chunks, "
            f"{SAFETY_FADE_MS}ms linear fades"
        )

    final_audio_np = final_audio_np.astype(np.float32, copy=False)

    peak_amplitude = np.abs(final_audio_np).max()
    if peak_amplitude > PEAK_NORMALIZE_THRESHOLD:
        final_audio_np = final_audio_np * (PEAK_NORMALIZE_TARGET / peak_amplitude)
        logger.warning(
            f"{log_prefix}: normalized to prevent clipping (peak was {peak_amplitude:.3f})"
        )

    _perf("Audio chunks stitched")

    if config_manager.get_bool("audio_processing.enable_silence_trimming", False):
        final_audio_np = utils.trim_lead_trail_silence(
            final_audio_np, engine_output_sample_rate
        )
        _perf("Global silence trim applied")

    if config_manager.get_bool(
        "audio_processing.enable_internal_silence_fix", False
    ):
        final_audio_np = utils.fix_internal_silence(
            final_audio_np, engine_output_sample_rate
        )
        _perf("Global internal silence fix applied")

    if (
        config_manager.get_bool("audio_processing.enable_unvoiced_removal", False)
        and utils.PARSELMOUTH_AVAILABLE
    ):
        final_audio_np = utils.remove_long_unvoiced_segments(
            final_audio_np, engine_output_sample_rate
        )
        _perf("Global unvoiced removal applied")

    if enable_smart_stitching and config_manager.get_bool(
        "audio_processing.enable_silence_trimming", False
    ):
        logger.warning(
            f"{log_prefix}: smart stitching adds sentence pauses, but silence trimming is "
            "enabled — leading/trailing pauses may be removed."
        )

    return final_audio_np


# --- End Audio Stitching Helper Functions ---
def _synthesize_tts_chunk_sync(
    chunk_text: str,
    audio_prompt_path_str: Optional[str],
    *,
    temperature: float,
    exaggeration: float,
    cfg_weight: float,
    seed: int,
    language: str,
    speed_factor: float,
) -> Tuple[np.ndarray, int]:
    """Return mono float32 waveform in [-1, 1] and engine sample rate (UI / /tts streaming)."""
    audio_tensor, sr = engine.synthesize(
        text=chunk_text,
        audio_prompt_path=audio_prompt_path_str,
        temperature=temperature,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        seed=seed,
        language=language,
    )
    if audio_tensor is None or sr is None:
        raise RuntimeError("TTS engine failed to synthesize audio for a text chunk.")
    if speed_factor != 1.0:
        audio_tensor, _ = utils.apply_speed_factor(audio_tensor, sr, speed_factor)
    chunk_np = audio_tensor.cpu().numpy().squeeze().astype(np.float32)
    return _ensure_mono_waveform_1d(chunk_np), sr


def _float32_to_pcm_s16le_bytes(
    wave_f32: np.ndarray, orig_sr: int, target_sr: int
) -> bytes:
    w = _ensure_mono_waveform_1d(np.asarray(wave_f32, dtype=np.float32))
    if target_sr != orig_sr:
        if not utils.LIBROSA_AVAILABLE:
            raise RuntimeError(
                "librosa is required to resample streaming PCM to the configured output sample rate."
            )
        w = librosa.resample(y=w, orig_sr=orig_sr, target_sr=target_sr)
    w = np.clip(w, -1.0, 1.0)
    return (w * 32767.0).astype(np.int16).tobytes()


def _get_audio_media_type(output_format: str) -> str:
    media_type_map = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "wav": "audio/wav",
        "pcm": "application/octet-stream",
    }
    return media_type_map.get(output_format, f"audio/{output_format}")


def _get_ffmpeg_stream_command(
    ffmpeg_path: str, *, input_sample_rate: int, output_format: str
) -> List[str]:
    base_cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-fflags",
        "nobuffer",
        "-f",
        "s16le",
        "-ar",
        str(input_sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-map_metadata",
        "-1",
        "-flush_packets",
        "1",
    ]

    if output_format == "opus":
        # Ogg muxer default page_duration is ~1s (ffmpeg libavformat), which delays the
        # first bytes on stdout until roughly that much encoded timeline is muxed.
        # Smaller pages + low-delay libopus settings improve time-to-first-byte for streaming.
        return base_cmd + [
            "-c:a",
            "libopus",
            "-application",
            "lowdelay",
            "-frame_duration",
            "20",
            "-f",
            "ogg",
            "-page_duration",
            "50000",
            "pipe:1",
        ]

    if output_format == "mp3":
        return base_cmd + [
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-write_xing",
            "0",
            "-id3v2_version",
            "0",
            "-f",
            "mp3",
            "pipe:1",
        ]

    raise ValueError(f"Unsupported compressed streaming format: {output_format}")


async def _stream_encoded_audio_from_pcm(
    *,
    text_chunks: List[str],
    target_sample_rate: int,
    output_format: str,
    sse: bool,
    log_prefix: str,
    synthesize_chunk_sync: Callable[[str], Tuple[np.ndarray, int]],
) -> AsyncIterator[bytes]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            f"{log_prefix}: ffmpeg is required for streaming '{output_format}' output."
        )

    command = _get_ffmpeg_stream_command(
        ffmpeg_path,
        input_sample_rate=target_sample_rate,
        output_format=output_format,
    )
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if proc.stdin is None or proc.stdout is None or proc.stderr is None:
        raise RuntimeError(f"{log_prefix}: failed to open ffmpeg streaming pipes.")

    stderr_chunks: List[bytes] = []
    stream_chunk_size = 8192
    inter_chunk_gap_bytes = np.zeros(
        int(target_sample_rate * 0.03), dtype=np.int16
    ).tobytes()
    writer_error: Optional[BaseException] = None

    async def _collect_stderr() -> None:
        while True:
            data = await proc.stderr.read(4096)
            if not data:
                break
            stderr_chunks.append(data)

    async def _writer() -> None:
        nonlocal writer_error
        try:
            for i, chunk_text in enumerate(text_chunks):
                wave, sr = await run_in_threadpool(
                    synthesize_chunk_sync,
                    chunk_text,
                )
                pcm = _float32_to_pcm_s16le_bytes(wave, sr, target_sample_rate)
                proc.stdin.write(pcm)
                await proc.stdin.drain()

                if i < len(text_chunks) - 1:
                    proc.stdin.write(inter_chunk_gap_bytes)
                    await proc.stdin.drain()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            writer_error = exc
        finally:
            with suppress(Exception):
                proc.stdin.close()
            with suppress(Exception):
                await proc.stdin.wait_closed()

    stderr_task = asyncio.create_task(_collect_stderr())
    writer_task = asyncio.create_task(_writer())

    try:
        while True:
            stdout_chunk = await proc.stdout.read(stream_chunk_size)
            if not stdout_chunk:
                break

            if sse:
                payload = json.dumps(
                    {
                        "type": "speech.audio.delta",
                        "audio": base64.standard_b64encode(stdout_chunk).decode(
                            "ascii"
                        ),
                    }
                )
                yield f"data: {payload}\n\n".encode("utf-8")
            else:
                yield stdout_chunk

        await writer_task
        await proc.wait()
        await stderr_task

        if writer_error is not None:
            raise RuntimeError(
                f"{log_prefix}: upstream PCM generation failed: {writer_error}"
            ) from writer_error

        if proc.returncode != 0:
            stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{log_prefix}: ffmpeg encoder exited with code {proc.returncode}: "
                f"{stderr_text.strip() or 'no stderr output'}"
            )

        if sse:
            done_payload = json.dumps({"type": "speech.audio.done"})
            yield f"data: {done_payload}\n\n".encode("utf-8")

    finally:
        if not writer_task.done():
            writer_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await writer_task
        if not stderr_task.done():
            stderr_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await stderr_task
        if proc.returncode is None:
            with suppress(ProcessLookupError):
                proc.kill()
            with suppress(Exception):
                await proc.wait()
