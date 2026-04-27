# Audio stitching, OpenAI streaming helpers, and synthesis utilities for TTS routes.

from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import shutil
import threading
import time
from contextlib import suppress
from typing import Any, AsyncIterator, Dict, List, Optional

import librosa
import numpy as np

import engine
import utils
from audio_output import CodecTimingPolicy, StreamFormat, get_audio_media_type
from config import config_manager

logger = logging.getLogger(__name__)
_FFMPEG_PATH_CACHE: Optional[str] = None


def _get_ffmpeg_path_cached() -> Optional[str]:
    global _FFMPEG_PATH_CACHE
    if _FFMPEG_PATH_CACHE is None:
        _FFMPEG_PATH_CACHE = shutil.which("ffmpeg")
    return _FFMPEG_PATH_CACHE


def _queue_put_interruptible(
    q: queue.Queue, item: Optional[bytes], stop_event: threading.Event
) -> bool:
    """Put into a bounded cross-thread queue without getting stuck after cancellation."""
    while not stop_event.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


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


def _apply_true_peak_limiter(audio: np.ndarray, peak_limit: float) -> np.ndarray:
    """Scale waveform if needed so absolute peak does not exceed ``peak_limit``."""
    limited = audio.astype(np.float32, copy=False)
    limit = max(0.01, min(1.0, float(peak_limit)))
    peak = float(np.max(np.abs(limited))) if limited.size else 0.0
    if peak > limit:
        limited = limited * (limit / peak)
    return limited


def _apply_gentle_voice_cleanup(
    audio: np.ndarray, sample_rate: int, *, log_prefix: str = "TTS"
) -> np.ndarray:
    """
    Optional conservative post-stitch cleanup for buffered generation.

    This intentionally avoids aggressive denoise to reduce metallic artifacts.
    """
    if not config_manager.get_bool("audio_processing.gentle_cleanup.enabled", False):
        return audio.astype(np.float32, copy=False)

    cleaned = audio.astype(np.float32, copy=False)

    if config_manager.get_bool("audio_processing.gentle_cleanup.enable_highpass", True):
        cutoff = config_manager.get_float("audio_processing.gentle_cleanup.highpass_hz", 80.0)
        cleaned = _remove_dc_offset(cleaned, sample_rate, cutoff_hz=max(15.0, cutoff))

    if config_manager.get_bool(
        "audio_processing.gentle_cleanup.enable_loudness_normalization", True
    ):
        target_lufs = config_manager.get_float(
            "audio_processing.gentle_cleanup.target_lufs", -18.0
        )
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sample_rate)
            integrated = meter.integrated_loudness(cleaned)
            if np.isfinite(integrated):
                cleaned = pyln.normalize.loudness(cleaned, integrated, float(target_lufs)).astype(
                    np.float32
                )
        except Exception as exc:
            logger.warning("%s: loudness normalization skipped: %s", log_prefix, exc)

    if config_manager.get_bool(
        "audio_processing.gentle_cleanup.enable_true_peak_limiter", True
    ):
        peak_limit = config_manager.get_float(
            "audio_processing.gentle_cleanup.true_peak_limit", 0.95
        )
        cleaned = _apply_true_peak_limiter(cleaned, peak_limit)

    return cleaned.astype(np.float32, copy=False)


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

        # Avoid repeated np.concatenate growth for normal speech chunks. Very short
        # chunks fall back to the simple implementation because the overlap size is
        # dynamically clamped in _crossfade_with_overlap.
        if fade_samples > 0 and all(len(chunk) >= fade_samples for chunk in chunks):
            fade_out, fade_in = _generate_equal_power_curves(fade_samples)
            total_len = (
                sum(len(chunk) for chunk in chunks)
                + (len(chunks) - 1) * silence_buffer_samples
                - (len(chunks) - 1) * fade_samples * 2
            )
            result = np.zeros(total_len, dtype=np.float32)
            cursor = len(chunks[0])
            result[:cursor] = chunks[0]
            for chunk in chunks[1:]:
                result[cursor - fade_samples : cursor] *= fade_out
                cursor = cursor + silence_buffer_samples - fade_samples
                result[cursor - fade_samples : cursor] += chunk[:fade_samples] * fade_in
                next_cursor = cursor + len(chunk) - fade_samples
                result[cursor:next_cursor] = chunk[fade_samples:]
                cursor = next_cursor
            final_audio_np = result[:cursor]
        else:
            result = chunks[0]
            for chunk in chunks[1:]:
                silence = np.zeros(silence_buffer_samples, dtype=np.float32)
                result = _crossfade_with_overlap(result, silence, fade_samples)
                result = _crossfade_with_overlap(result, chunk, fade_samples)
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

    final_audio_np = _apply_gentle_voice_cleanup(
        final_audio_np, engine_output_sample_rate, log_prefix=log_prefix
    )
    _perf("Gentle voice cleanup applied")

    if enable_smart_stitching and config_manager.get_bool(
        "audio_processing.enable_silence_trimming", False
    ):
        logger.warning(
            f"{log_prefix}: smart stitching adds sentence pauses, but silence trimming is "
            "enabled — leading/trailing pauses may be removed."
        )

    return final_audio_np


# --- End Audio Stitching Helper Functions ---
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


async def async_iter_locked_pcm_s16le(
    text_chunks: List[str],
    target_sample_rate: int,
    locked_synthesis: Dict[str, Any],
    *,
    perf_monitor: Optional[Any] = None,
    log_prefix: str = "PCM stream",
    inter_chunk_gap_sec: float = 0.03,
) -> AsyncIterator[bytes]:
    """
    Yield s16le mono PCM chunks (with inter-chunk gaps) under a single inference lock.
    Used by OpenAI-compatible raw PCM streaming.
    """
    n = len(text_chunks)
    if n == 0:
        return

    inter_chunk_gap_bytes = np.zeros(
        int(target_sample_rate * inter_chunk_gap_sec), dtype=np.int16
    ).tobytes()
    pcm_queue: queue.Queue = queue.Queue(maxsize=8)
    thread_exc: List[BaseException] = []
    stop_event = threading.Event()

    def _producer() -> None:
        try:
            speed = float(locked_synthesis.get("speed_factor", 1.0))
            for i, (audio_tensor, sr) in enumerate(
                engine.iter_synthesize_under_lock(
                    text_chunks,
                    locked_synthesis.get("audio_prompt_path"),
                    float(locked_synthesis["temperature"]),
                    float(locked_synthesis["exaggeration"]),
                    float(locked_synthesis["cfg_weight"]),
                    int(locked_synthesis["seed"]),
                    str(locked_synthesis["language"]),
                    perf_monitor=perf_monitor,
                    log_prefix=f"{log_prefix} engine",
                )
            ):
                if stop_event.is_set():
                    break
                if speed != 1.0:
                    audio_tensor, sr = utils.apply_speed_factor(
                        audio_tensor, sr, speed
                    )
                if hasattr(audio_tensor, "device") and getattr(audio_tensor.device, "type", "") != "cpu":
                    audio_tensor = audio_tensor.cpu()
                chunk_np = audio_tensor.numpy().squeeze().astype(np.float32)
                wave = _ensure_mono_waveform_1d(chunk_np)
                if not _queue_put_interruptible(
                    pcm_queue,
                    _float32_to_pcm_s16le_bytes(wave, sr, target_sample_rate),
                    stop_event,
                ):
                    break
                if i < n - 1:
                    if not _queue_put_interruptible(
                        pcm_queue, inter_chunk_gap_bytes, stop_event
                    ):
                        break
            _queue_put_interruptible(pcm_queue, None, stop_event)
        except BaseException as exc:
            thread_exc.append(exc)
            _queue_put_interruptible(pcm_queue, None, stop_event)

    threading.Thread(target=_producer, daemon=True).start()
    stream_t0 = time.monotonic()
    first_pcm = True
    try:
        while True:
            item = await asyncio.to_thread(pcm_queue.get)
            if item is None:
                if thread_exc:
                    raise thread_exc[0]
                break
            if first_pcm and perf_monitor is not None:
                perf_monitor.record_duration(
                    f"{log_prefix} streaming TTFB (first PCM ready)",
                    time.monotonic() - stream_t0,
                )
                first_pcm = False
            yield item
    finally:
        stop_event.set()
        with suppress(queue.Full):
            pcm_queue.put_nowait(None)


async def async_iter_sse_audio_from_pcm(
    pcm_stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    async for pcm in pcm_stream:
        b64 = base64.standard_b64encode(pcm).decode("ascii")
        payload = json.dumps({"type": "speech.audio.delta", "audio": b64})
        yield f"data: {payload}\n\n".encode("utf-8")
    done = json.dumps({"type": "speech.audio.done"})
    yield f"data: {done}\n\n".encode("utf-8")


def _get_audio_media_type(output_format: str) -> str:
    return get_audio_media_type(output_format)


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
        # Smaller pages improve time-to-first-byte while retaining a standard Opus profile.
        return base_cmd + [
            "-c:a",
            "libopus",
            "-application",
            "audio",
            "-frame_duration",
            "20",
            "-f",
            "ogg",
            "-page_duration",
            "10000",
            "pipe:1",
        ]

    if output_format == "mp3":
        # -reservoir 0 reduces encoder delay so shorter PCM still produces frames.
        return base_cmd + [
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-reservoir",
            "0",
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
    locked_synthesis: Optional[Dict[str, Any]] = None,
    perf_monitor: Optional[Any] = None,
    timing_policy: Optional[CodecTimingPolicy] = None,
) -> AsyncIterator[bytes]:
    """
    Stream encoded audio via ffmpeg. Use ``locked_synthesis`` for multi-chunk requests
    so reference conditioning is not corrupted between chunks (single inference lock).
    """
    if locked_synthesis is None:
        raise ValueError(f"{log_prefix}: provide locked_synthesis")

    ffmpeg_path = _get_ffmpeg_path_cached()
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
    if timing_policy is None:
        timing_policy = CodecTimingPolicy.for_request(
            output_format=output_format,
            stream_format=StreamFormat.SSE if sse else StreamFormat.AUDIO,
            chunk_count=len(text_chunks),
        )
    # libmp3lame may write zero bytes on stdout if total PCM is shorter than ~2.5s
    # at 24 kHz (even after stdin EOF). Pad with trailing silence so short lines work.
    min_mp3_pcm_samples = int(target_sample_rate * timing_policy.min_mp3_pcm_sec)
    # Some Ogg Opus streaming clients/bridges do not make the decoder ready until a few
    # pages have passed. Feed enough initial silence that any startup loss is silence.
    min_streaming_opus_preroll_samples = int(
        target_sample_rate * timing_policy.streaming_opus_preroll_sec
    )
    writer_error: Optional[BaseException] = None

    async def _collect_stderr() -> None:
        while True:
            data = await proc.stderr.read(4096)
            if not data:
                break
            stderr_chunks.append(data)

    async def _writer() -> None:
        nonlocal writer_error
        pcm_samples_written = 0
        stream_t0 = time.monotonic()
        first_pcm_logged = False
        cancelled = False
        # Coalesce small PCM items before drain() to reduce event-loop/syscall overhead.
        flush_threshold_bytes = 32 * 1024
        pending_write = bytearray()

        async def _flush_pending() -> None:
            if not pending_write:
                return
            proc.stdin.write(bytes(pending_write))
            pending_write.clear()
            await proc.stdin.drain()

        try:
            lead_pad_samples = 0
            if output_format in ("mp3", "opus"):
                lead_pad_samples = utils.lossy_encode_leading_silence_sample_count(
                    target_sample_rate,
                    timing_policy.leading_pad_sec,
                )
                if output_format == "opus":
                    lead_pad_samples = max(
                        lead_pad_samples, min_streaming_opus_preroll_samples
                    )
            if lead_pad_samples > 0:
                proc.stdin.write(b"\x00\x00" * lead_pad_samples)
                await proc.stdin.drain()
                pcm_samples_written += lead_pad_samples

            async for item in async_iter_locked_pcm_s16le(
                text_chunks,
                target_sample_rate,
                locked_synthesis,
                perf_monitor=perf_monitor,
                log_prefix=f"{log_prefix} pcm",
                inter_chunk_gap_sec=timing_policy.inter_chunk_gap_sec,
            ):
                if not first_pcm_logged and perf_monitor is not None:
                    perf_monitor.record_duration(
                        f"{log_prefix} streaming TTFB (first PCM ready)",
                        time.monotonic() - stream_t0,
                    )
                    first_pcm_logged = True
                pending_write.extend(item)
                if len(pending_write) >= flush_threshold_bytes:
                    await _flush_pending()
                pcm_samples_written += len(item) // 2

            await _flush_pending()

            # Trailing PCM silence so libmp3lame / libopus finish the last frame(s) instead
            # of cutting off speech (common when stdin closes immediately after content).
            trail_flush = int(target_sample_rate * timing_policy.trailing_flush_sec)
            if trail_flush > 0:
                proc.stdin.write(b"\x00\x00" * trail_flush)
                await proc.stdin.drain()
                pcm_samples_written += trail_flush

            if output_format == "mp3":
                pad_samples = min_mp3_pcm_samples - pcm_samples_written
                if pad_samples > 0:
                    proc.stdin.write(b"\x00\x00" * pad_samples)
                    await proc.stdin.drain()
        except asyncio.CancelledError:
            cancelled = True
            raise
        except BaseException as exc:
            writer_error = exc
        finally:
            if not cancelled:
                with suppress(Exception):
                    proc.stdin.close()

    stderr_task = asyncio.create_task(_collect_stderr())
    writer_task = asyncio.create_task(_writer())

    stream_out_t0 = time.monotonic()
    first_encoded_byte = True
    try:
        while True:
            stdout_chunk = await proc.stdout.read(stream_chunk_size)
            if not stdout_chunk:
                break

            if first_encoded_byte and stdout_chunk:
                first_encoded_byte = False
                tfb = time.monotonic() - stream_out_t0
                if perf_monitor is not None and getattr(perf_monitor, "enabled", False):
                    perf_monitor.record_duration(
                        f"{log_prefix} streaming TTFB (first encoded byte)", tfb
                    )
                logger.info(
                    "%s: first encoded audio byte(s) on wire after %.3fs (ffmpeg stdout)",
                    log_prefix,
                    tfb,
                )

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
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                with suppress(ProcessLookupError):
                    proc.kill()
                with suppress(Exception):
                    await proc.wait()
