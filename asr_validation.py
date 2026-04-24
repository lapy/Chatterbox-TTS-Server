# Optional OpenAI-compatible speech-to-text validation for chunk quality retries.

from __future__ import annotations

import difflib
import io
import logging
import os
import re
from typing import Any, Optional

import numpy as np
import requests
import soundfile as sf

from config import config_manager

logger = logging.getLogger(__name__)


def _text_units(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def normalize_text_for_asr_compare(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy match."""
    s = (text or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def _wav_bytes_from_f32_mono(audio_f32_mono: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    pcm = np.clip(np.asarray(audio_f32_mono, dtype=np.float32), -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    sf.write(buf, pcm_i16, int(sample_rate), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def transcribe_openai_compatible(
    audio_f32_mono: np.ndarray,
    sample_rate: int,
    *,
    base_url: str,
    access_token: str,
    model: str,
    timeout_sec: float,
    language: str = "",
) -> str:
    """
    POST multipart audio to ``{base_url}/audio/transcriptions`` (OpenAI-style).
    ``base_url`` should be the API root including ``/v1``, e.g. ``https://api.openai.com/v1``.
    """
    url = base_url.rstrip("/") + "/audio/transcriptions"
    headers = {"Authorization": f"Bearer {access_token}"}
    wav_bytes = _wav_bytes_from_f32_mono(audio_f32_mono, sample_rate)
    files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
    data: dict[str, Any] = {"model": model}
    if language and language.strip():
        data["language"] = language.strip()

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout_sec)
    resp.raise_for_status()
    ctype = (resp.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        payload = resp.json()
        if isinstance(payload, dict) and "text" in payload:
            return str(payload["text"]).strip()
        logger.warning("ASR JSON response missing 'text' key: %s", payload)
        return ""
    return resp.text.strip()


def transcription_mismatch_reason(
    waveform_mono_f32: np.ndarray,
    sample_rate: int,
    reference_text: str,
) -> Optional[str]:
    """
    If ASR is enabled and configured, transcribe the chunk and compare to ``reference_text``.
    Returns a short reason string when the match is too weak, else ``None``.
    """
    if not config_manager.get_bool("asr.enabled", False):
        return None

    base = (config_manager.get("asr.openai_compatible_base_url") or "").strip()
    token = (os.environ.get("CHATTERBOX_ASR_ACCESS_TOKEN", "") or "").strip()
    if not token:
        token = (config_manager.get("asr.access_token") or "").strip()
    if not base or not token:
        logger.warning(
            "ASR validation is enabled but openai_compatible_base_url or access token "
            "(config or CHATTERBOX_ASR_ACCESS_TOKEN) is missing; skipping ASR check."
        )
        return None

    min_len = max(0, config_manager.get_int("asr.skip_if_text_shorter_than", 12))
    if _text_units(reference_text) < min_len:
        return None

    model = (config_manager.get("asr.model") or "whisper-1").strip() or "whisper-1"
    timeout_sec = float(
        config_manager.get_float("asr.timeout_sec", 120.0) or 120.0
    )
    min_sim = float(config_manager.get_float("asr.min_similarity", 0.82) or 0.82)
    lang_raw = config_manager.get("asr.language")
    language = lang_raw.strip() if isinstance(lang_raw, str) else ""

    try:
        hyp = transcribe_openai_compatible(
            waveform_mono_f32,
            sample_rate,
            base_url=base,
            access_token=token,
            model=model,
            timeout_sec=max(5.0, timeout_sec),
            language=language,
        )
    except requests.RequestException as e:
        logger.warning("ASR request failed: %s", e)
        return "asr_request_failed"

    ref_n = normalize_text_for_asr_compare(reference_text)
    hyp_n = normalize_text_for_asr_compare(hyp)
    if not ref_n:
        return None
    if not hyp_n:
        return "asr_empty_transcript"

    ratio = difflib.SequenceMatcher(None, ref_n, hyp_n).ratio()
    if ratio < min_sim:
        return f"asr_low_similarity_{ratio:.2f}"
    return None
