"""Audio output policy and encoding helpers for TTS responses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import utils
from config import config_manager


def get_audio_media_type(output_format: str) -> str:
    media_type_map = {
        "mp3": "audio/mpeg",
        "opus": "audio/ogg; codecs=opus",
        "wav": "audio/wav",
        "pcm": "application/octet-stream",
    }
    return media_type_map.get(output_format, f"audio/{output_format}")


class EndpointKind(str, Enum):
    CUSTOM = "custom"
    OPENAI = "openai"


class StreamFormat(str, Enum):
    NONE = "none"
    AUDIO = "audio"
    SSE = "sse"


@dataclass(frozen=True)
class CodecTimingPolicy:
    """Named codec timing policy; keeps lossy guard silence out of route logic."""

    leading_pad_sec: Optional[float]
    trailing_flush_sec: float
    streaming_opus_preroll_sec: float
    min_mp3_pcm_sec: float
    inter_chunk_gap_sec: float

    @classmethod
    def for_request(
        cls, *, output_format: str, stream_format: StreamFormat, chunk_count: int
    ) -> "CodecTimingPolicy":
        def cfg(name: str, fallback: float) -> float:
            return config_manager.get_float(f"audio_output.codec_timing.{name}", fallback)

        short_leading = cfg(
            "lossy_short_leading_pad_sec", utils.LOSSY_ENCODE_SHORT_LEADING_PAD_SEC
        )
        multichunk_leading = cfg(
            "lossy_multichunk_leading_pad_sec", utils.LOSSY_ENCODE_LEADING_PAD_SEC
        )
        trailing_flush = cfg(
            "lossy_trailing_flush_sec", utils.LOSSY_ENCODE_TRAILING_FLUSH_SEC
        )
        lossy = output_format in {"mp3", "opus"}
        if not lossy:
            leading = 0.0
        elif stream_format == StreamFormat.NONE or chunk_count <= 1:
            leading = short_leading
        else:
            leading = multichunk_leading

        return cls(
            leading_pad_sec=leading,
            trailing_flush_sec=(trailing_flush if lossy else 0.0),
            streaming_opus_preroll_sec=(
                cfg("streaming_opus_preroll_sec", utils.LOSSY_ENCODE_LEADING_PAD_SEC)
                if output_format == "opus"
                and stream_format in {StreamFormat.AUDIO, StreamFormat.SSE}
                and chunk_count > 1
                else 0.0
            ),
            min_mp3_pcm_sec=(
                cfg("streaming_mp3_min_pcm_sec", 2.6)
                if output_format == "mp3"
                else 0.0
            ),
            inter_chunk_gap_sec=cfg("streaming_inter_chunk_gap_sec", 0.03),
        )


@dataclass(frozen=True)
class AudioOutputPolicy:
    output_format: str
    target_sample_rate: int
    chunk_count: int
    stream_format: StreamFormat = StreamFormat.NONE

    @property
    def media_type(self) -> str:
        if self.stream_format == StreamFormat.SSE:
            return "text/event-stream"
        return get_audio_media_type(self.output_format)

    @property
    def timing(self) -> CodecTimingPolicy:
        return CodecTimingPolicy.for_request(
            output_format=self.output_format,
            stream_format=self.stream_format,
            chunk_count=self.chunk_count,
        )

    @property
    def is_streaming(self) -> bool:
        return self.stream_format != StreamFormat.NONE

    @property
    def is_compressed_streaming(self) -> bool:
        return self.is_streaming and self.output_format in {"mp3", "opus"}


@dataclass(frozen=True)
class EncodedAudio:
    data: bytes
    media_type: str
    output_format: str


def encode_audio_with_policy(
    audio_array: np.ndarray,
    sample_rate: int,
    policy: AudioOutputPolicy,
) -> EncodedAudio:
    encoded = utils.encode_audio(
        audio_array=audio_array,
        sample_rate=sample_rate,
        output_format=policy.output_format,
        target_sample_rate=policy.target_sample_rate,
        lossy_leading_pad_sec=policy.timing.leading_pad_sec,
    )
    if encoded is None:
        raise RuntimeError(f"Failed to encode audio to {policy.output_format}.")
    return EncodedAudio(
        data=encoded,
        media_type=policy.media_type,
        output_format=policy.output_format,
    )


def streaming_headers(
    policy: AudioOutputPolicy, *, download_filename: Optional[str] = None
) -> Dict[str, str]:
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if download_filename:
        headers["Content-Disposition"] = f'attachment; filename="{download_filename}"'
    if policy.output_format == "pcm":
        headers["X-Sample-Rate"] = str(policy.target_sample_rate)
    elif policy.output_format == "opus":
        headers["X-Stream-Audio-Codec"] = "opus"
        headers["X-Stream-Container"] = "ogg"
    elif policy.output_format == "mp3":
        headers["X-Stream-Audio-Codec"] = "mp3"
        headers["X-Stream-Container"] = "mp3"
    return headers


def write_encoded_audio_if_enabled(
    data: bytes, output_dir: Path, filename: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / filename
    with open(output_file_path, "wb") as f:
        f.write(data)
    if not output_file_path.exists() or output_file_path.stat().st_size < 100:
        raise RuntimeError(f"Failed to save audio file to {output_file_path}")
    return output_file_path
