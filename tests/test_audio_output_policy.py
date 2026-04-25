from __future__ import annotations

import utils
from audio_output import AudioOutputPolicy, StreamFormat


def test_audio_output_policy_short_lossy_single_chunk():
    policy = AudioOutputPolicy(
        output_format="opus",
        target_sample_rate=24000,
        chunk_count=1,
        stream_format=StreamFormat.NONE,
    )

    assert policy.media_type == "audio/ogg; codecs=opus"
    assert policy.timing.leading_pad_sec == utils.LOSSY_ENCODE_SHORT_LEADING_PAD_SEC
    assert policy.timing.streaming_opus_preroll_sec == 0.0


def test_audio_output_policy_multichunk_streaming_opus_preroll():
    policy = AudioOutputPolicy(
        output_format="opus",
        target_sample_rate=24000,
        chunk_count=3,
        stream_format=StreamFormat.AUDIO,
    )

    assert policy.is_compressed_streaming
    assert policy.timing.leading_pad_sec == utils.LOSSY_ENCODE_LEADING_PAD_SEC
    assert policy.timing.streaming_opus_preroll_sec == utils.LOSSY_ENCODE_LEADING_PAD_SEC


def test_audio_output_policy_wav_has_no_lossy_delay():
    policy = AudioOutputPolicy(
        output_format="wav",
        target_sample_rate=24000,
        chunk_count=1,
    )

    assert policy.media_type == "audio/wav"
    assert policy.timing.leading_pad_sec == 0.0
    assert policy.timing.trailing_flush_sec == 0.0
