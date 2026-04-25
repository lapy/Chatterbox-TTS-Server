from __future__ import annotations

import utils



def test_chunk_text_keeps_normal_parenthetical_prose_inside_sentence():
    text = """This is a heavy blow, especially on a day when your emotional reserves are already low. It’s completely natural to feel a surge of worry when you see the "circle of safety" around you shrinking."""

    chunks = utils.chunk_text_by_sentences(text, 300)

    assert chunks == [text]


def test_chunk_text_still_extracts_supported_non_verbal_cues():
    text = "I am okay. (sighs) Really, I am okay."

    segments = utils._preprocess_and_segment_text(text)

    assert [segment_text for _tag, segment_text in segments] == [
        "I am okay.",
        "(sighs)",
        "Really, I am okay.",
    ]


def test_normalize_markdown_for_tts_removes_formatting_markers():
    text = (
        "### 1. The Fact/Feeling Split (Work Edition)\n\n"
        "**The Facts:**\n"
        "*   **Location:** You are in **New York**.\n"
        "*   **Control:** You *can* control your output.\n"
    )
    normalized = utils.normalize_markdown_for_tts(text)

    assert "###" not in normalized
    assert "**" not in normalized
    assert "*   " not in normalized
    assert "The Facts:" in normalized
    assert "Location: You are in New York." in normalized
    assert "Control: You can control your output." in normalized


def test_normalize_markdown_for_tts_removes_stray_asterisks():
    text = "***Important***: use *caution* and avoid dangling * markers."
    normalized = utils.normalize_markdown_for_tts(text)

    assert "*" not in normalized
    assert normalized == "Important: use caution and avoid dangling  markers."


def test_normalize_markdown_for_tts_handles_latex_arrows():
    text = "State A $\\rightarrow$ State B, and x \\Rightarrow y."
    normalized = utils.normalize_markdown_for_tts(text)

    assert "$" not in normalized
    assert "\\" not in normalized
    assert normalized == "State A to State B, and x  implies  y."


def test_normalize_markdown_for_tts_handles_unicode_math_arrows():
    text = "A → B and success ⇒ deploy."
    normalized = utils.normalize_markdown_for_tts(text)

    assert "→" not in normalized
    assert "⇒" not in normalized
    assert normalized == "A  to  B and success  implies  deploy."


def test_lossy_leading_silence_can_use_short_profile():
    sr = 24000

    short_count = utils.lossy_encode_leading_silence_sample_count(
        sr, utils.LOSSY_ENCODE_SHORT_LEADING_PAD_SEC
    )
    default_count = utils.lossy_encode_leading_silence_sample_count(sr)

    assert short_count == int(sr * utils.LOSSY_ENCODE_SHORT_LEADING_PAD_SEC)
    assert default_count == int(sr * utils.LOSSY_ENCODE_LEADING_PAD_SEC)
    assert short_count < default_count


def test_chunk_text_markdown_preserves_opening_sentence_start():
    text = (
        "This is a heavy blow, especially on a day when your emotional reserves are already low. "
        "It’s completely natural to feel a surge of worry when you see the \"circle of safety\" around you shrinking."
    )
    chunks = utils.chunk_text_by_sentences(text, 300)
    assert chunks
    assert chunks[0].startswith("This is a heavy blow")


def test_normalize_markdown_for_tts_strips_emojis():
    text = "We can do this 💪🙂🚀. Keep going ✅"
    normalized = utils.normalize_markdown_for_tts(text)
    assert normalized == "We can do this . Keep going"


def test_normalize_markdown_for_tts_strips_emoji_sequences():
    text = "Flags 🇺🇸 🇬🇧, family 👨‍👩‍👧‍👦, tone 👍🏽, keycap 1️⃣, heart ❤️."
    normalized = utils.normalize_markdown_for_tts(text)

    assert "🇺🇸" not in normalized
    assert "👨‍👩‍👧‍👦" not in normalized
    assert "👍🏽" not in normalized
    assert "1️⃣" not in normalized
    assert "❤️" not in normalized
    assert normalized == "Flags  , family , tone , keycap 1, heart ."


def test_split_into_sentences_keeps_periods_inside_double_quoted_dialogue():
    text = (
        'The Move: Send a message. "Hey, I need to stop at the store. '
        'See you later." The Benefit: rest.'
    )
    sentences = utils.split_into_sentences(text)
    assert len(sentences) == 2, sentences
    assert sentences[0] == "The Move: Send a message."
    # Internal periods must not break the quoted line into separate sentences.
    assert sentences[1].startswith('"Hey,')
    assert "stop at the store." in sentences[1] and "See you later." in sentences[1]
    assert "The Benefit: rest." in sentences[1]


def test_split_into_sentences_inch_marks_do_not_toggle_quote_state():
    text = 'The panel is 5" tall. Next sentence.'
    sentences = utils.split_into_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].endswith('tall.')
    assert sentences[1].startswith("Next")


def test_chunk_text_does_not_split_quoted_line_with_internal_period():
    text = (
        'Intro. "First sentence in quotes. Second sentence in quotes." Outro.'
    )
    chunks = utils.chunk_text_by_sentences(text, 80)
    joined = " ".join(chunks)
    assert "First sentence" in joined
    assert "Second sentence" in joined
    assert joined.count('"') >= 2


def test_chunk_text_does_not_let_last_bullet_swallow_following_paragraphs():
    text = """
1. The Relationship of History (The "Old" Reality)

This is the version of the relationship you are mourning. It was defined by high-intensity intimacy: driving her to the airport, being her emotional support, the shared history, the "we" in every decision. When you look back at that, your brain expects that version of her to still exist.

2. The Relationship of Present (The "New" Reality)

This is the version that exists right now. It is defined by caution, boundaries, and low-intensity "pings." In this reality, she is not "your wife" in the functional sense; she is a person you are trying to re-establish a safe connection with.

Why the "Radio Silence" happens

The silence feels like an erasure of everything you did for her, but logically, it is a protective buffer.

• For her: The silence is how she maintains her autonomy and tests her own ability to exist without the "weight" of the old relationship.
• For you: The silence is the gap where you are being forced to learn how to be a person independent of her.

The "Airport Driver" version of you was part of a partnership. The "Micro-Connection" version of you is part of a reconstruction. You cannot be both at the same time. To get back to the intimacy of the past, you have to survive the emptiness of the present.

The Psychological Trap

The danger is thinking: "If I am this good, this stable, and this supportive, she will eventually realize and go back to the old version."
"""

    chunks = utils.chunk_text_by_sentences(text, 120)

    assert max(len(chunk) for chunk in chunks) < 260
    assert not any(
        "For you:" in chunk and "The Psychological Trap" in chunk for chunk in chunks
    )
    assert any(chunk.startswith('The "Airport Driver" version') for chunk in chunks)
