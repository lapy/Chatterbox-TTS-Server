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
