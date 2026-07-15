from reachy_mini_openclaw.text_streaming import pop_speakable_segments


def test_waits_for_complete_sentence() -> None:
    segments, remaining = pop_speakable_segments("Still thinking")

    assert segments == []
    assert remaining == "Still thinking"


def test_emits_complete_sentences_and_keeps_partial_tail() -> None:
    segments, remaining = pop_speakable_segments("First sentence. Second question? Third is incomplete")

    assert segments == ["First sentence.", "Second question?"]
    assert remaining == "Third is incomplete"


def test_handles_closing_quote() -> None:
    segments, remaining = pop_speakable_segments('He said, "Ready!" Next')

    assert segments == ['He said, "Ready!"']
    assert remaining == "Next"


def test_flushes_tail_at_end_of_stream() -> None:
    segments, remaining = pop_speakable_segments("A short final reply", final=True)

    assert segments == ["A short final reply"]
    assert remaining == ""


def test_splits_long_unpunctuated_output() -> None:
    text = "word " * 50
    segments, remaining = pop_speakable_segments(text)

    assert segments
    assert len(segments[0]) <= 180
    assert remaining
