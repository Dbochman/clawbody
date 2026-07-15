"""Helpers for turning incremental agent text into speakable phrases."""

import re

MAX_STREAMING_PHRASE_CHARS = 180
_SENTENCE_BOUNDARY = re.compile(r'^(.+?[.!?](?:["\'’”)]*)?)(?:\s+|$)', re.DOTALL)


def pop_speakable_segments(text: str, *, final: bool = False) -> tuple[list[str], str]:
    """Remove and return complete sentences from a growing text stream."""
    segments: list[str] = []
    remaining = text

    while remaining:
        match = _SENTENCE_BOUNDARY.match(remaining)
        if match:
            segment = match.group(1).strip()
            if segment:
                segments.append(segment)
            remaining = remaining[match.end() :]
            continue

        if len(remaining) >= MAX_STREAMING_PHRASE_CHARS:
            split_at = max(remaining.rfind(mark, 60, MAX_STREAMING_PHRASE_CHARS) for mark in (",", ";", ":", " "))
            if split_at <= 0:
                split_at = MAX_STREAMING_PHRASE_CHARS
            elif remaining[split_at] != " ":
                split_at += 1
            segment = remaining[:split_at].strip()
            if segment:
                segments.append(segment)
            remaining = remaining[split_at:].lstrip()
            continue

        break

    if final and remaining.strip():
        segments.append(remaining.strip())
        remaining = ""
    return segments, remaining
