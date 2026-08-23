"""Sentence segmentation for streaming synthesis."""

from __future__ import annotations

import re
from typing import List

_SENTENCE_BOUNDARY = re.compile(
    r"""
    (?<=[.!?…])            # terminal punctuation just ended
    \s+                    # whitespace gap
    (?=["'“(\[]?[A-Z0-9])  # next sentence starts capital/digit/quote/bracket
    """,
    re.VERBOSE,
)

_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_ABBREV = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|Fig|approx|Inc|Ltd|Co)\.?\s*$",
    re.IGNORECASE,
)


def split_sentences(text: str, max_chars: int = 300) -> List[str]:
    """Split ``text`` into synthesis-friendly sentence chunks.

    Splits on terminal punctuation followed by a capitalized start,
    protecting decimals and common abbreviations. Sentences longer than
    ``max_chars`` are hard-wrapped at the last comma/space before the cap
    so no single chunk dominates memory or time-to-first-audio.

    Returns a list of non-empty stripped chunks (at least one).
    """
    text = " ".join(text.split())
    if not text:
        return []

    protected = _DECIMAL.sub("\x00", text)
    parts = _SENTENCE_BOUNDARY.split(protected)
    parts = [p.replace("\x00", ".") for p in parts]

    merged: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and _ABBREV.search(merged[-1]):
            merged[-1] += " " + part
        else:
            merged.append(part)

    if not merged:
        merged = [text]

    wrapped: List[str] = []
    for chunk in merged:
        wrapped.extend(_wrap_long(chunk, max_chars))
    return wrapped


def _wrap_long(chunk: str, max_chars: int) -> List[str]:
    if len(chunk) <= max_chars:
        return [chunk]
    pieces: List[str] = []
    rest = chunk
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(window.rfind(","), window.rfind(";"), window.rfind(" "))
        if cut < int(max_chars * 0.4):
            cut = max_chars
        pieces.append(rest[: cut + 1].strip())
        rest = rest[cut + 1 :].strip()
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]
