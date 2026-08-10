"""Turn word-level timestamps into complete, non-overlapping sentences."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .models import Sentence, WordToken


_HARD_PUNCTUATION = frozenset("。！？!?；;")
_WEAK_PUNCTUATION = frozenset("，,、：:")
_CLOSING_MARKS = "\"'”’）)]】》」』"
_COMMON_ABBREVIATIONS = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
    }
)


def _terminal_character(text: str) -> str:
    candidate = text.rstrip().rstrip(_CLOSING_MARKS).rstrip()
    return candidate[-1] if candidate else ""


def _is_hard_boundary(token: WordToken) -> bool:
    char = _terminal_character(token.text)
    if char in _HARD_PUNCTUATION:
        return True
    if char != ".":
        return False
    normalized = token.text.strip().rstrip(_CLOSING_MARKS).lower()
    if normalized in _COMMON_ABBREVIATIONS:
        return False
    if re.fullmatch(r"(?:[a-z]\.){2,}", normalized):
        return False
    return True


def _is_weak_boundary(token: WordToken) -> bool:
    return _terminal_character(token.text) in _WEAK_PUNCTUATION


def join_token_text(tokens: Sequence[WordToken]) -> str:
    """Join Whisper tokens while preserving natural CJK and Latin spacing."""

    result = ""
    for token in tokens:
        piece = token.text
        if not piece:
            continue
        if result and not piece[0].isspace():
            previous = result[-1]
            current = piece[0]
            if (
                previous.isascii()
                and previous.isalnum()
                and current.isascii()
                and current.isalnum()
            ):
                result += " "
        result += piece

    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", result)
    return result


def _normalize_tokens(tokens: Iterable[WordToken]) -> list[WordToken]:
    valid: list[WordToken] = []
    for token in tokens:
        text = token.text
        start = max(0.0, float(token.start))
        end = max(start, float(token.end))
        if text and text.strip() and end > start:
            valid.append(WordToken(text=text, start=start, end=end))
    valid.sort(key=lambda item: (item.start, item.end))
    return valid


def _choose_long_split(tokens: Sequence[WordToken], max_duration: float) -> int:
    start = tokens[0].start
    allowed = [
        index for index in range(1, len(tokens)) if tokens[index - 1].end - start <= max_duration
    ]
    if not allowed:
        return 1

    weak = [index for index in allowed if _is_weak_boundary(tokens[index - 1])]
    if weak:
        return weak[-1]

    def gap_before(index: int) -> tuple[float, int]:
        return (max(0.0, tokens[index].start - tokens[index - 1].end), index)

    best_gap, best_index = max((gap_before(index) for index in allowed), default=(0.0, 1))
    if best_gap >= 0.15:
        return best_index
    return allowed[-1]


def _split_long_group(tokens: Sequence[WordToken], max_duration: float) -> list[list[WordToken]]:
    remaining = list(tokens)
    groups: list[list[WordToken]] = []
    while remaining:
        if remaining[-1].end - remaining[0].start <= max_duration:
            groups.append(remaining)
            break
        cut = _choose_long_split(remaining, max_duration)
        groups.append(remaining[:cut])
        remaining = remaining[cut:]
    return groups


def split_sentences(
    tokens: Iterable[WordToken],
    *,
    sentence_gap: float = 0.8,
    max_duration: float = 20.0,
) -> list[Sentence]:
    """Split timestamped tokens by punctuation, silence and maximum duration."""

    if sentence_gap < 0:
        raise ValueError("sentence_gap 不能为负数")
    if max_duration <= 0:
        raise ValueError("max_duration 必须大于 0")

    normalized = _normalize_tokens(tokens)
    if not normalized:
        return []

    coarse_groups: list[list[WordToken]] = []
    current: list[WordToken] = []
    for index, token in enumerate(normalized):
        current.append(token)
        is_last = index == len(normalized) - 1
        next_gap = max(0.0, normalized[index + 1].start - token.end) if not is_last else 0.0
        if is_last or _is_hard_boundary(token) or next_gap >= sentence_gap:
            coarse_groups.append(current)
            current = []

    final_groups: list[list[WordToken]] = []
    for group in coarse_groups:
        final_groups.extend(_split_long_group(group, max_duration))

    sentences: list[Sentence] = []
    for group in final_groups:
        text = join_token_text(group)
        if text:
            sentences.append(Sentence(text=text, start=group[0].start, end=group[-1].end))
    return sentences


def apply_clip_ranges(
    sentences: Sequence[Sentence],
    *,
    audio_duration: float,
    padding_before: float = 0.12,
    padding_after: float = 0.18,
    min_clip_duration: float = 0.35,
) -> list[Sentence]:
    """Add padded clip ranges without overlap or out-of-file timestamps."""

    if not sentences:
        return []
    if audio_duration <= 0:
        audio_duration = max(sentence.end for sentence in sentences)
    if min(padding_before, padding_after, min_clip_duration) < 0:
        raise ValueError("留白和最短片段时长不能为负数")

    ordered = sorted(sentences, key=lambda sentence: (sentence.start, sentence.end))
    boundaries = [0.0]
    for previous, current in zip(ordered, ordered[1:]):
        midpoint = (previous.end + current.start) / 2.0
        boundaries.append(max(boundaries[-1], min(audio_duration, midpoint)))
    boundaries.append(audio_duration)

    result: list[Sentence] = []
    for index, sentence in enumerate(ordered):
        left_limit = boundaries[index]
        right_limit = max(left_limit, boundaries[index + 1])
        clip_start = max(left_limit, sentence.start - padding_before)
        clip_end = min(right_limit, sentence.end + padding_after)

        target = min(min_clip_duration, right_limit - left_limit)
        deficit = max(0.0, target - (clip_end - clip_start))
        if deficit:
            extend_left = min(deficit / 2.0, clip_start - left_limit)
            clip_start -= extend_left
            deficit -= extend_left
            extend_right = min(deficit, right_limit - clip_end)
            clip_end += extend_right
            deficit -= extend_right
            clip_start -= min(deficit, clip_start - left_limit)

        sentence.clip_start = round(max(0.0, clip_start), 3)
        sentence.clip_end = round(min(audio_duration, max(clip_start, clip_end)), 3)
        result.append(sentence)
    return result
