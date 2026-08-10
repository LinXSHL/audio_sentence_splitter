"""Text and JSON artifact formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .models import PipelineOptions, RecognitionResult, Sentence


def format_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(float(seconds) * 1000))
    milliseconds = total_milliseconds % 1000
    total_seconds = total_milliseconds // 1000
    second = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    if hour:
        return f"{hour:02d}:{minute:02d}:{second:02d}.{milliseconds:03d}"
    return f"{total_minutes:02d}:{second:02d}.{milliseconds:03d}"


def sentence_basename(index: int, count: int) -> str:
    width = max(3, len(str(max(1, count))))
    return f"{index:0{width}d}_"


def render_sentence_text(sentence: Sentence) -> str:
    return f"{sentence.text}  {format_timestamp(sentence.start)} - {format_timestamp(sentence.end)}"


def render_transcript(sentences: Sequence[Sentence]) -> str:
    count = len(sentences)
    lines = [
        f"{sentence_basename(index, count)}  "
        f"{format_timestamp(sentence.start)} - {format_timestamp(sentence.end)}  "
        f"{sentence.text}"
        for index, sentence in enumerate(sentences, start=1)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def build_manifest(
    options: PipelineOptions,
    recognition: RecognitionResult,
    sentences: Sequence[Sentence],
    *,
    audio_duration: float | None = None,
) -> dict[str, Any]:
    count = len(sentences)
    return {
        "schema_version": 1,
        "input_file": str(options.input_path.resolve()),
        "model": options.model,
        "language": recognition.language,
        "language_probability": round(recognition.language_probability, 6),
        "audio_duration": round(
            recognition.duration if audio_duration is None else audio_duration,
            3,
        ),
        "device": recognition.device,
        "compute_type": recognition.compute_type,
        "warnings": list(recognition.warnings),
        "segments": [
            {
                "index": index,
                "text": sentence.text,
                "start": round(sentence.start, 3),
                "end": round(sentence.end, 3),
                "clip_start": round(sentence.clip_start, 3),
                "clip_end": round(sentence.clip_end, 3),
                "audio_file": f"{sentence_basename(index, count)}.{options.output_format}",
                "text_file": f"{sentence_basename(index, count)}.txt",
            }
            for index, sentence in enumerate(sentences, start=1)
        ],
    }


def write_utf8_bom(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8-sig", newline="\n")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
