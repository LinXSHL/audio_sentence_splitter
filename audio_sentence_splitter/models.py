"""Shared data structures for recognition, segmentation and export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


StatusCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class WordToken:
    text: str
    start: float
    end: float


@dataclass(slots=True)
class Sentence:
    text: str
    start: float
    end: float
    clip_start: float = 0.0
    clip_end: float = 0.0
    output_path: Path | None = None
    text_path: Path | None = None


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    tokens: tuple[WordToken, ...]
    language: str
    language_probability: float
    duration: float
    device: str
    compute_type: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class PipelineOptions:
    input_path: Path
    output_dir: Path | None = None
    model: str = "small"
    language: str | None = None
    device: str = "auto"
    compute_type: str | None = None
    output_format: str = "wav"
    sentence_gap: float = 0.8
    max_sentence_duration: float = 20.0
    min_clip_duration: float = 0.35
    padding_before: float = 0.12
    padding_after: float = 0.18
    beam_size: int = 5
    vad_min_silence_ms: int = 500
    initial_prompt: str | None = None
    offline: bool = False
    overwrite: bool = False
    cpu_threads: int = 0


@dataclass(frozen=True, slots=True)
class PipelineResult:
    input_path: Path
    output_dir: Path
    manifest_path: Path
    sentences: tuple[Sentence, ...]
    language: str
    language_probability: float
    audio_duration: float
    model: str
    device: str
    compute_type: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
