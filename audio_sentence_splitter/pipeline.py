"""End-to-end recognition, sentence segmentation and artifact export."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from .artifacts import (
    build_manifest,
    render_sentence_text,
    render_transcript,
    sentence_basename,
    write_manifest,
    write_utf8_bom,
)
from .audio import export_audio_clip
from .errors import AudioSplitterError, CancelledError, NoSpeechError
from .models import PipelineOptions, PipelineResult, RecognitionResult, StatusCallback
from .recognizer import recognize_audio
from .segmenter import apply_clip_ranges, split_sentences


Recognizer = Callable[..., RecognitionResult]
Exporter = Callable[..., None]
_SUPPORTED_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
}


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("任务已取消")


def _validate_options(options: PipelineOptions) -> None:
    if not options.input_path.exists() or not options.input_path.is_file():
        raise AudioSplitterError(f"输入文件不存在：{options.input_path}")
    if options.model not in _SUPPORTED_MODELS:
        raise AudioSplitterError(f"不支持的模型：{options.model}")
    if options.output_format not in {"wav", "flac", "mp3"}:
        raise AudioSplitterError("输出格式必须是 wav、flac 或 mp3")
    if options.sentence_gap < 0:
        raise AudioSplitterError("静音分句阈值不能为负数")
    if options.max_sentence_duration <= 0:
        raise AudioSplitterError("最长句子时长必须大于 0")
    if (
        min(
            options.min_clip_duration,
            options.padding_before,
            options.padding_after,
            options.vad_min_silence_ms,
        )
        < 0
    ):
        raise AudioSplitterError("时长和 VAD 参数不能为负数")
    if options.beam_size < 1:
        raise AudioSplitterError("beam size 必须至少为 1")


def default_output_directory(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}_sentences"


def _preflight_targets(targets: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    conflicts = [path for path in targets if path.exists()]
    if conflicts:
        sample = "、".join(path.name for path in conflicts[:3])
        suffix = "……" if len(conflicts) > 3 else ""
        raise AudioSplitterError(f"输出文件已存在：{sample}{suffix}。请更换目录或允许覆盖。")


def split_audio(
    options: PipelineOptions,
    status_callback: StatusCallback | None = None,
    cancel_event: threading.Event | None = None,
    *,
    recognizer: Recognizer | None = None,
    exporter: Exporter | None = None,
) -> PipelineResult:
    """Run the complete local audio sentence splitting pipeline."""

    status = status_callback or (lambda _message: None)
    recognize = recognizer or recognize_audio
    export = exporter or export_audio_clip
    options.input_path = Path(options.input_path).expanduser()
    options.output_dir = (
        Path(options.output_dir).expanduser()
        if options.output_dir is not None
        else default_output_directory(options.input_path)
    )
    options.output_format = options.output_format.lower()
    _validate_options(options)
    _check_cancelled(cancel_event)

    status(f"输入文件：{options.input_path}")
    recognition = recognize(
        options.input_path,
        options,
        status_callback=status,
        cancel_event=cancel_event,
    )
    _check_cancelled(cancel_event)
    status("正在按标点和停顿组织完整句子……")
    sentences = split_sentences(
        recognition.tokens,
        sentence_gap=options.sentence_gap,
        max_duration=options.max_sentence_duration,
    )
    if not sentences:
        raise NoSpeechError("未识别到可用的人声内容，请检查音频或尝试更大的模型。")

    audio_duration = max(recognition.duration, max(sentence.end for sentence in sentences))
    sentences = apply_clip_ranges(
        sentences,
        audio_duration=audio_duration,
        padding_before=options.padding_before,
        padding_after=options.padding_after,
        min_clip_duration=options.min_clip_duration,
    )
    output_dir = options.output_dir
    if output_dir.exists() and not output_dir.is_dir():
        raise AudioSplitterError(f"输出路径不是文件夹：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    count = len(sentences)
    final_targets: list[Path] = []
    for index, sentence in enumerate(sentences, start=1):
        basename = sentence_basename(index, count)
        sentence.output_path = output_dir / f"{basename}.{options.output_format}"
        sentence.text_path = output_dir / f"{basename}.txt"
        final_targets.extend((sentence.output_path, sentence.text_path))
    transcript_path = output_dir / "transcript.txt"
    manifest_path = output_dir / "segments.json"
    final_targets.extend((transcript_path, manifest_path))
    _preflight_targets(final_targets, options.overwrite)

    staging = Path(tempfile.mkdtemp(prefix=".splitter-staging-", dir=output_dir))
    try:
        for index, sentence in enumerate(sentences, start=1):
            _check_cancelled(cancel_event)
            basename = sentence_basename(index, count)
            staged_audio = staging / f"{basename}.{options.output_format}"
            staged_text = staging / f"{basename}.txt"
            status(f"正在导出第 {index}/{count} 句：{sentence.text}")
            export(
                options.input_path,
                staged_audio,
                sentence.clip_start,
                sentence.clip_end,
                overwrite=True,
            )
            write_utf8_bom(staged_text, render_sentence_text(sentence) + "\n")

        _check_cancelled(cancel_event)
        write_utf8_bom(staging / "transcript.txt", render_transcript(sentences))
        write_manifest(
            staging / "segments.json",
            build_manifest(options, recognition, sentences, audio_duration=audio_duration),
        )
        _preflight_targets(final_targets, options.overwrite)

        status("正在提交输出文件……")
        for staged_file in sorted(staging.iterdir()):
            os.replace(staged_file, output_dir / staged_file.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    status(f"完成：共生成 {count} 句，输出目录为 {output_dir}")
    return PipelineResult(
        input_path=options.input_path,
        output_dir=output_dir,
        manifest_path=manifest_path,
        sentences=tuple(sentences),
        language=recognition.language,
        language_probability=recognition.language_probability,
        audio_duration=audio_duration,
        model=options.model,
        device=recognition.device,
        compute_type=recognition.compute_type,
        warnings=recognition.warnings,
    )
