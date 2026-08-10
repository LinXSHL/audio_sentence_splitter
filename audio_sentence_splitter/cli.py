"""Command-line interface for the audio sentence splitter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .errors import AudioSplitterError, CancelledError
from .models import PipelineOptions
from .pipeline import split_audio


MODEL_CHOICES = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-sentence-split",
        description="本地识别人声并按完整句子拆分音频，同时输出文字和时间。",
    )
    parser.add_argument("input", nargs="?", type=Path, help="要处理的音频或视频文件")
    parser.add_argument("-o", "--output-dir", type=Path, help="输出目录")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="small", help="Whisper 模型")
    parser.add_argument(
        "--language",
        default="auto",
        help="语言代码（如 zh、en）；默认 auto 自动检测",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--compute-type",
        choices=("int8", "int8_float16", "float16", "float32"),
        help="高级选项：覆盖默认计算精度",
    )
    parser.add_argument("--format", choices=("wav", "flac", "mp3"), default="wav")
    parser.add_argument("--sentence-gap", type=float, default=0.8, help="静音分句阈值（秒）")
    parser.add_argument(
        "--max-sentence-duration",
        type=float,
        default=20.0,
        help="单句最长时长（秒）",
    )
    parser.add_argument("--min-clip-duration", type=float, default=0.35, help="最短片段（秒）")
    parser.add_argument("--padding-before", type=float, default=0.12, help="句前留白（秒）")
    parser.add_argument("--padding-after", type=float, default=0.18, help="句后留白（秒）")
    parser.add_argument("--beam-size", type=int, default=5, help="识别 beam size")
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=500,
        help="VAD 最短静音（毫秒）",
    )
    parser.add_argument("--initial-prompt", help="用于专有名词等内容的识别提示")
    parser.add_argument("--cpu-threads", type=int, default=0, help="CPU 线程数，0 表示自动")
    parser.add_argument("--offline", action="store_true", help="只加载已下载的本地模型")
    parser.add_argument("--overwrite", action="store_true", help="覆盖同名输出文件")
    parser.add_argument("--gui", action="store_true", help="启动 Tkinter 图形界面")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _launch_gui() -> int:
    from .gui import main as gui_main

    gui_main()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        return _launch_gui()

    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.gui:
        return _launch_gui()
    if args.input is None:
        parser.error("必须提供输入文件，或使用 --gui")

    options = PipelineOptions(
        input_path=args.input,
        output_dir=args.output_dir,
        model=args.model,
        language=None if args.language.lower() == "auto" else args.language.lower(),
        device=args.device,
        compute_type=args.compute_type,
        output_format=args.format,
        sentence_gap=args.sentence_gap,
        max_sentence_duration=args.max_sentence_duration,
        min_clip_duration=args.min_clip_duration,
        padding_before=args.padding_before,
        padding_after=args.padding_after,
        beam_size=args.beam_size,
        vad_min_silence_ms=args.vad_min_silence_ms,
        initial_prompt=args.initial_prompt,
        offline=args.offline,
        overwrite=args.overwrite,
        cpu_threads=args.cpu_threads,
    )

    def report(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        result = split_audio(options, status_callback=report)
    except CancelledError:
        print("任务已取消。", file=sys.stderr)
        return 130
    except AudioSplitterError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("任务已由用户中断。", file=sys.stderr)
        return 130

    print(f"共生成 {len(result.sentences)} 句：{result.output_dir}")
    for sentence in result.sentences:
        print(f"{sentence.output_path}\t{sentence.start:.3f}-{sentence.end:.3f}\t{sentence.text}")
    if result.warnings:
        for warning in result.warnings:
            print(f"警告：{warning}", file=sys.stderr)
    return 0
