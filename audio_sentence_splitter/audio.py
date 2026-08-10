"""Precise audio clip export through the bundled FFmpeg executable."""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

from .errors import AudioSplitterError, DependencyError


_FORMAT_ARGUMENTS = {
    "wav": ["-c:a", "pcm_s16le"],
    "flac": ["-c:a", "flac"],
    "mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
}


def get_ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise DependencyError("找不到 FFmpeg。请运行：python -m pip install -e .") from exc


def export_audio_clip(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
    *,
    overwrite: bool = False,
    ffmpeg_executable: str | None = None,
) -> None:
    """Export the first audio stream to one sentence file using an atomic replace."""

    output_format = output_path.suffix.lower().lstrip(".")
    if output_format not in _FORMAT_ARGUMENTS:
        raise AudioSplitterError(f"不支持的输出格式：{output_format}")
    if end <= start:
        raise AudioSplitterError(f"无效的音频时间范围：{start:.3f} - {end:.3f}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在：{output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix}"
    )
    executable = ffmpeg_executable or get_ffmpeg_executable()
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{max(0.0, start):.6f}",
        "-i",
        str(input_path),
        "-t",
        f"{end - start:.6f}",
        "-map",
        "0:a:0",
        "-vn",
        *_FORMAT_ARGUMENTS[output_format],
        str(temporary),
    ]
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        creationflags = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
        if completed.returncode != 0 or not temporary.exists():
            detail = completed.stderr.strip() or "FFmpeg 未生成输出文件"
            raise AudioSplitterError(f"音频导出失败：{detail}")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
