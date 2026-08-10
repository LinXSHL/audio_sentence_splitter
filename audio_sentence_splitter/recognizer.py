"""Local faster-whisper integration with automatic CUDA fallback."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import AudioSplitterError, CancelledError, DependencyError
from .models import PipelineOptions, RecognitionResult, StatusCallback, WordToken


ModelFactory = Callable[..., Any]
CudaDetector = Callable[[], bool]


def _default_model_factory(model_name: str, **kwargs: Any) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise DependencyError("缺少 faster-whisper。请先运行：python -m pip install -e .") from exc
    return WhisperModel(model_name, **kwargs)


def _default_cuda_detector() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, RuntimeError, OSError):
        return False


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("任务已取消")


def _friendly_model_error(exc: Exception, offline: bool) -> AudioSplitterError:
    message = str(exc).strip()
    if offline:
        return AudioSplitterError(
            "离线模式下无法加载所选模型。请先联网运行一次以下载模型，"
            f"或关闭离线模式。详细信息：{message}"
        )
    return AudioSplitterError(f"无法加载或运行语音识别模型：{message}")


def _run_attempt(
    path: Path,
    options: PipelineOptions,
    *,
    device: str,
    compute_type: str,
    model_factory: ModelFactory,
    status_callback: StatusCallback,
    cancel_event: threading.Event | None,
) -> RecognitionResult:
    _check_cancelled(cancel_event)
    status_callback(f"正在加载 {options.model} 模型（{device}/{compute_type}）……")
    model = model_factory(
        options.model,
        device=device,
        compute_type=compute_type,
        cpu_threads=options.cpu_threads,
        local_files_only=options.offline,
    )
    _check_cancelled(cancel_event)
    status_callback("正在识别人声并计算词级时间戳……")
    segments, info = model.transcribe(
        str(path),
        task="transcribe",
        language=options.language,
        beam_size=options.beam_size,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": options.vad_min_silence_ms},
        condition_on_previous_text=True,
        initial_prompt=options.initial_prompt or None,
        multilingual=True,
    )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    tokens: list[WordToken] = []
    last_reported_second = -5
    for segment in segments:
        _check_cancelled(cancel_event)
        words = getattr(segment, "words", None)
        if words:
            for word in words:
                start = getattr(word, "start", None)
                end = getattr(word, "end", None)
                text = getattr(word, "word", "")
                if start is not None and end is not None and text:
                    tokens.append(WordToken(str(text), float(start), float(end)))
        else:
            text = str(getattr(segment, "text", ""))
            start = float(getattr(segment, "start", 0.0))
            end = float(getattr(segment, "end", start))
            if text.strip() and end > start:
                tokens.append(WordToken(text, start, end))

        current_second = int(float(getattr(segment, "end", 0.0)))
        if current_second - last_reported_second >= 5:
            if duration > 0:
                percent = min(100, int(current_second / duration * 100))
                status_callback(f"识别进度约 {percent}%（{current_second:.0f}/{duration:.0f} 秒）")
            else:
                status_callback(f"已识别到 {current_second:.0f} 秒")
            last_reported_second = current_second

    if duration <= 0 and tokens:
        duration = max(token.end for token in tokens)
    return RecognitionResult(
        tokens=tuple(tokens),
        language=str(getattr(info, "language", options.language or "unknown")),
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=duration,
        device=device,
        compute_type=compute_type,
    )


def recognize_audio(
    path: Path,
    options: PipelineOptions,
    *,
    status_callback: StatusCallback | None = None,
    cancel_event: threading.Event | None = None,
    model_factory: ModelFactory | None = None,
    cuda_detector: CudaDetector | None = None,
) -> RecognitionResult:
    """Recognize an audio file locally and retry on CPU when auto CUDA fails."""

    status = status_callback or (lambda _message: None)
    factory = model_factory or _default_model_factory
    detect_cuda = cuda_detector or _default_cuda_detector
    requested = options.device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise AudioSplitterError("device 必须是 auto、cpu 或 cuda")

    if requested == "auto":
        initial_device = "cuda" if detect_cuda() else "cpu"
    else:
        initial_device = requested
    initial_compute = options.compute_type or ("float16" if initial_device == "cuda" else "int8")

    try:
        return _run_attempt(
            path,
            options,
            device=initial_device,
            compute_type=initial_compute,
            model_factory=factory,
            status_callback=status,
            cancel_event=cancel_event,
        )
    except CancelledError:
        raise
    except Exception as exc:
        if requested != "auto" or initial_device != "cuda":
            if isinstance(exc, DependencyError):
                raise
            raise _friendly_model_error(exc, options.offline) from exc

        warning = f"CUDA 初始化或识别失败，已自动回退 CPU：{exc}"
        status(warning)
        try:
            fallback_compute = (
                options.compute_type if options.compute_type in {"int8", "float32"} else "int8"
            )
            result = _run_attempt(
                path,
                options,
                device="cpu",
                compute_type=fallback_compute,
                model_factory=factory,
                status_callback=status,
                cancel_event=cancel_event,
            )
            return RecognitionResult(
                tokens=result.tokens,
                language=result.language,
                language_probability=result.language_probability,
                duration=result.duration,
                device=result.device,
                compute_type=result.compute_type,
                warnings=(warning,),
            )
        except CancelledError:
            raise
        except Exception as cpu_exc:
            if isinstance(cpu_exc, DependencyError):
                raise
            raise _friendly_model_error(cpu_exc, options.offline) from cpu_exc
