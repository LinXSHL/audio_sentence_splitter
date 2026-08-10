from __future__ import annotations

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from audio_sentence_splitter.errors import AudioSplitterError, CancelledError
from audio_sentence_splitter.models import PipelineOptions
from audio_sentence_splitter.recognizer import recognize_audio


class FakeModel:
    def transcribe(self, path, **kwargs):
        del path, kwargs
        words = [SimpleNamespace(word="hello.", start=0.0, end=1.0)]
        segments = [SimpleNamespace(text="hello.", start=0.0, end=1.0, words=words)]
        info = SimpleNamespace(duration=1.2, language="en", language_probability=0.95)
        return iter(segments), info


class RecognizerTests(unittest.TestCase):
    def test_auto_cuda_failure_retries_cpu(self) -> None:
        attempts: list[tuple[str, str]] = []

        def factory(model, **kwargs):
            del model
            attempts.append((kwargs["device"], kwargs["compute_type"]))
            if kwargs["device"] == "cuda":
                raise RuntimeError("missing cudnn")
            return FakeModel()

        options = PipelineOptions(Path("input.wav"), model="tiny", device="auto")
        result = recognize_audio(
            Path("input.wav"),
            options,
            model_factory=factory,
            cuda_detector=lambda: True,
        )
        self.assertEqual(attempts, [("cuda", "float16"), ("cpu", "int8")])
        self.assertEqual(result.device, "cpu")
        self.assertIn("CUDA", result.warnings[0])

    def test_forced_cuda_does_not_hide_configuration_error(self) -> None:
        def factory(model, **kwargs):
            del model, kwargs
            raise RuntimeError("CUDA unavailable")

        options = PipelineOptions(Path("input.wav"), model="tiny", device="cuda")
        with self.assertRaisesRegex(AudioSplitterError, "CUDA unavailable"):
            recognize_audio(Path("input.wav"), options, model_factory=factory)

    def test_offline_load_error_explains_cache_requirement(self) -> None:
        def factory(model, **kwargs):
            del model, kwargs
            raise RuntimeError("model not found")

        options = PipelineOptions(Path("input.wav"), model="tiny", device="cpu", offline=True)
        with self.assertRaisesRegex(AudioSplitterError, "离线模式"):
            recognize_audio(Path("input.wav"), options, model_factory=factory)

    def test_pre_cancelled_recognition_stops_before_model_load(self) -> None:
        cancelled = threading.Event()
        cancelled.set()
        called = False

        def factory(model, **kwargs):
            nonlocal called
            del model, kwargs
            called = True
            return FakeModel()

        with self.assertRaises(CancelledError):
            recognize_audio(
                Path("input.wav"),
                PipelineOptions(Path("input.wav"), device="cpu"),
                cancel_event=cancelled,
                model_factory=factory,
            )
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
