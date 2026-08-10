from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from audio_sentence_splitter.errors import AudioSplitterError, CancelledError, NoSpeechError
from audio_sentence_splitter.models import PipelineOptions, RecognitionResult, WordToken
from audio_sentence_splitter.pipeline import split_audio


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_path = self.root / "source.wav"
        self.input_path.write_bytes(b"placeholder")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _recognizer(path, options, **kwargs):
        del path, options, kwargs
        return RecognitionResult(
            tokens=(
                WordToken("第一句。", 0.0, 1.0),
                WordToken("第二句。", 1.2, 2.0),
            ),
            language="zh",
            language_probability=0.99,
            duration=2.2,
            device="cpu",
            compute_type="int8",
        )

    @staticmethod
    def _exporter(input_path, output_path, start, end, **kwargs):
        del input_path, kwargs
        output_path.write_text(f"{start:.3f}-{end:.3f}", encoding="ascii")

    def test_complete_pipeline_outputs_pairs_and_summaries(self) -> None:
        output = self.root / "output"
        options = PipelineOptions(self.input_path, output_dir=output)
        result = split_audio(options, recognizer=self._recognizer, exporter=self._exporter)

        self.assertEqual(len(result.sentences), 2)
        self.assertTrue((output / "001_.wav").is_file())
        self.assertTrue((output / "001_.txt").is_file())
        self.assertTrue((output / "002_.wav").is_file())
        self.assertTrue((output / "transcript.txt").is_file())
        manifest = json.loads((output / "segments.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["segments"][1]["text"], "第二句。")
        paired = (output / "001_.txt").read_text(encoding="utf-8-sig")
        self.assertEqual(paired, "第一句。  00:00.000 - 00:01.000\n")

    def test_existing_outputs_are_never_silently_overwritten(self) -> None:
        output = self.root / "output"
        output.mkdir()
        existing = output / "001_.wav"
        existing.write_text("keep", encoding="ascii")
        options = PipelineOptions(self.input_path, output_dir=output)
        with self.assertRaises(AudioSplitterError):
            split_audio(options, recognizer=self._recognizer, exporter=self._exporter)
        self.assertEqual(existing.read_text(encoding="ascii"), "keep")

    def test_overwrite_is_explicit(self) -> None:
        output = self.root / "output"
        output.mkdir()
        (output / "001_.wav").write_text("old", encoding="ascii")
        options = PipelineOptions(self.input_path, output_dir=output, overwrite=True)
        split_audio(options, recognizer=self._recognizer, exporter=self._exporter)
        self.assertNotEqual((output / "001_.wav").read_text(encoding="ascii"), "old")

    def test_pre_cancelled_job_creates_no_output(self) -> None:
        output = self.root / "output"
        cancelled = threading.Event()
        cancelled.set()
        options = PipelineOptions(self.input_path, output_dir=output)
        with self.assertRaises(CancelledError):
            split_audio(
                options,
                cancel_event=cancelled,
                recognizer=self._recognizer,
                exporter=self._exporter,
            )
        self.assertFalse(output.exists())

    def test_no_speech_has_clear_error(self) -> None:
        def empty_recognizer(path, options, **kwargs):
            del path, options, kwargs
            return RecognitionResult((), "unknown", 0.0, 1.0, "cpu", "int8")

        with self.assertRaises(NoSpeechError):
            split_audio(
                PipelineOptions(self.input_path, output_dir=self.root / "output"),
                recognizer=empty_recognizer,
                exporter=self._exporter,
            )


if __name__ == "__main__":
    unittest.main()
