from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audio_sentence_splitter.artifacts import (
    build_manifest,
    format_timestamp,
    render_sentence_text,
    render_transcript,
    sentence_basename,
    write_manifest,
    write_utf8_bom,
)
from audio_sentence_splitter.models import PipelineOptions, RecognitionResult, Sentence


class ArtifactTests(unittest.TestCase):
    def test_timestamp_formats_minutes_and_hours(self) -> None:
        self.assertEqual(format_timestamp(0), "00:00.000")
        self.assertEqual(format_timestamp(65.4321), "01:05.432")
        self.assertEqual(format_timestamp(3661.5), "01:01:01.500")

    def test_numeric_basename_has_at_least_three_digits(self) -> None:
        self.assertEqual(sentence_basename(1, 3), "001_")
        self.assertEqual(sentence_basename(1, 1200), "0001_")

    def test_text_layout_matches_contract(self) -> None:
        sentence = Sentence("大家好。", 0.0, 5.0)
        self.assertEqual(render_sentence_text(sentence), "大家好。  00:00.000 - 00:05.000")
        self.assertEqual(
            render_transcript([sentence]),
            "001_  00:00.000 - 00:05.000  大家好。\n",
        )

    def test_manifest_contains_recognition_and_clip_ranges(self) -> None:
        options = PipelineOptions(Path("input.wav"), output_format="wav")
        recognition = RecognitionResult(
            tokens=(),
            language="zh",
            language_probability=0.98,
            duration=6.0,
            device="cpu",
            compute_type="int8",
        )
        sentence = Sentence("测试。", 1.0, 2.0, 0.88, 2.18)
        manifest = build_manifest(options, recognition, [sentence])
        self.assertEqual(manifest["language"], "zh")
        self.assertEqual(manifest["segments"][0]["audio_file"], "001_.wav")
        self.assertEqual(manifest["segments"][0]["clip_start"], 0.88)

    def test_text_has_utf8_bom_but_json_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_path = root / "001_.txt"
            json_path = root / "segments.json"
            write_utf8_bom(text_path, "中文\n")
            write_manifest(json_path, {"text": "中文"})
            self.assertTrue(text_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertFalse(json_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["text"], "中文")


if __name__ == "__main__":
    unittest.main()
