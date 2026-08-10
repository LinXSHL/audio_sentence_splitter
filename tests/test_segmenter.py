from __future__ import annotations

import unittest

from audio_sentence_splitter.models import Sentence, WordToken
from audio_sentence_splitter.segmenter import apply_clip_ranges, split_sentences


def token(text: str, start: float, end: float) -> WordToken:
    return WordToken(text=text, start=start, end=end)


class SentenceSegmentationTests(unittest.TestCase):
    def test_chinese_and_english_terminal_punctuation(self) -> None:
        tokens = [
            token("大家好，", 0.0, 0.8),
            token("欢迎来到我的频道。", 0.8, 2.0),
            token(" Today", 2.1, 2.5),
            token(" we", 2.5, 2.7),
            token(" learn", 2.7, 3.1),
            token(" Python.", 3.1, 3.7),
            token("现在开始吧！", 3.8, 4.7),
        ]
        sentences = split_sentences(tokens)
        self.assertEqual(
            [sentence.text for sentence in sentences],
            ["大家好，欢迎来到我的频道。", "Today we learn Python.", "现在开始吧！"],
        )
        self.assertEqual(
            [(s.start, s.end) for s in sentences], [(0.0, 2.0), (2.1, 3.7), (3.8, 4.7)]
        )

    def test_silence_splits_unpunctuated_speech(self) -> None:
        tokens = [
            token("hello", 0.0, 0.4),
            token(" world", 0.4, 0.8),
            token("second", 1.7, 2.1),
            token(" phrase", 2.1, 2.6),
        ]
        sentences = split_sentences(tokens, sentence_gap=0.8)
        self.assertEqual([s.text for s in sentences], ["hello world", "second phrase"])

    def test_common_abbreviation_does_not_end_sentence(self) -> None:
        tokens = [
            token("Dr.", 0.0, 0.2),
            token(" Smith", 0.2, 0.5),
            token(" arrived.", 0.5, 1.0),
        ]
        self.assertEqual([s.text for s in split_sentences(tokens)], ["Dr. Smith arrived."])

    def test_long_sentence_prefers_weak_punctuation(self) -> None:
        tokens = [
            token("one", 0.0, 0.8),
            token(" two,", 0.8, 1.6),
            token(" three", 1.6, 2.4),
            token(" four", 2.4, 3.2),
            token(" five", 3.2, 4.0),
            token(" six", 4.0, 4.8),
        ]
        sentences = split_sentences(tokens, max_duration=3.0)
        self.assertEqual(sentences[0].text, "one two,")
        self.assertEqual(" ".join(s.text for s in sentences[1:]), "three four five six")
        self.assertTrue(all((s.end - s.start) <= 3.0 for s in sentences))

    def test_empty_and_invalid_tokens_are_ignored(self) -> None:
        tokens = [token("", 0, 1), token("  ", 1, 2), token("okay.", 2, 3)]
        self.assertEqual([s.text for s in split_sentences(tokens)], ["okay."])


class ClipRangeTests(unittest.TestCase):
    def test_padding_is_clamped_and_never_overlaps(self) -> None:
        sentences = [
            Sentence("first.", 0.05, 1.0),
            Sentence("second.", 1.1, 2.0),
            Sentence("third.", 2.1, 2.95),
        ]
        result = apply_clip_ranges(
            sentences,
            audio_duration=3.0,
            padding_before=0.2,
            padding_after=0.3,
        )
        self.assertEqual(result[0].clip_start, 0.0)
        self.assertEqual(result[-1].clip_end, 3.0)
        for previous, current in zip(result, result[1:]):
            self.assertLessEqual(previous.clip_end, current.clip_start)

    def test_minimum_clip_expands_inside_neighbor_bounds(self) -> None:
        sentences = [Sentence("a.", 1.0, 1.05)]
        result = apply_clip_ranges(
            sentences,
            audio_duration=2.0,
            padding_before=0,
            padding_after=0,
            min_clip_duration=0.35,
        )
        self.assertAlmostEqual(result[0].clip_end - result[0].clip_start, 0.35, places=3)


if __name__ == "__main__":
    unittest.main()
