from __future__ import annotations

import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from audio_sentence_splitter.audio import export_audio_clip


class AudioExportIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("importlib").util.find_spec("imageio_ffmpeg"),
        "imageio-ffmpeg is not installed",
    )
    def test_real_ffmpeg_exports_requested_wav_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            output = root / "clip.wav"
            rate = 16_000
            samples = [
                struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * index / rate)))
                for index in range(rate)
            ]
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(b"".join(samples))

            export_audio_clip(source, output, 0.2, 0.6)
            with wave.open(str(output), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            self.assertAlmostEqual(duration, 0.4, delta=0.03)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("imageio_ffmpeg"),
        "imageio-ffmpeg is not installed",
    )
    def test_real_ffmpeg_supports_all_documented_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            rate = 8_000
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(b"\x00\x00" * rate)

            for extension in ("wav", "flac", "mp3"):
                with self.subTest(extension=extension):
                    output = root / f"clip.{extension}"
                    export_audio_clip(source, output, 0.1, 0.5)
                    self.assertGreater(output.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
