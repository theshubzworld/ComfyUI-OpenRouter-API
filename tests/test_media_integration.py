import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from openrouter_simple.cancellation import NodeDeadline, run_process
from openrouter_simple.media import (
    AUDIO_LIMIT,
    IMAGE_LIMIT,
    VIDEO_LIMIT,
    prepare_audio,
    prepare_image,
    prepare_video,
)


class FileVideo:
    def __init__(self, path: str, trim: tuple[float, float] = (0.0, 0.0)):
        self.path = path
        self.trim = trim

    def get_stream_source(self):
        return self.path

    def get_active_trim_window(self):
        return self.trim


class MediaIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_is_webp_below_cap(self):
        image = torch.rand((1, 1024, 1024, 3))
        prepared = await prepare_image(NodeDeadline(15), image)
        self.assertEqual(prepared.mime_type, "image/webp")
        self.assertLessEqual(len(prepared.data), IMAGE_LIMIT)

    async def test_image_batch_accepts_first_image(self):
        batch = torch.rand((4, 256, 256, 3))
        prepared = await prepare_image(NodeDeadline(15), batch)
        self.assertEqual(prepared.mime_type, "image/webp")
        self.assertEqual(prepared.details["width"], 256)
        self.assertEqual(prepared.details["height"], 256)

    async def test_audio_is_mp3_below_cap(self):
        sample_rate = 48_000
        waveform = torch.sin(torch.arange(sample_rate * 3) * (440 * 2 * torch.pi / sample_rate)).reshape(1, 1, -1)
        prepared = await prepare_audio(
            NodeDeadline(15), {"waveform": waveform, "sample_rate": sample_rate}
        )
        self.assertEqual(prepared.mime_type, "audio/mpeg")
        self.assertLessEqual(len(prepared.data), AUDIO_LIMIT)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is required")
    async def test_video_is_mp4_below_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            await run_process(
                NodeDeadline(15),
                shutil.which("ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x360:rate=24",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000",
                "-t",
                "1.5",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(source),
            )
            source_data = source.read_bytes()
            prepared = await prepare_video(NodeDeadline(30), FileVideo(str(source)))
        self.assertEqual(prepared.mime_type, "video/mp4")
        self.assertLessEqual(len(prepared.data), VIDEO_LIMIT)
        self.assertEqual(prepared.data, source_data)
        self.assertEqual(prepared.details["transcode"], "none")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is required")
    async def test_under_cap_incompatible_video_is_not_enlarged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            await run_process(
                NodeDeadline(15),
                shutil.which("ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x360:rate=24",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000",
                "-t",
                "2",
                "-c:v",
                "mpeg4",
                "-q:v",
                "2",
                "-c:a",
                "aac",
                str(source),
            )
            source_bytes = source.stat().st_size
            prepared = await prepare_video(NodeDeadline(30), FileVideo(str(source)))
        self.assertLess(source_bytes, VIDEO_LIMIT)
        self.assertLessEqual(len(prepared.data), source_bytes)
        self.assertEqual(prepared.details["transcode"], "H.264/AAC")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is required")
    async def test_active_trim_disables_compatible_passthrough(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            await run_process(
                NodeDeadline(15),
                shutil.which("ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x360:rate=24",
                "-t",
                "2",
                "-c:v",
                "libx264",
                str(source),
            )
            source_data = source.read_bytes()
            prepared = await prepare_video(
                NodeDeadline(30), FileVideo(str(source), trim=(0.5, 1.0))
            )
        self.assertNotEqual(prepared.data, source_data)
        self.assertLessEqual(len(prepared.data), len(source_data))
        self.assertAlmostEqual(prepared.details["duration_seconds"], 1.0, places=2)
        self.assertEqual(prepared.details["transcode"], "H.264/AAC")


if __name__ == "__main__":
    unittest.main()
