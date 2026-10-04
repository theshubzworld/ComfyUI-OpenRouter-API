from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import shutil
import tempfile
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cancellation import NodeDeadline, run_process

IMAGE_LIMIT = 1_000_000
VIDEO_LIMIT = 10_000_000
AUDIO_LIMIT = 1_000_000


@dataclass(frozen=True)
class PreparedMedia:
    modality: str
    mime_type: str
    data: bytes
    source_bytes: int
    details: dict[str, Any]

    @property
    def compressed_bytes(self) -> int:
        return len(self.data)

    @property
    def base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @property
    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.base64}"

    def public_info(self) -> dict[str, Any]:
        return {
            "source_bytes": self.source_bytes,
            "compressed_bytes": self.compressed_bytes,
            **self.details,
        }


def _image_to_webp(image_tensor: Any) -> PreparedMedia:
    if getattr(image_tensor, "ndim", None) != 4 or int(image_tensor.shape[0]) < 1:
        raise ValueError("image input must be a 4D IMAGE tensor with at least one image")
    array = image_tensor[0].detach().cpu().clamp(0, 1).mul(255).byte().numpy()
    if array.shape[-1] not in (3, 4):
        raise ValueError("image input must have RGB or RGBA channels")
    source_bytes = int(array.nbytes)
    has_alpha = array.shape[-1] == 4
    image = Image.fromarray(array)
    original_size = image.size
    best: tuple[bytes, int, tuple[int, int]] | None = None

    # Try high-quality and standard qualities with fast method=4
    quality_candidates = (85, 92, 75, 60, 45, 30)
    for quality in quality_candidates:
        output = io.BytesIO()
        save_kwargs: dict[str, Any] = {"format": "WEBP", "quality": quality, "method": 4}
        if has_alpha:
            save_kwargs["exact"] = True
        image.save(output, **save_kwargs)
        encoded = output.getvalue()
        if best is None or len(encoded) < len(best[0]):
            best = (encoded, quality, image.size)
        if len(encoded) <= IMAGE_LIMIT:
            return PreparedMedia(
                modality="image",
                mime_type="image/webp",
                data=encoded,
                source_bytes=source_bytes,
                details={
                    "width": image.width,
                    "height": image.height,
                    "quality": quality,
                    "resampler": "Pillow Lanczos" if image.size != original_size else "none",
                },
            )

    # If full resolution exceeds limit, downscale proportionally and retry
    for _ in range(10):
        assert best is not None
        if min(image.size) <= 64:
            break
        ratio = min(0.9, math.sqrt(IMAGE_LIMIT / max(len(best[0]), 1)) * 0.94)
        next_size = (
            max(64, int(image.width * ratio)),
            max(64, int(image.height * ratio)),
        )
        if next_size == image.size:
            break
        image = image.resize(next_size, Image.Resampling.LANCZOS)
        for quality in (80, 60, 40):
            output = io.BytesIO()
            save_kwargs = {"format": "WEBP", "quality": quality, "method": 4}
            if has_alpha:
                save_kwargs["exact"] = True
            image.save(output, **save_kwargs)
            encoded = output.getvalue()
            if best is None or len(encoded) < len(best[0]):
                best = (encoded, quality, image.size)
            if len(encoded) <= IMAGE_LIMIT:
                return PreparedMedia(
                    modality="image",
                    mime_type="image/webp",
                    data=encoded,
                    source_bytes=source_bytes,
                    details={
                        "width": image.width,
                        "height": image.height,
                        "quality": quality,
                        "resampler": "Pillow Lanczos" if image.size != original_size else "none",
                    },
                )

    raise ValueError(
        f"image could not be compressed below 1 MB (smallest result: {len(best[0]) if best else 0} bytes)"
    )


async def prepare_image(deadline: NodeDeadline, image_tensor: Any) -> PreparedMedia:
    return await deadline.run(asyncio.to_thread(_image_to_webp, image_tensor))


def _require_ffmpeg() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("video/audio inputs require ffmpeg and ffprobe on PATH")
    return ffmpeg, ffprobe


async def _copy_stream_source(source: Any, destination: Path, deadline: NodeDeadline) -> int:
    if isinstance(source, (str, os.PathLike)):
        return int(os.path.getsize(source))
    if not hasattr(source, "read"):
        raise ValueError("VIDEO input did not provide a path or readable stream source")

    if hasattr(source, "seek"):
        source.seek(0)
    with destination.open("wb") as output:
        while True:
            deadline.checkpoint()
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            await asyncio.sleep(0)
    return destination.stat().st_size


def _fraction(value: str | None, default: float) -> float:
    try:
        return float(Fraction(value)) if value else default
    except (ValueError, ZeroDivisionError):
        return default


async def _probe_video(deadline: NodeDeadline, ffprobe: str, source: str) -> dict[str, Any]:
    out, _ = await run_process(
        deadline,
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,avg_frame_rate,bit_rate:format=duration,format_name,bit_rate:format_tags=major_brand",
        "-of",
        "json",
        source,
    )
    try:
        payload = json.loads(out)
        streams = payload["streams"]
        video_stream = next(stream for stream in streams if stream.get("codec_type") == "video")
        audio_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "audio"), None
        )
        format_info = payload["format"]
        duration = float(format_info["duration"])
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        fps = _fraction(video_stream.get("avg_frame_rate"), 24.0)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("could not determine video duration and dimensions") from exc
    if duration <= 0 or width <= 0 or height <= 0:
        raise ValueError("video has invalid duration or dimensions")
    tags = format_info.get("tags") if isinstance(format_info.get("tags"), dict) else {}

    def optional_int(value: Any) -> int | None:
        try:
            result = int(value)
            return result if result > 0 else None
        except (TypeError, ValueError):
            return None

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": max(1.0, fps),
        "format_names": tuple(
            part.strip().lower()
            for part in str(format_info.get("format_name") or "").split(",")
            if part.strip()
        ),
        "major_brand": str(tags.get("major_brand") or "").strip().lower(),
        "video_codec": str(video_stream.get("codec_name") or "").lower(),
        "audio_codec": (
            str(audio_stream.get("codec_name") or "").lower() if audio_stream else None
        ),
        "video_bitrate": optional_int(video_stream.get("bit_rate")),
        "audio_bitrate": optional_int(audio_stream.get("bit_rate")) if audio_stream else None,
        "total_bitrate": optional_int(format_info.get("bit_rate")),
    }


def _has_active_trim(start_time: float, active_duration: float, source_duration: float) -> bool:
    if start_time > 0.001:
        return True
    return active_duration > 0 and active_duration < source_duration - 0.001


def _can_pass_video_through(
    probe: dict[str, Any], source_bytes: int, start_time: float, active_duration: float
) -> bool:
    mp4_family = "mp4" in probe["format_names"] and probe["major_brand"] != "qt"
    codecs_compatible = probe["video_codec"] == "h264" and probe["audio_codec"] in {
        None,
        "aac",
    }
    return (
        source_bytes <= VIDEO_LIMIT
        and not _has_active_trim(start_time, active_duration, probe["duration"])
        and mp4_family
        and codecs_compatible
    )


def _video_geometry(width: int, height: int, fps: float, bitrate: int) -> tuple[int, int, float]:
    output_fps = min(fps, 30.0)
    if bitrate < 350_000:
        output_fps = min(output_fps, 20.0)
    target_pixels = bitrate / max(output_fps * 0.07, 1)
    if width * height <= target_pixels:
        return width - width % 2, height - height % 2, output_fps
    scale = math.sqrt(target_pixels / (width * height))
    out_w = max(160, int(width * scale) // 2 * 2)
    out_h = max(90, int(height * scale) // 2 * 2)
    return out_w, out_h, output_fps


async def _encode_video_passes(
    deadline: NodeDeadline,
    ffmpeg: str,
    source: str,
    output: Path,
    passlog: Path,
    *,
    video_bitrate: int,
    audio_bitrate: int,
    width: int,
    height: int,
    fps: float,
    start_time: float,
    duration: float,
) -> None:
    seek = ["-ss", f"{start_time:.6f}"] if start_time > 0 else []
    trim = ["-t", f"{duration:.6f}"] if duration > 0 else []
    common = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-xerror",
        "-y",
        *seek,
        "-i",
        source,
        *trim,
        "-map",
        "0:v:0",
        "-vf",
        f"scale={width}:{height}:flags=lanczos,fps={fps:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        str(video_bitrate),
        "-maxrate",
        str(int(video_bitrate * 1.10)),
        "-bufsize",
        str(video_bitrate * 2),
        "-passlogfile",
        str(passlog),
    ]
    await run_process(deadline, *common, "-pass", "1", "-an", "-f", "null", os.devnull)
    audio = (
        ["-map", "0:a:0?", "-c:a", "aac", "-b:a", str(audio_bitrate)]
        if audio_bitrate > 0
        else ["-an"]
    )
    await run_process(
        deadline,
        *common,
        "-pass",
        "2",
        *audio,
        "-movflags",
        "+faststart",
        str(output),
    )


async def prepare_video(deadline: NodeDeadline, video: Any) -> PreparedMedia:
    ffmpeg, ffprobe = _require_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="openrouter-simple-video-") as temp_dir:
        temp = Path(temp_dir)
        source_obj = video.get_stream_source()
        if isinstance(source_obj, (str, os.PathLike)):
            source_path = str(source_obj)
            source_bytes = int(os.path.getsize(source_path))
        else:
            copied = temp / "input.bin"
            source_bytes = await _copy_stream_source(source_obj, copied, deadline)
            source_path = str(copied)
        probe = await _probe_video(deadline, ffprobe, source_path)
        start_time, active_duration = (0.0, 0.0)
        if hasattr(video, "get_active_trim_window"):
            start_time, active_duration = video.get_active_trim_window()
        start_time = float(start_time)
        active_duration = float(active_duration)
        duration = float(active_duration or probe["duration"] - start_time)
        if duration <= 0:
            raise ValueError("video trim window contains no frames")

        if _can_pass_video_through(probe, source_bytes, start_time, active_duration):
            data = await deadline.run(asyncio.to_thread(Path(source_path).read_bytes))
            return PreparedMedia(
                modality="video",
                mime_type="video/mp4",
                data=data,
                source_bytes=source_bytes,
                details={
                    "duration_seconds": round(probe["duration"], 3),
                    "width": probe["width"],
                    "height": probe["height"],
                    "fps": round(probe["fps"], 3),
                    "video_bitrate": probe["video_bitrate"],
                    "audio_bitrate": probe["audio_bitrate"],
                    "resampler": "none",
                    "transcode": "none",
                },
            )

        target_bytes = min(VIDEO_LIMIT, source_bytes)
        total_bitrate = int(target_bytes * 8 * 0.92 / duration)
        audio_bitrate = (
            min(128_000, max(32_000, int(total_bitrate * 0.12)))
            if probe["audio_codec"]
            else 0
        )
        video_bitrate = total_bitrate - audio_bitrate
        if video_bitrate < 100_000:
            raise ValueError(
                "video cannot fit the non-expanding byte target at the minimum safe bitrate"
            )
        width, height, fps = _video_geometry(
            probe["width"], probe["height"], probe["fps"], video_bitrate
        )
        output = temp / "output.mp4"
        for attempt in range(2):
            await _encode_video_passes(
                deadline,
                ffmpeg,
                source_path,
                output,
                temp / f"pass-{attempt}",
                video_bitrate=video_bitrate,
                audio_bitrate=audio_bitrate,
                width=width,
                height=height,
                fps=fps,
                start_time=start_time,
                duration=active_duration,
            )
            size = output.stat().st_size
            if size <= target_bytes:
                data = await deadline.run(asyncio.to_thread(output.read_bytes))
                return PreparedMedia(
                    modality="video",
                    mime_type="video/mp4",
                    data=data,
                    source_bytes=source_bytes,
                    details={
                        "duration_seconds": round(duration, 3),
                        "width": width,
                        "height": height,
                        "fps": round(fps, 3),
                        "video_bitrate": video_bitrate,
                        "audio_bitrate": audio_bitrate,
                        "resampler": "ffmpeg Lanczos" if (width, height) != (probe["width"], probe["height"]) else "none",
                        "transcode": "H.264/AAC",
                    },
                )
            ratio = target_bytes / max(size, 1) * 0.90
            video_bitrate = int(video_bitrate * ratio)
            if video_bitrate < 100_000:
                break
            width, height, fps = _video_geometry(probe["width"], probe["height"], probe["fps"], video_bitrate)
        raise ValueError(
            f"video encoder could not produce a result below {target_bytes} bytes"
        )


def _write_audio_wav(audio: dict[str, Any], path: Path) -> tuple[int, float, int, int]:
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate") or 0)
    if waveform is None or sample_rate <= 0 or getattr(waveform, "ndim", None) != 3 or int(waveform.shape[0]) != 1:
        raise ValueError("AUDIO input must contain waveform [1, channels, samples] and sample_rate")
    pcm = waveform[0].detach().cpu().float().clamp(-1, 1).numpy()
    channels, samples = int(pcm.shape[0]), int(pcm.shape[1])
    if channels < 1 or samples < 1:
        raise ValueError("AUDIO input is empty")
    interleaved = (np.transpose(pcm, (1, 0)) * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(interleaved.tobytes())
    return int(pcm.nbytes), samples / sample_rate, sample_rate, channels


async def prepare_audio(deadline: NodeDeadline, audio: dict[str, Any]) -> PreparedMedia:
    ffmpeg, _ = _require_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="openrouter-simple-audio-") as temp_dir:
        temp = Path(temp_dir)
        source = temp / "input.wav"
        source_bytes, duration, sample_rate, channels = await deadline.run(
            asyncio.to_thread(_write_audio_wav, audio, source)
        )
        target_kbps = min(320, int(AUDIO_LIMIT * 8 * 0.92 / max(duration, 0.001) / 1000))
        if target_kbps < 16:
            raise ValueError("audio is too long to fit below 1 MB at the minimum safe bitrate")
        output = temp / "output.mp3"
        resample_filter = "aresample=resampler=soxr:precision=28"
        resampler_name = "SoXr"

        def audio_command(filter_value: str) -> tuple[str, ...]:
            return (
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-xerror",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                str(min(channels, 2)),
                "-ar",
                str(min(sample_rate, 48_000)),
                "-af",
                filter_value,
                "-c:a",
                "libmp3lame",
                "-b:a",
                f"{target_kbps}k",
                str(output),
            )

        for _ in range(2):
            try:
                await run_process(deadline, *audio_command(resample_filter))
            except RuntimeError as exc:
                if resampler_name != "SoXr" or "resampling engine is unavailable" not in str(exc):
                    raise
                resample_filter = (
                    "aresample=resampler=swr:filter_size=64:phase_shift=10:"
                    "linear_interp=0:exact_rational=1"
                )
                resampler_name = "FFmpeg SWR high precision"
                await run_process(deadline, *audio_command(resample_filter))
            size = output.stat().st_size
            if size <= AUDIO_LIMIT:
                data = await deadline.run(asyncio.to_thread(output.read_bytes))
                return PreparedMedia(
                    modality="audio",
                    mime_type="audio/mpeg",
                    data=data,
                    source_bytes=source_bytes,
                    details={
                        "duration_seconds": round(duration, 3),
                        "bitrate_kbps": target_kbps,
                        "sample_rate": min(sample_rate, 48_000),
                        "channels": min(channels, 2),
                        "resampler": resampler_name,
                    },
                )
            target_kbps = int(target_kbps * AUDIO_LIMIT / max(size, 1) * 0.88)
            if target_kbps < 16:
                break
        raise ValueError("audio encoder could not produce a result below 1 MB")
