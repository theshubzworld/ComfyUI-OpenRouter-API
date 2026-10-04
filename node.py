from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from .openrouter_simple.cancellation import NodeDeadline, NodeTimeoutError
from .openrouter_simple.client import create_chat, lookup_credits, resolve_generation_key
from .openrouter_simple.media import PreparedMedia, prepare_audio, prepare_image, prepare_video
from .openrouter_simple.minimax_prompts import (
    DEFAULT_SYSTEM_PRESET,
    SYSTEM_PRESETS,
    build_effective_system_prompt,
)
from .openrouter_simple.models import CATALOG, ModelInfo
from .openrouter_simple.payload import build_payload

CHOOSE_MODEL = "— choose a compatible OpenRouter model —"
NO_MODEL = "— no compatible text-output model —"
MODEL_SENTINELS = {CHOOSE_MODEL, NO_MODEL, "Loading OpenRouter models…"}
MEDIA_SLOT_NAMES = (
    "image",
    "image_2",
    "image_3",
    "video",
    "video_2",
    "video_3",
    "audio",
    "audio_2",
    "audio_3",
)


async def _prepare_media(
    deadline: NodeDeadline,
    *,
    media_inputs: list[tuple[str, Any]],
) -> list[tuple[str, PreparedMedia]]:
    tasks: list[tuple[str, asyncio.Task[PreparedMedia]]] = []
    for slot_name, value in media_inputs:
        modality = slot_name.split("_", 1)[0]
        if modality == "image":
            awaitable = prepare_image(deadline, value)
        elif modality == "video":
            awaitable = prepare_video(deadline, value)
        elif modality == "audio":
            awaitable = prepare_audio(deadline, value)
        else:
            raise ValueError(f"unsupported media input slot: {slot_name}")
        task = asyncio.create_task(awaitable, name=f"openrouter-{slot_name.replace('_', '-')}")
        tasks.append((slot_name, task))
    try:
        prepared = await asyncio.gather(*(task for _name, task in tasks)) if tasks else []
        return [
            (slot_name, item)
            for (slot_name, _task), item in zip(tasks, prepared, strict=True)
        ]
    except BaseException:
        for _name, task in tasks:
            task.cancel()
        for _name, task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        raise


class OpenRouterSimple:
    """Bounded multimodal context in, text only out."""

    @classmethod
    def INPUT_TYPES(cls):
        cached = CATALOG.cached_ids()
        model_values = [CHOOSE_MODEL, *cached] if cached else ["Loading OpenRouter models…"]
        return {
            "required": {
                "model": (model_values, {"default": model_values[0]}),
                "reasoning_effort": (
                    ["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
                    {"default": "auto"},
                ),
                "timeout_seconds": (
                    "INT",
                    {"default": 120, "min": 1, "max": 3600, "step": 1, "display": "number"},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.01},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 4096, "min": 1, "max": 1_000_000, "step": 1, "display": "number"},
                ),
                "response_format": (["text", "json_object"], {"default": "text"}),
                "zdr": ("BOOLEAN", {"default": False}),
                "regenerate": ("BOOLEAN", {"default": True}),
                "system_preset": (
                    list(SYSTEM_PRESETS),
                    {"default": DEFAULT_SYSTEM_PRESET},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Additional custom system prompt (leave blank to use selected preset default)",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {"multiline": True, "default": "Describe the supplied context."},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "video": ("VIDEO",),
                "video_2": ("VIDEO",),
                "video_3": ("VIDEO",),
                "audio": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "OpenRouter API key. Leave blank to use OPENROUTER_API_KEY or LLM_KEY. Keys entered here are saved with the workflow.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "info", "credits")
    FUNCTION = "run"
    CATEGORY = "Shubz/OpenRouter"
    DESCRIPTION = "OpenRouter multimodal context node by Shubz with bounded preprocessing and text-only output."

    @classmethod
    def VALIDATE_INPUTS(cls, model: str):
        if not model or model in MODEL_SENTINELS or model.startswith("—"):
            return "Choose a compatible OpenRouter model before queueing"
        return True

    @classmethod
    def IS_CHANGED(cls, regenerate: bool = True, **_kwargs):
        return float("nan") if regenerate else False

    async def run(
        self,
        user_prompt: str,
        model: str,
        reasoning_effort: str,
        timeout_seconds: int,
        temperature: float,
        max_tokens: int,
        response_format: str,
        zdr: bool,
        regenerate: bool,
        system_preset: str = DEFAULT_SYSTEM_PRESET,
        system_prompt: str = "",
        image: Any | None = None,
        video: Any | None = None,
        audio: dict[str, Any] | None = None,
        image_2: Any | None = None,
        image_3: Any | None = None,
        video_2: Any | None = None,
        video_3: Any | None = None,
        audio_2: dict[str, Any] | None = None,
        audio_3: dict[str, Any] | None = None,
        api_key: str = "",
    ):
        api_key = resolve_generation_key(api_key)
        if not api_key:
            raise ValueError("Enter or connect api_key, or set OPENROUTER_API_KEY (or legacy LLM_KEY) in the ComfyUI environment")

        deadline = NodeDeadline(float(timeout_seconds))
        try:
            snapshot = await CATALOG.get(deadline=deadline)
            selected = snapshot.find_model(model)
            if selected is None and "/" in model:
                selected = ModelInfo(
                    id=model.strip(),
                    name=model.strip(),
                    input_modalities=("text", "image", "video", "audio"),
                    output_modalities=("text",),
                    supported_parameters=("temperature", "max_tokens", "max_completion_tokens", "reasoning"),
                    reasoning=True,
                )
            if selected is None:
                similar = [m.id for m in snapshot.models if any(part in m.id.lower() for part in model.lower().split())][:5]
                hint = f". Did you mean: {', '.join(repr(s) for s in similar)}?" if similar else ""
                raise ValueError(f"model '{model}' is absent from the current OpenRouter catalog{hint}")
            values = {
                "image": image,
                "image_2": image_2,
                "image_3": image_3,
                "video": video,
                "video_2": video_2,
                "video_3": video_3,
                "audio": audio,
                "audio_2": audio_2,
                "audio_3": audio_3,
            }
            media_inputs = [(name, values[name]) for name in MEDIA_SLOT_NAMES if values[name] is not None]
            required_modalities = {
                "text",
                *(name.split("_", 1)[0] for name, _value in media_inputs),
            }
            if not selected.accepts(required_modalities):
                required = ", ".join(sorted(required_modalities))
                raise ValueError(f"model '{model}' does not accept every connected modality ({required})")

            media = await _prepare_media(deadline, media_inputs=media_inputs)
            deadline.checkpoint()
            effective_system = build_effective_system_prompt(system_preset, system_prompt)
            payload, parameter_info = build_payload(
                model=selected,
                system_prompt=effective_system,
                user_prompt=user_prompt,
                media=[item for _name, item in media],
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                zdr=zdr,
            )
            result = await create_chat(deadline, payload, api_key)

            credits = "Credits not checked (node deadline exhausted)"
            if deadline.remaining > 0.1:
                credit_deadline = NodeDeadline(min(3.0, deadline.remaining))
                credits = await lookup_credits(credit_deadline, api_key)

            info = {
                "model": model,
                "system_preset": system_preset,
                "response_id": result.response_id,
                "usage": result.usage,
                "required_modalities": sorted(required_modalities),
                "media": {name: item.public_info() for name, item in media},
                "parameters": parameter_info,
                "model_catalog": {
                    "stale": snapshot.stale,
                    "warning": snapshot.warning,
                    "fetched_at": snapshot.fetched_at,
                },
                "elapsed_seconds": round(deadline.elapsed, 3),
            }
            return result.text, json.dumps(info, ensure_ascii=False, separators=(",", ":")), credits
        except NodeTimeoutError as exc:
            raise RuntimeError(f"OpenRouter node timed out after {timeout_seconds} seconds") from exc


NODE_CLASS_MAPPINGS = {"OpenRouterSimple": OpenRouterSimple}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenRouterSimple": "Shubz - OpenRouter Simple (Text Output)"}
