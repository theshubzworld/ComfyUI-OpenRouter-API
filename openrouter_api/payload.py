from __future__ import annotations

from typing import Any

from .media import PreparedMedia
from .models import ModelInfo


def _user_content(user_prompt: str, media: list[PreparedMedia]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for item in media:
        if item.modality == "image":
            content.append({"type": "image_url", "image_url": {"url": item.data_url}})
        elif item.modality == "video":
            content.append({"type": "video_url", "video_url": {"url": item.data_url}})
        elif item.modality == "audio":
            content.append({"type": "input_audio", "input_audio": {"data": item.base64, "format": "mp3"}})
        else:
            raise ValueError(f"unsupported prepared modality: {item.modality}")
    return content


def build_payload(
    *,
    model: ModelInfo,
    system_prompt: str,
    user_prompt: str,
    media: list[PreparedMedia],
    reasoning_effort: str,
    temperature: float,
    max_tokens: int,
    response_format: str,
    zdr: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _user_content(user_prompt, media)})
    payload: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "modalities": ["text"],
    }
    applied: dict[str, Any] = {"modalities": ["text"]}
    omitted: dict[str, str] = {}

    if model.supports("temperature"):
        payload["temperature"] = float(temperature)
        applied["temperature"] = float(temperature)
    else:
        omitted["temperature"] = "selected model does not advertise temperature"

    if model.supports("max_completion_tokens"):
        payload["max_completion_tokens"] = int(max_tokens)
        applied["max_tokens"] = {"wire_parameter": "max_completion_tokens", "value": int(max_tokens)}
    elif model.supports("max_tokens"):
        payload["max_tokens"] = int(max_tokens)
        applied["max_tokens"] = {"wire_parameter": "max_tokens", "value": int(max_tokens)}
    else:
        omitted["max_tokens"] = "selected model advertises neither output-budget parameter"

    if reasoning_effort != "auto":
        if not (model.reasoning or model.supports("reasoning")):
            raise ValueError(f"model '{model.id}' does not advertise reasoning controls")
        if reasoning_effort == "none":
            payload["reasoning"] = {"enabled": False}
        else:
            payload["reasoning"] = {"effort": reasoning_effort}
        applied["reasoning_effort"] = reasoning_effort

    if response_format == "json_object":
        if not (model.supports("response_format") or model.supports("structured_outputs")):
            raise ValueError(f"model '{model.id}' does not advertise JSON response formatting")
        payload["response_format"] = {"type": "json_object"}
        applied["response_format"] = "json_object"
    elif response_format != "text":
        raise ValueError(f"unsupported response format: {response_format}")

    if zdr:
        payload["provider"] = {"zdr": True}
        applied["zdr"] = True

    return payload, {"applied": applied, "omitted": omitted}
