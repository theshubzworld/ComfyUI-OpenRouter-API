from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from .cancellation import NodeDeadline, NodeTimeoutError
from .http import ResponseTooLarge, read_bounded
from .models import api_base_url

MAX_ERROR_BYTES = 16_384


class OpenRouterRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResult:
    text: str
    response_id: str | None
    usage: dict[str, Any]


def load_dotenv() -> None:
    """Loads environment variables from local .env files if not already set."""
    if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_KEY"):
        return
    from pathlib import Path
    candidates = [
        Path(__file__).resolve().parents[1] / ".env",
        Path.cwd() / ".env",
    ]
    try:
        import folder_paths
        candidates.append(Path(folder_paths.base_path) / ".env")
        candidates.append(Path(folder_paths.get_user_directory()) / ".env")
    except Exception:
        pass
    for path in candidates:
        if path.is_file():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_KEY"):
                    break
            except Exception:
                pass


def resolve_generation_key(api_key: str = "") -> str:
    """Explicit node key wins; blank values fall back without changing the environment."""
    for value in (api_key, os.environ.get("OPENROUTER_API_KEY"), os.environ.get("LLM_KEY")):
        if value and value.strip():
            return value.strip()
    return ""


def sanitize_message(message: str) -> str:
    message = re.sub(
        r"(?i)data:[a-z0-9.+-]+/[a-z0-9.+-]+;base64,[a-z0-9+/=_-]+",
        "[REDACTED MEDIA]",
        message,
    )
    message = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", message)
    message = re.sub(r"sk-or-v1-[A-Za-z0-9]+", "[REDACTED]", message)
    words: list[str] = []
    for word in message.split():
        if "://" in word:
            try:
                parts = urlsplit(word.strip("'\"(),"))
                if parts.scheme and parts.netloc:
                    word = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
            except ValueError:
                pass
        words.append(word)
    return " ".join(words)[:2000]


def _friendly_status(status: int) -> str:
    return {
        400: "OpenRouter rejected the request as invalid",
        401: "OpenRouter authentication failed; check api_key or the configured environment key",
        402: "OpenRouter reported insufficient credits",
        403: "OpenRouter denied this model or request",
        408: "OpenRouter timed out while processing the request",
        413: "OpenRouter rejected the request because its encoded payload is too large",
        422: "OpenRouter could not process one of the supplied inputs",
        429: "OpenRouter rate-limited the request",
        524: "The upstream OpenRouter provider timed out",
        529: "The selected OpenRouter provider is overloaded",
    }.get(status, "OpenRouter request failed")


async def _limited_json(response: aiohttp.ClientResponse, limit: int) -> Any:
    try:
        body = await read_bounded(response.content, limit, label="OpenRouter response")
    except ResponseTooLarge as exc:
        raise OpenRouterRequestError(str(exc)) from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise OpenRouterRequestError("OpenRouter returned malformed JSON") from exc


def _provider_diagnostic(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return _provider_diagnostic(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return sanitize_message(stripped)
    if isinstance(value, dict):
        for key in ("error", "message", "detail", "error_description"):
            if key in value:
                detail = _provider_diagnostic(value[key])
                if detail:
                    return detail
    return ""


def _error_detail(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            parts: list[str] = []
            message = sanitize_message(str(error.get("message") or error.get("code") or ""))
            if message:
                parts.append(message)
            metadata = error.get("metadata")
            if isinstance(metadata, dict):
                provider = sanitize_message(str(metadata.get("provider_name") or ""))
                provider_code = sanitize_message(
                    str(metadata.get("provider_error_code") or metadata.get("error_code") or "")
                )
                raw = _provider_diagnostic(metadata.get("raw"))
                if provider:
                    parts.append(f"provider={provider}")
                if provider_code:
                    parts.append(f"provider_code={provider_code}")
                if raw and raw != message:
                    parts.append(f"upstream={raw}")
            router = payload.get("openrouter_metadata")
            if isinstance(router, dict):
                summary = sanitize_message(str(router.get("summary") or ""))
                if summary:
                    parts.append(f"routing={summary}")
            return sanitize_message("; ".join(dict.fromkeys(parts)))
        if error:
            return sanitize_message(str(error))
    return ""


def _strip_thinking(text: str) -> str:
    """Crops model internal thinking/reasoning tags from final output."""
    if not text:
        return ""
    # Strip <think>...</think>, <thought>...</thought>, and <reasoning>...</reasoning>
    cleaned = re.sub(r"(?is)<\s*(?:think|thought|reasoning)\s*>.*?<\s*/\s*(?:think|thought|reasoning)\s*>", "", text)
    # Strip unclosed thinking block at start (e.g. truncated thinking)
    cleaned = re.sub(r"(?is)^<\s*(?:think|thought|reasoning)\s*>.*$", "", cleaned)
    # Strip common standalone prefixes like "Thinking Process:\n" if followed by thinking
    cleaned = re.sub(r"(?i)^thinking process:\s*", "", cleaned)
    return cleaned.strip()


def _extract_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    raw_text = ""
    if isinstance(content, str) and content.strip():
        raw_text = content
    elif isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"} and isinstance(block.get("text"), str):
                chunks.append(block["text"])
        if chunks:
            raw_text = "".join(chunks)

    if raw_text:
        return _strip_thinking(raw_text)

    return ""


async def create_chat(deadline: NodeDeadline, payload: dict[str, Any], api_key: str) -> ChatResult:
    async def request() -> ChatResult:
        timeout = aiohttp.ClientTimeout(total=max(0.1, deadline.remaining))
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{api_base_url()}/chat/completions", headers=headers, json=payload) as response:
                if response.status >= 400:
                    try:
                        body = await read_bounded(response.content, MAX_ERROR_BYTES, label="OpenRouter error response")
                    except ResponseTooLarge:
                        detail = "response body omitted"
                    else:
                        try:
                            detail = _error_detail(json.loads(body))
                        except json.JSONDecodeError:
                            detail = sanitize_message(body.decode("utf-8", errors="replace"))
                    base = _friendly_status(response.status)
                    raise OpenRouterRequestError(f"{base} (HTTP {response.status})" + (f": {detail}" if detail else ""))
                data = await _limited_json(response, 2_000_000)
        choices = data.get("choices") if isinstance(data, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        text = _extract_text(message)
        if not text:
            finish_reason = first_choice.get("finish_reason")
            reasoning = message.get("reasoning") if isinstance(message, dict) else None
            if finish_reason == "length":
                raise OpenRouterRequestError(
                    "Model reached max_tokens limit during internal thinking before generating the prompt. "
                    "Increase max_tokens (e.g. 8192 or 16384) or set reasoning_effort to 'none' or 'low'."
                )
            refusal = message.get("refusal") if isinstance(message, dict) else None
            if refusal:
                raise OpenRouterRequestError(f"Model refused request: {refusal}")
            if reasoning:
                raise OpenRouterRequestError(
                    "Model produced only internal reasoning thoughts without a final prompt response. "
                    "Set reasoning_effort to 'none' or 'low', or choose another model."
                )
            raise OpenRouterRequestError("OpenRouter returned no text completion; media-only output is not accepted")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ChatResult(text=text, response_id=data.get("id"), usage=usage)

    try:
        return await deadline.run(request())
    except asyncio.TimeoutError as exc:
        raise NodeTimeoutError("OpenRouter node deadline expired") from exc


async def lookup_credits(deadline: NodeDeadline, generation_key: str) -> str:
    management_key = (os.environ.get("OPENROUTER_MANAGEMENT_KEY") or "").strip()
    key = management_key or generation_key
    endpoint = "/credits" if management_key else "/key"

    async def request() -> str:
        timeout = aiohttp.ClientTimeout(total=max(0.1, deadline.remaining))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{api_base_url()}{endpoint}", headers={"Authorization": f"Bearer {key}"}) as response:
                if response.status >= 400:
                    raise OpenRouterRequestError(f"credits lookup returned HTTP {response.status}")
                payload = await _limited_json(response, 64_000)
        data = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else {}
        if management_key:
            total = data.get("total_credits")
            used = data.get("total_usage")
            if isinstance(total, (int, float)) and isinstance(used, (int, float)):
                return f"Account credits remaining: ${max(0.0, total - used):.6f}"
            raise OpenRouterRequestError("credits response omitted total_credits or total_usage")
        remaining = data.get("limit_remaining")
        if isinstance(remaining, (int, float)):
            return f"API key limit remaining: ${remaining:.6f}"
        return "API key limit remaining: unlimited or not configured"

    try:
        return await deadline.run(request())
    except Exception as exc:
        return f"Credits unavailable ({sanitize_message(str(exc))})"
