from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

from .cancellation import NodeDeadline
from .http import read_bounded

OPENROUTER_URL = "https://openrouter.ai/api/v1"


def api_base_url() -> str:
    return os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_URL).rstrip("/")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    supported_parameters: tuple[str, ...]
    reasoning: bool
    is_free: bool = False

    def accepts(self, required: set[str]) -> bool:
        return required.issubset(set(self.input_modalities)) and "text" in self.output_modalities

    def supports(self, parameter: str) -> bool:
        return parameter in self.supported_parameters


FALLBACK_SEED_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("qwen/qwen3.8-27b:free", "Qwen: Qwen3.8 27B (free)", ("image", "text", "video"), ("text",), ("max_tokens", "temperature", "reasoning"), True, True),
    ModelInfo("google/gemini-2.5-flash", "Google: Gemini 2.5 Flash", ("image", "text", "video", "audio"), ("text",), ("max_tokens", "temperature", "reasoning"), True, False),
    ModelInfo("openai/gpt-4o-mini", "OpenAI: GPT-4o-mini", ("image", "text"), ("text",), ("max_tokens", "temperature"), False, False),
    ModelInfo("anthropic/claude-3.5-sonnet", "Anthropic: Claude 3.5 Sonnet", ("image", "text"), ("text",), ("max_tokens", "temperature"), False, False),
    ModelInfo("deepseek/deepseek-chat", "DeepSeek: DeepSeek V3", ("text",), ("text",), ("max_tokens", "temperature"), False, False),
)


@dataclass(frozen=True)
class ModelSnapshot:
    models: tuple[ModelInfo, ...]
    fetched_at: float
    stale: bool = False
    warning: str | None = None

    def by_id(self, model_id: str) -> ModelInfo | None:
        return next((model for model in self.models if model.id == model_id), None)

    def free_models(self) -> tuple[ModelInfo, ...]:
        return tuple(m for m in self.models if m.is_free or ":free" in m.id)

    def find_model(self, model_id: str) -> ModelInfo | None:
        clean = (model_id or "").strip()
        if not clean:
            return None
        exact = self.by_id(clean)
        if exact is not None:
            return exact
        clean_lower = clean.lower()
        for model in self.models:
            if model.id.lower() == clean_lower:
                return model
        # Token-based match (e.g. 'qwen 27b' -> 'qwen/qwen3.8-27b:free')
        tokens = [t for t in re.split(r"[ /:_\-]+", clean_lower) if t]
        if tokens:
            matches = [m for m in self.models if all(t in m.id.lower() for t in tokens)]
            if len(matches) == 1:
                return matches[0]
            if matches and any("free" in m.id.lower() for m in matches):
                free_matches = [m for m in matches if "free" in m.id.lower()]
                if len(free_matches) == 1:
                    return free_matches[0]
            if matches:
                return matches[0]
        return None

    def public(self) -> dict[str, Any]:
        return {
            "models": [asdict(model) for model in self.models],
            "free_count": sum(1 for m in self.models if m.is_free or ":free" in m.id),
            "total_count": len(self.models),
            "fetched_at": self.fetched_at,
            "stale": self.stale,
            "warning": self.warning,
        }


def _cache_path() -> Path:
    configured = os.environ.get("OPENROUTER_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser()
    try:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / "openrouter_api" / "models.json"
    except (ImportError, AttributeError):
        return Path.home() / ".cache" / "comfyui-openrouter-simple" / "models.json"


def _bundled_cache_path() -> Path:
    return Path(__file__).resolve().parent / "models.json"


def _modalities(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(sorted({str(item).strip().lower() for item in value if str(item).strip()}))


def normalize_model(raw: dict[str, Any]) -> ModelInfo | None:
    model_id = str(raw.get("id") or "").strip()
    architecture = raw.get("architecture") if isinstance(raw.get("architecture"), dict) else {}
    inputs = _modalities(architecture.get("input_modalities"))
    outputs = _modalities(architecture.get("output_modalities"))
    if not model_id or "text" not in outputs:
        return None
    parameters = raw.get("supported_parameters")
    supported = tuple(sorted({str(item) for item in parameters or [] if isinstance(item, str)}))
    reasoning_meta = raw.get("reasoning")
    reasoning = bool(reasoning_meta) or "reasoning" in supported
    pricing = raw.get("pricing") if isinstance(raw.get("pricing"), dict) else {}
    prompt_p = str(pricing.get("prompt", "1")).strip()
    comp_p = str(pricing.get("completion", "1")).strip()
    is_free = ":free" in model_id or (prompt_p in {"0", "0.0"} and comp_p in {"0", "0.0"})
    return ModelInfo(
        id=model_id,
        name=str(raw.get("name") or model_id),
        input_modalities=inputs or ("text",),
        output_modalities=outputs,
        supported_parameters=supported,
        reasoning=reasoning,
        is_free=is_free,
    )


class ModelCatalog:
    def __init__(self, *, ttl_seconds: float = 3600):
        self.ttl_seconds = ttl_seconds
        self._snapshot: ModelSnapshot | None = None
        self._lock = asyncio.Lock()
        self._load_disk()

    def _load_disk(self) -> None:
        candidates = [_cache_path(), _bundled_cache_path()]
        try:
            import folder_paths
            candidates.append(Path(folder_paths.get_user_directory()) / "openrouter_api" / "models.json")
        except Exception:
            pass
        for path in candidates:
            try:
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                models = tuple(ModelInfo(**item) for item in payload.get("models", []))
                if models:
                    self._snapshot = ModelSnapshot(models=models, fetched_at=float(payload.get("fetched_at", 0)), stale=True)
                    return
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        if self._snapshot is None:
            self._snapshot = ModelSnapshot(models=FALLBACK_SEED_MODELS, fetched_at=0, stale=True, warning="Initial seed catalog")

    def cached_ids(self, free_first: bool = True) -> list[str]:
        if not self._snapshot:
            return []
        if not free_first:
            return [model.id for model in self._snapshot.models]
        free = [model.id for model in self._snapshot.models if model.is_free or ":free" in model.id]
        paid = [model.id for model in self._snapshot.models if not (model.is_free or ":free" in model.id)]
        return [*free, *paid]

    def cached_free_ids(self) -> list[str]:
        if not self._snapshot:
            return []
        return [model.id for model in self._snapshot.models if model.is_free or ":free" in model.id]

    def _save_disk(self, snapshot: ModelSnapshot) -> None:
        path = _cache_path()
        payload = {"fetched_at": snapshot.fetched_at, "models": [asdict(model) for model in snapshot.models]}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            return

    async def _fetch(self, timeout_seconds: float) -> ModelSnapshot:
        timeout = aiohttp.ClientTimeout(total=max(0.1, min(timeout_seconds, 5.0)))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{api_base_url()}/models", params={"output_modalities": "text"}) as response:
                body = await read_bounded(response.content, 2_000_000, label="OpenRouter model catalog")
                if response.status >= 400:
                    raise RuntimeError(f"OpenRouter model catalog returned HTTP {response.status}")
        payload = json.loads(body)
        raw_models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(raw_models, list):
            raise RuntimeError("OpenRouter model catalog response did not contain a data list")
        models = tuple(sorted(filter(None, (normalize_model(item) for item in raw_models if isinstance(item, dict))), key=lambda item: item.id))
        if not models:
            raise RuntimeError("OpenRouter model catalog contained no text-output models")
        return ModelSnapshot(models=models, fetched_at=time.time())

    async def get(self, *, deadline: NodeDeadline | None = None, force: bool = False) -> ModelSnapshot:
        current = self._snapshot
        if current and not force and time.time() - current.fetched_at < self.ttl_seconds:
            return current
        async with self._lock:
            current = self._snapshot
            if current and not force and time.time() - current.fetched_at < self.ttl_seconds:
                return current
            try:
                if deadline is None:
                    snapshot = await asyncio.wait_for(self._fetch(5.0), timeout=5.2)
                else:
                    snapshot = await deadline.run(self._fetch(min(5.0, deadline.remaining)))
                self._snapshot = snapshot
                if deadline is None:
                    await asyncio.to_thread(self._save_disk, snapshot)
                else:
                    await deadline.run(asyncio.to_thread(self._save_disk, snapshot))
                return snapshot
            except Exception as exc:
                if deadline is not None:
                    deadline.checkpoint()
                if current:
                    return ModelSnapshot(
                        models=current.models,
                        fetched_at=current.fetched_at,
                        stale=True,
                        warning=f"Using cached model metadata: {type(exc).__name__}",
                    )
                raise RuntimeError("OpenRouter model metadata is unavailable; no cached catalog exists") from exc


CATALOG = ModelCatalog()
