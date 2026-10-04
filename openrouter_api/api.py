from __future__ import annotations

from .models import CATALOG


def register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    @PromptServer.instance.routes.get("/openrouter_api/models")
    async def openrouter_api_models(request):
        try:
            force_refresh = request.query.get("refresh", "").lower() in ("true", "1", "yes")
            snapshot = await CATALOG.get(force=force_refresh)
            data = snapshot.public()
            free_models = [m["id"] for m in data.get("models", []) if m.get("is_free") or ":free" in m.get("id", "")]
            data["free_models"] = free_models
            data["free_count"] = len(free_models)
            return web.json_response(data)
        except Exception as exc:
            return web.json_response(
                {"models": [], "fetched_at": 0, "stale": False, "warning": str(exc)},
                status=503,
            )

    @PromptServer.instance.routes.get("/openrouter_api/presets")
    async def openrouter_api_presets(_request):
        from .minimax_prompts import _PRESET_MAP, DEFAULT_SYSTEM_PRESET, SYSTEM_PRESETS
        return web.json_response({
            "presets": list(SYSTEM_PRESETS),
            "default": DEFAULT_SYSTEM_PRESET,
            "prompts": _PRESET_MAP,
        })
