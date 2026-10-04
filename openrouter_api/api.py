from __future__ import annotations

from .models import CATALOG


def register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    @PromptServer.instance.routes.get("/openrouter_api/models")
    async def openrouter_api_models(_request):
        try:
            snapshot = await CATALOG.get()
            return web.json_response(snapshot.public())
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
