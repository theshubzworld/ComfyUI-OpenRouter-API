from __future__ import annotations

from .models import CATALOG


def register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    @PromptServer.instance.routes.get("/openrouter_simple/models")
    async def openrouter_simple_models(_request):
        try:
            snapshot = await CATALOG.get()
            return web.json_response(snapshot.public())
        except Exception as exc:
            return web.json_response(
                {"models": [], "fetched_at": 0, "stale": False, "warning": str(exc)},
                status=503,
            )
