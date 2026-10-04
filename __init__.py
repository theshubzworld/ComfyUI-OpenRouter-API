from .node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .openrouter_api.api import register_routes
from .openrouter_api.client import load_dotenv

WEB_DIRECTORY = "./web"

load_dotenv()
register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
