"""Baseball: Major League Baseball (Retrosheet data)."""

from .plugin import BaseballPlugin

PLUGIN = BaseballPlugin()

__all__ = ["PLUGIN", "BaseballPlugin"]
