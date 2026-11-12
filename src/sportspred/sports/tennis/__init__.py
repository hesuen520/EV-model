"""Tennis: ATP / WTA tour level, qualifying, Challenger / WTA 125 and ITF singles."""

from .plugin import TennisPlugin

PLUGIN = TennisPlugin()

__all__ = ["PLUGIN", "TennisPlugin"]
