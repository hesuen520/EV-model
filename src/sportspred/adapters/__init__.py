"""Data adapters: cached downloads, official feeds and documented CSV imports."""

from .http import Fetched, Fetcher, NotFound

__all__ = ["Fetched", "Fetcher", "NotFound"]
