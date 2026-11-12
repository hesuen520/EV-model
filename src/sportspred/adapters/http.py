"""Cached, retried downloads with provenance.

Every download records its URL, collection time, publication time (the
server's ``Last-Modified`` when sent) and content hash next to the cached
file, so the store can attribute every row to a source.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

import requests

from ..core.timeutil import UTC, age_hours, iso_z, utcnow
from ..sports import DataUnavailable

__all__ = ["Fetcher", "Fetched", "NotFound"]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


class NotFound(DataUnavailable):
    """The source answered that this file does not exist (HTTP 404)."""


@dataclass
class Fetched:
    path: Path
    url: str
    provider: str
    sha256: str
    n_bytes: int
    collected_at_utc: str
    published_at_utc: Optional[str]
    from_cache: bool
    stale: bool = False            # a refresh was wanted but failed; the cached copy is being used
    license: Optional[str] = None

    def register(self, store: Any, *, notes: Optional[str] = None, is_synthetic: bool = False) -> str:
        """Record this download in the store; returns the ``source_id``."""
        return store.add_source(
            self.provider, self.url, collected_at=self.collected_at_utc,
            published_at=self.published_at_utc, sha256=self.sha256, n_bytes=self.n_bytes,
            license=self.license, notes=notes, is_synthetic=is_synthetic,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Fetcher:
    """Download files into ``cache_dir`` and reuse them.

    ``refresh`` on :meth:`fetch`: ``None`` re-downloads only when the cached
    copy is older than ``max_age_hours`` (never, if that is ``None``);
    ``True`` always downloads; ``False`` never touches the network.
    With ``offline=True`` the fetcher behaves as if ``refresh=False`` always.
    """

    def __init__(self, cache_dir: PathLike = "data/raw", *, timeout: float = 120.0, retries: int = 4,
                 backoff: float = 2.0, offline: bool = False,
                 session: Optional[requests.Session] = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.offline = offline or os.environ.get("SPORTSPRED_OFFLINE") == "1"
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "sportspred/0.2 (research client)")

    def _paths(self, provider: str, url: str, relpath: Optional[str]) -> tuple[Path, Path]:
        if relpath is None:
            parsed = urlparse(url)
            relpath = (parsed.netloc + parsed.path).strip("/")
        path = self.cache_dir / provider / relpath
        return path, path.with_name(path.name + ".meta.json")

    def fetch(self, url: str, *, provider: str, relpath: Optional[str] = None,
              max_age_hours: Optional[float] = None, refresh: Optional[bool] = None,
              license: Optional[str] = None) -> Fetched:
        path, meta_path = self._paths(provider, url, relpath)
        cached = path.exists() and path.stat().st_size > 0 and meta_path.exists()
        meta = json.loads(meta_path.read_text()) if cached else {}
        if self.offline:
            refresh = False
        want_download = (
            refresh is True
            or not cached
            or (refresh is None and max_age_hours is not None
                and age_hours(meta.get("collected_at_utc", "1970-01-01T00:00:00Z")) > max_age_hours)
        )
        if refresh is False:
            want_download = False
        if not want_download:
            if not cached:
                raise DataUnavailable(
                    f"{url} is not in the local cache ({path}) and network access is disabled"
                )
            return self._from_meta(path, meta, provider, url, license, from_cache=True)
        try:
            self._download(url, path, meta_path, provider)
        except NotFound:
            raise
        except DataUnavailable as exc:
            if cached:
                log.warning("refresh of %s failed (%s); using the cached copy collected %s",
                            url, exc, meta.get("collected_at_utc"))
                out = self._from_meta(path, meta, provider, url, license, from_cache=True)
                out.stale = True
                return out
            raise
        meta = json.loads(meta_path.read_text())
        return self._from_meta(path, meta, provider, url, license, from_cache=False)

    @staticmethod
    def _from_meta(path: Path, meta: dict, provider: str, url: str, license: Optional[str], *,
                   from_cache: bool) -> Fetched:
        return Fetched(
            path=path, url=url, provider=provider, sha256=meta["sha256"], n_bytes=int(meta["n_bytes"]),
            collected_at_utc=meta["collected_at_utc"], published_at_utc=meta.get("published_at_utc"),
            from_cache=from_cache, license=license,
        )

    def _download(self, url: str, path: Path, meta_path: Path, provider: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        last_error: Optional[str] = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.backoff ** attempt)
            try:
                with self.session.get(url, stream=True, timeout=self.timeout) as resp:
                    if resp.status_code == 404:
                        raise NotFound(f"no file at {url} (HTTP 404)")
                    if resp.status_code in (401, 403, 407):
                        raise DataUnavailable(
                            f"{url} refused the request (HTTP {resp.status_code}); the host may need "
                            "credentials or be blocked by the network policy"
                        )
                    if resp.status_code == 429 or resp.status_code >= 500:
                        last_error = f"HTTP {resp.status_code}"
                        continue
                    resp.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 16):
                            fh.write(chunk)
                    published = None
                    if resp.headers.get("Last-Modified"):
                        try:
                            published = iso_z(parsedate_to_datetime(resp.headers["Last-Modified"]).astimezone(UTC))
                        except (TypeError, ValueError):
                            published = None
                os.replace(tmp, path)
                meta = {
                    "url": url, "provider": provider, "sha256": _sha256(path),
                    "n_bytes": path.stat().st_size, "collected_at_utc": iso_z(utcnow()),
                    "published_at_utc": published,
                }
                meta_path.write_text(json.dumps(meta, indent=1))
                return
            except NotFound:
                raise
            except DataUnavailable:
                raise
            except requests.exceptions.ProxyError as exc:
                raise DataUnavailable(
                    f"{url} is not reachable: the network policy blocked the connection ({exc.__class__.__name__})"
                ) from exc
            except requests.RequestException as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
        raise DataUnavailable(f"could not download {url} after {self.retries + 1} attempts ({last_error})")
