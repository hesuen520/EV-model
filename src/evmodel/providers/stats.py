"""Player game logs from free, public, auto-updated datasets.

No API key is needed.  Files are downloaded once into a cache directory and
re-used; the current season's file is refreshed when it is more than a few
hours old.

==========  ===========================================================
sport       source
==========  ===========================================================
``nba``     sportsdataverse (ESPN box scores), one parquet per season
``nhl``     sportsdataverse (NHL box scores), 2023-24 season onwards
``nfl``     nflverse weekly player stats (CSV) + schedules + snap counts
==========  ===========================================================

Seasons are identified by the year they **start** in: ``2026`` is the NBA and
NHL 2026-27 season and the NFL 2026 season.

Every loader returns the normalised game-log table described in
:mod:`evmodel.schema`, with these stat columns:

* NBA: ``points rebounds assists threes steals blocks turnovers`` plus
  ``fga fg3a fta oreb dreb fouls``, and ``started``, ``position``.
* NHL: ``goals assists points shots_on_goal blocked_shots hits saves``.
* NFL: ``passing_yards passing_tds pass_attempts completions interceptions
  rushing_yards carries rushing_tds receptions targets receiving_yards
  receiving_tds`` plus ``week``, ``position`` and, when snap counts are
  available, ``snap_pct``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests

from ..names import player_key
from ..schema import normalize_gamelogs

__all__ = [
    "SUPPORTED_SPORTS",
    "DEFAULT_CACHE",
    "current_season",
    "season_file_url",
    "fetch_season_file",
    "load_gamelogs",
    "normalize_nba",
    "normalize_nhl",
    "normalize_nfl",
]

SUPPORTED_SPORTS = ("nba", "nhl", "nfl")
DEFAULT_CACHE = Path("data/raw")

_SDV = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
_NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
_NFL_GAMES_URL = f"{_NFLVERSE}/schedules/games.csv"

#: Re-download the current season's file when the cached copy is older than this.
_REFRESH_SECONDS = 6 * 3600

_PathLike = Union[str, Path]


def current_season(sport: str, today: Optional[pd.Timestamp] = None) -> int:
    """Starting year of the season in progress (or most recently finished)."""
    today = pd.Timestamp.now() if today is None else pd.Timestamp(today)
    sport = sport.lower()
    if sport in ("nba", "nhl"):
        return today.year if today.month >= 10 else today.year - 1
    if sport == "nfl":
        return today.year if today.month >= 9 else today.year - 1
    raise ValueError(f"unsupported sport {sport!r}; choose from {SUPPORTED_SPORTS}")


def season_file_url(sport: str, season: int, kind: str = "boxscores") -> str:
    """Download URL of a season file.  ``kind``: ``boxscores`` or (NFL) ``snaps``."""
    sport = sport.lower()
    if sport == "nba":
        return f"{_SDV}/espn_nba_player_boxscores/player_box_{season + 1}.parquet"
    if sport == "nhl":
        return f"{_SDV}/nhl_player_boxscores/player_box_{season + 1}.parquet"
    if sport == "nfl":
        if kind == "snaps":
            return f"{_NFLVERSE}/snap_counts/snap_counts_{season}.csv"
        return f"{_NFLVERSE}/stats_player/stats_player_week_{season}.csv"
    raise ValueError(f"unsupported sport {sport!r}; choose from {SUPPORTED_SPORTS}")


def _cache_path(sport: str, season: int, cache_dir: _PathLike, kind: str = "boxscores") -> Path:
    sport = sport.lower()
    base = Path(cache_dir)
    if sport == "nba":
        return base / "nba_sdv" / f"player_box_{season + 1}.parquet"
    if sport == "nhl":
        return base / "nhl_sdv" / f"player_box_{season + 1}.parquet"
    if kind == "snaps":
        return base / "nfl" / f"snap_counts_{season}.csv"
    return base / "nfl" / f"stats_player_week_{season}.csv"


def _download(url: str, dest: Path, timeout: float = 120.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        if response.status_code == 404:
            raise FileNotFoundError(f"no data file at {url}")
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
    os.replace(tmp, dest)
    return dest


def _stale(path: Path) -> bool:
    return (time.time() - path.stat().st_mtime) > _REFRESH_SECONDS


def fetch_season_file(
    sport: str,
    season: int,
    cache_dir: _PathLike = DEFAULT_CACHE,
    refresh: Optional[bool] = None,
    kind: str = "boxscores",
) -> Path:
    """Return the cached file for a season, downloading it if necessary.

    ``refresh=None`` (default) re-downloads only the current season when the
    cached copy is stale; ``True`` always downloads; ``False`` never does
    (raising ``FileNotFoundError`` when the file is not cached).  If a
    refresh fails but a cached copy exists, the cached copy is used.
    """
    path = _cache_path(sport, season, cache_dir, kind)
    exists = path.exists() and path.stat().st_size > 0
    if refresh is False:
        if not exists:
            raise FileNotFoundError(f"{path} is not cached and refresh=False")
        return path
    wanted = refresh is True or not exists or (season >= current_season(sport) and _stale(path))
    if not wanted:
        return path
    try:
        return _download(season_file_url(sport, season, kind), path)
    except (requests.RequestException, OSError):
        if exists:
            return path
        raise


def _fetch_nfl_games(cache_dir: _PathLike, refresh: Optional[bool]) -> Optional[Path]:
    path = Path(cache_dir) / "nfl" / "games.csv"
    exists = path.exists() and path.stat().st_size > 0
    if refresh is False:
        return path if exists else None
    if exists and not (refresh is True or _stale(path)):
        return path
    try:
        return _download(_NFL_GAMES_URL, path)
    except (requests.RequestException, OSError):
        return path if exists else None


# --------------------------------------------------------------------- NBA
def _minutes_to_float(series: pd.Series) -> pd.Series:
    """Minutes as a float from numbers or ``MM:SS`` strings."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype("string").str.strip()
    parts = text.str.split(":", n=1, expand=True)
    minutes = pd.to_numeric(parts[0], errors="coerce")
    if parts.shape[1] > 1:
        minutes = minutes + pd.to_numeric(parts[1], errors="coerce").fillna(0.0) / 60.0
    return minutes.astype(float)


def normalize_nba(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a sportsdataverse ``espn_nba_player_boxscores`` table."""
    df = raw
    # 2 = regular season, 3 = playoffs, 5 = play-in.  (1 = preseason, 4 = all-star.)
    df = df[df["season_type"].isin([2, 3, 5])]
    minutes = _minutes_to_float(df["minutes"])
    played = (~df["did_not_play"].fillna(False).astype(bool)) & (minutes > 0)
    df, minutes = df[played], minutes[played]
    out = pd.DataFrame(
        {
            "sport": "nba",
            "season": df["season"].astype(int) - 1,
            "game_id": df["game_id"].astype(str),
            "game_date": pd.to_datetime(df["game_date"]),
            "player_id": df["athlete_id"].astype("Int64").astype(str),
            "player": df["athlete_display_name"],
            "team": df["team_abbreviation"],
            "opponent": df["opponent_team_abbreviation"],
            "is_home": df["home_away"].astype(str).str.lower() == "home",
            "minutes": minutes,
            "points": df["points"],
            "rebounds": df["rebounds"],
            "assists": df["assists"],
            "threes": df["three_point_field_goals_made"],
            "steals": df["steals"],
            "blocks": df["blocks"],
            "turnovers": df["turnovers"],
            "fga": df["field_goals_attempted"],
            "fg3a": df["three_point_field_goals_attempted"],
            "fta": df["free_throws_attempted"],
            "oreb": df["offensive_rebounds"],
            "dreb": df["defensive_rebounds"],
            "fouls": df["fouls"],
            "started": df["starter"].fillna(False).astype(bool),
            "position": df["athlete_position_abbreviation"].fillna("").astype(str),
            "season_type": np.where(df["season_type"] == 2, "REG", "POST"),
        }
    )
    stats = ["points", "rebounds", "assists", "threes", "steals", "blocks", "turnovers",
             "fga", "fg3a", "fta", "oreb", "dreb", "fouls"]
    out = out.dropna(subset=["player", "points"])
    return normalize_gamelogs(out, stat_columns=stats)


# --------------------------------------------------------------------- NHL
def normalize_nhl(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a sportsdataverse ``nhl_player_boxscores`` table.

    Raises ``ValueError`` for the older files (before 2023-24) that carry no
    game ids or dates and so cannot be ordered in time.
    """
    required = {"game_id", "season", "game_date", "player_id", "player_name", "team_abbrev", "toi"}
    absent = sorted(required - set(raw.columns))
    if absent:
        raise ValueError(f"NHL box-score file lacks columns {absent}")
    df = raw.copy()
    df["game_id"] = df["game_id"].astype(str)
    # NHL game ids are YYYYTTNNNN with TT = 02 regular season, 03 playoffs.
    game_type = df["game_id"].str[4:6]
    df = df[game_type.isin(["02", "03"])]
    teams = df[["game_id", "team_abbrev", "home_away"]].drop_duplicates(["game_id", "team_abbrev"])
    other = teams.rename(columns={"team_abbrev": "opponent"})[["game_id", "opponent"]]
    pairs = teams.merge(other, on="game_id")
    pairs = pairs[pairs["team_abbrev"] != pairs["opponent"]][["game_id", "team_abbrev", "opponent"]]
    pairs = pairs.drop_duplicates(["game_id", "team_abbrev"])
    df = df.merge(pairs, on=["game_id", "team_abbrev"], how="left")
    minutes = _minutes_to_float(df["toi"])
    out = pd.DataFrame(
        {
            "sport": "nhl",
            "season": (df["season"].astype(int) // 10000),
            "game_id": df["game_id"],
            "game_date": pd.to_datetime(df["game_date"]),
            "player_id": df["player_id"].astype("Int64").astype(str),
            "player": df["player_name"],
            "team": df["team_abbrev"],
            "opponent": df["opponent"].fillna(""),
            "is_home": df["home_away"].astype(str).str.lower() == "home",
            "minutes": minutes,
            "goals": df["goals"],
            "assists": df["assists"],
            "points": df["points"],
            "shots_on_goal": df["shots_on_goal"],
            "blocked_shots": df["blocked_shots"],
            "hits": df["hits"],
            "saves": df["saves"],
            "position": df["position"].fillna("").astype(str),
            "season_type": np.where(df["game_id"].str[4:6] == "02", "REG", "POST"),
        }
    )
    out = out[out["minutes"] > 0]
    stats = ["goals", "assists", "points", "shots_on_goal", "blocked_shots", "hits", "saves"]
    return normalize_gamelogs(out, stat_columns=stats)


# --------------------------------------------------------------------- NFL
_NFL_STATS = {
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "pass_attempts": "attempts",
    "completions": "completions",
    "interceptions": "passing_interceptions",
    "rushing_yards": "rushing_yards",
    "carries": "carries",
    "rushing_tds": "rushing_tds",
    "receptions": "receptions",
    "targets": "targets",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
}


def _pseudo_date(season: pd.Series, week: pd.Series) -> pd.Series:
    """Approximate game date when no schedule is available (keeps ordering right)."""
    start = pd.to_datetime(season.astype(int).astype(str) + "-09-07")
    return start + pd.to_timedelta((week.astype(int) - 1) * 7, unit="D")


def normalize_nfl(
    raw: pd.DataFrame,
    games: Optional[pd.DataFrame] = None,
    snaps: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Normalise an nflverse ``stats_player_week`` table.

    ``games`` (nflverse schedules) supplies real game dates; ``snaps``
    (nflverse snap counts) adds ``snap_pct``, the share of offensive snaps.
    """
    df = raw.copy()
    if "team" not in df.columns and "recent_team" in df.columns:
        df = df.rename(columns={"recent_team": "team"})
    if "passing_interceptions" not in df.columns and "interceptions" in df.columns:
        df = df.rename(columns={"interceptions": "passing_interceptions"})
    if "game_id" not in df.columns:
        df["game_id"] = (
            df["season"].astype(str) + "_" + df["week"].astype(int).astype(str).str.zfill(2)
            + "_" + df["team"].astype(str) + "_" + df["opponent_team"].astype(str)
        )
        home = pd.Series(False, index=df.index)
    else:
        home = df["game_id"].astype(str).str.rsplit("_", n=1).str[-1] == df["team"].astype(str)

    game_date = _pseudo_date(df["season"], df["week"])
    if games is not None and "gameday" in games.columns:
        dates = pd.to_datetime(games.set_index("game_id")["gameday"], errors="coerce")
        mapped = df["game_id"].map(dates)
        game_date = mapped.where(mapped.notna(), game_date)

    out = pd.DataFrame(
        {
            "sport": "nfl",
            "season": df["season"].astype(int),
            "game_id": df["game_id"].astype(str),
            "game_date": game_date,
            "player_id": df["player_id"].astype(str),
            "player": df["player_display_name"],
            "team": df["team"],
            "opponent": df["opponent_team"],
            "is_home": home,
            "minutes": np.nan,
            "week": df["week"].astype(int),
            "position": df["position"].fillna("").astype(str),
            "season_type": df["season_type"].astype(str),
        }
    )
    for name, source in _NFL_STATS.items():
        out[name] = pd.to_numeric(df[source], errors="coerce").fillna(0.0) if source in df.columns else 0.0

    out = normalize_gamelogs(out, stat_columns=list(_NFL_STATS))
    if snaps is not None and not snaps.empty:
        s = snaps[["game_id", "player", "team", "offense_pct"]].copy()
        s["player_key"] = s["player"].map(player_key)
        s = s.drop_duplicates(["game_id", "team", "player_key"])
        out = out.merge(
            s[["game_id", "team", "player_key", "offense_pct"]].rename(columns={"offense_pct": "snap_pct"}),
            on=["game_id", "team", "player_key"],
            how="left",
        )
    return out


# ------------------------------------------------------------------ loader
def load_gamelogs(
    sport: str,
    seasons: Optional[Sequence[int]] = None,
    cache_dir: _PathLike = DEFAULT_CACHE,
    refresh: Optional[bool] = None,
    include_playoffs: bool = True,
) -> pd.DataFrame:
    """Load normalised game logs for the given seasons.

    ``seasons`` defaults to the current season and the one before it, which
    is what the projection engine needs.  See :func:`fetch_season_file` for
    ``refresh``.
    """
    sport = sport.lower()
    if sport not in SUPPORTED_SPORTS:
        raise ValueError(f"unsupported sport {sport!r}; choose from {SUPPORTED_SPORTS}")
    if seasons is None:
        now = current_season(sport)
        seasons = [now - 1, now]
    seasons = sorted({int(s) for s in seasons})

    frames: List[pd.DataFrame] = []
    games = None
    if sport == "nfl":
        games_path = _fetch_nfl_games(cache_dir, refresh)
        games = pd.read_csv(games_path, low_memory=False) if games_path is not None else None
    missing: List[int] = []
    for season in seasons:
        try:
            path = fetch_season_file(sport, season, cache_dir, refresh)
        except FileNotFoundError:
            missing.append(season)
            continue
        if sport == "nba":
            frames.append(normalize_nba(pd.read_parquet(path)))
        elif sport == "nhl":
            try:
                frames.append(normalize_nhl(pd.read_parquet(path)))
            except ValueError:
                missing.append(season)  # pre-2023 files have no dates
        else:
            snaps = None
            try:
                snaps = pd.read_csv(fetch_season_file(sport, season, cache_dir, refresh, kind="snaps"))
            except (FileNotFoundError, requests.RequestException, OSError):
                snaps = None
            frames.append(normalize_nfl(pd.read_csv(path, low_memory=False), games=games, snaps=snaps))
    if not frames:
        raise FileNotFoundError(f"no {sport} game-log files available for seasons {seasons}")
    logs = pd.concat(frames, ignore_index=True)
    if not include_playoffs and "season_type" in logs.columns:
        logs = logs[logs["season_type"] == "REG"]
    logs = logs.sort_values(["player_key", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    logs.attrs["missing_seasons"] = missing
    return logs
