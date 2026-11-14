"""Schedule context: rest, congestion and travel, known before an event starts.

All inputs are the schedule itself (start times and venues), which is public
in advance, so these features are point-in-time by construction.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["haversine_km", "team_schedule_features"]


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometres (NaN where a coordinate is missing)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def team_schedule_features(
    events: pd.DataFrame,
    *,
    venue_coords: Optional[Dict[str, Tuple[float, float]]] = None,
    home_coords: Optional[Dict[str, Tuple[float, float]]] = None,
    rest_cap_days: float = 7.0,
) -> pd.DataFrame:
    """Per-side rest and travel for two-sided events.

    ``events`` needs ``event_id, start_time_utc, season, home_id, away_id`` and
    optionally ``neutral_site`` and ``lat``/``lon``.  When ``lat``/``lon`` are
    given they locate each event.  Otherwise a game's location is the home
    side's base (``home_coords[home_id]``) unless it is a neutral-site game,
    where the location is unknown (travel is then NaN, not guessed).  Rest is measured
    from the side's previous scheduled start in the same season; the first
    game of a season gets the cap.

    Returns one row per event with, for each side ``a`` (home) and ``b``
    (away): ``rest_days_*``, ``b2b_*`` (previous game started under 36 hours
    earlier), ``games_7d_*``, ``travel_km_*`` (from the previous game's
    location), ``tz_shift_*`` (degrees of longitude / 15, eastward positive).
    """
    df = events.sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
    home_coords = home_coords or {}
    neutral = df["neutral_site"].fillna(0).astype(bool) if "neutral_site" in df else pd.Series(False, index=df.index)
    if {"lat", "lon"} <= set(df.columns):
        # The caller located each event itself (e.g. from the venue city).
        loc_lat = df["lat"].to_numpy(dtype=float)
        loc_lon = df["lon"].to_numpy(dtype=float)
    else:
        loc_lat = np.array([np.nan if n else home_coords.get(h, (np.nan, np.nan))[0]
                            for h, n in zip(df["home_id"], neutral)])
        loc_lon = np.array([np.nan if n else home_coords.get(h, (np.nan, np.nan))[1]
                            for h, n in zip(df["home_id"], neutral)])

    long = pd.concat([
        pd.DataFrame({"row": df.index, "team": df["home_id"], "side": "a"}),
        pd.DataFrame({"row": df.index, "team": df["away_id"], "side": "b"}),
    ], ignore_index=True)
    long["t"] = df["start_time_utc"].to_numpy()[long["row"]]
    long["season"] = df["season"].to_numpy()[long["row"]]
    long["lat"] = loc_lat[long["row"]]
    long["lon"] = loc_lon[long["row"]]
    long = long.sort_values(["team", "t", "row"]).reset_index(drop=True)

    same = (long["team"] == long["team"].shift()) & (long["season"] == long["season"].shift())
    prev_t = long["t"].shift().where(same)
    gap_h = (long["t"] - prev_t).dt.total_seconds() / 3600.0
    long["rest_days"] = (gap_h / 24.0).clip(upper=rest_cap_days).fillna(rest_cap_days)
    long["b2b"] = (gap_h < 36.0).astype(float)
    prev_lat = long["lat"].shift().where(same)
    prev_lon = long["lon"].shift().where(same)
    # A side's first game of the season starts from its own base.
    base_lat = long["team"].map(lambda t: home_coords.get(t, (np.nan, np.nan))[0])
    base_lon = long["team"].map(lambda t: home_coords.get(t, (np.nan, np.nan))[1])
    prev_lat = prev_lat.where(same, base_lat)
    prev_lon = prev_lon.where(same, base_lon)
    long["travel_km"] = haversine_km(prev_lat, prev_lon, long["lat"], long["lon"])
    long["tz_shift"] = (long["lon"] - prev_lon) / 15.0

    # Games started in the previous 7 days (excluding this one), per team.
    counts = np.zeros(len(long))
    t_ns = long["t"].astype("int64").to_numpy()
    week = np.int64(7 * 24 * 3600) * np.int64(10 ** 9 if str(long["t"].dtype).endswith("ns, UTC]") else 10 ** 6)
    start = 0
    teams = long["team"].to_numpy()
    for i in range(len(long)):
        if i and teams[i] != teams[i - 1]:
            start = i
        while t_ns[i] - t_ns[start] > week:
            start += 1
        counts[i] = i - start
    long["games_7d"] = counts

    out = pd.DataFrame({"event_id": df["event_id"]})
    for side in ("a", "b"):
        part = long[long["side"] == side].set_index("row")
        for col in ("rest_days", "b2b", "games_7d", "travel_km", "tz_shift"):
            out[f"{col}_{side}"] = part[col].reindex(df.index).to_numpy()
    return out
