"""NFL franchise reference: abbreviations, names and stadium locations.

Public reference data only (who a team is and roughly where it plays), used
to keep one participant per franchise across relocations and to measure how
far the visiting side travels.  Coordinates are stadium-area precision.  A
site that is not listed yields a missing travel value rather than a guess.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

__all__ = ["franchise", "team_name", "team_coords", "site_coords", "NFL_TEAMS"]

_NAN = (float("nan"), float("nan"))

#: abbreviation used by nflverse -> (franchise abbreviation, name in that era, latitude, longitude)
NFL_TEAMS: Dict[str, Tuple[str, str, float, float]] = {
    "ARI": ("ARI", "Arizona Cardinals", 33.528, -112.263),
    "ATL": ("ATL", "Atlanta Falcons", 33.755, -84.401),
    "BAL": ("BAL", "Baltimore Ravens", 39.278, -76.623),
    "BUF": ("BUF", "Buffalo Bills", 42.774, -78.787),
    "CAR": ("CAR", "Carolina Panthers", 35.226, -80.853),
    "CHI": ("CHI", "Chicago Bears", 41.862, -87.617),
    "CIN": ("CIN", "Cincinnati Bengals", 39.095, -84.516),
    "CLE": ("CLE", "Cleveland Browns", 41.506, -81.700),
    "DAL": ("DAL", "Dallas Cowboys", 32.747, -97.094),
    "DEN": ("DEN", "Denver Broncos", 39.744, -105.020),
    "DET": ("DET", "Detroit Lions", 42.340, -83.046),
    "GB": ("GB", "Green Bay Packers", 44.501, -88.062),
    "HOU": ("HOU", "Houston Texans", 29.685, -95.411),
    "IND": ("IND", "Indianapolis Colts", 39.760, -86.164),
    "JAX": ("JAX", "Jacksonville Jaguars", 30.324, -81.637),
    "KC": ("KC", "Kansas City Chiefs", 39.049, -94.484),
    "LA": ("LA", "Los Angeles Rams", 33.953, -118.339),
    "LAC": ("LAC", "Los Angeles Chargers", 33.953, -118.339),
    "LV": ("LV", "Las Vegas Raiders", 36.091, -115.183),
    "MIA": ("MIA", "Miami Dolphins", 25.958, -80.239),
    "MIN": ("MIN", "Minnesota Vikings", 44.974, -93.258),
    "NE": ("NE", "New England Patriots", 42.091, -71.264),
    "NO": ("NO", "New Orleans Saints", 29.951, -90.081),
    "NYG": ("NYG", "New York Giants", 40.813, -74.074),
    "NYJ": ("NYJ", "New York Jets", 40.813, -74.074),
    "PHI": ("PHI", "Philadelphia Eagles", 39.901, -75.168),
    "PIT": ("PIT", "Pittsburgh Steelers", 40.447, -80.016),
    "SEA": ("SEA", "Seattle Seahawks", 47.595, -122.332),
    "SF": ("SF", "San Francisco 49ers", 37.403, -121.969),
    "TB": ("TB", "Tampa Bay Buccaneers", 27.976, -82.503),
    "TEN": ("TEN", "Tennessee Titans", 36.166, -86.771),
    "WAS": ("WAS", "Washington Commanders", 38.908, -76.864),
    # Earlier homes of franchises that moved; the participant is the franchise.
    "OAK": ("LV", "Oakland Raiders", 37.752, -122.201),
    "SD": ("LAC", "San Diego Chargers", 32.783, -117.120),
    "STL": ("LA", "St. Louis Rams", 38.633, -90.189),
}

#: Neutral sites, keyed by the stadium ID in the schedule file (exact ID first, then its
#: three-letter prefix).  A value is either coordinates or the abbreviation of the team whose
#: stadium area it is.
_SITES: Dict[str, object] = {
    "BUF01": (43.641, -79.389),          # Toronto
    "LON": (51.556, -0.280), "MEX": (19.303, -99.150), "GER": (48.219, 11.625), "MUN": (48.219, 11.625),
    "FRA": (50.069, 8.645), "MAD": (40.453, -3.688), "SAO": (-23.545, -46.474), "RIO": (-22.912, -43.230),
    "MEL": (-37.820, 144.983), "PAR": (48.924, 2.360),
    "ATL": "ATL", "CLE": "CLE", "DAL": "DAL", "DET": "DET", "HOU": "HOU", "IND": "IND", "JAX": "JAX",
    "LAX": "LA", "MIA": "MIA", "MIN": "MIN", "NOR": "NO", "NYC": "NYG", "PHO": "ARI", "PIT": "PIT",
    "SDG": "SD", "SFO": "SF", "TAM": "TB", "VEG": "LV",
}


def franchise(abbr: str) -> str:
    """Franchise abbreviation for a team abbreviation of any era (unknown ones map to themselves)."""
    return NFL_TEAMS[abbr][0] if abbr in NFL_TEAMS else abbr


def team_name(abbr: str) -> str:
    return NFL_TEAMS[abbr][1] if abbr in NFL_TEAMS else abbr


def team_coords(abbr: Optional[str]) -> Tuple[float, float]:
    """``(lat, lon)`` of a team's stadium area in the era of ``abbr``; NaN when unknown."""
    if abbr in NFL_TEAMS:
        return NFL_TEAMS[abbr][2], NFL_TEAMS[abbr][3]
    return _NAN


def site_coords(stadium_id: Optional[str]) -> Tuple[float, float]:
    """``(lat, lon)`` of a neutral site from its stadium ID; NaN when the site is not listed."""
    if not isinstance(stadium_id, str) or len(stadium_id) < 3:
        return _NAN
    hit = _SITES.get(stadium_id, _SITES.get(stadium_id[:3]))
    if hit is None:
        return _NAN
    return team_coords(hit) if isinstance(hit, str) else hit  # type: ignore[return-value]
