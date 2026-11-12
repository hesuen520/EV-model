"""NHL club reference data: names, arena locations and roster continuity.

The schedule files identify clubs by tri-code only, and the code changes when
a club moves.  Two chains are treated as one continuing team because the
playing roster moved with the club:

* Atlanta Thrashers (``ATL``) -> Winnipeg Jets (``WPG``), 2011;
* Phoenix Coyotes (``PHX``) -> Arizona Coyotes (``ARI``), 2014 -> Utah
  (``UTA``), 2024.  The league books Utah as a new franchise, but it took over
  Arizona's players and staff, so its ratings continue Arizona's.

Coordinates are the home arena's city (public geography) and are used only to
measure travel.  A tri-code that is not listed gets a missing travel value,
never a guess.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

__all__ = ["canonical_code", "team_names", "team_coords"]

#: tri-code -> (club name, latitude, longitude)
_TEAMS: Dict[str, Tuple[str, float, float]] = {
    "ANA": ("Anaheim Ducks", 33.808, -117.877), "ARI": ("Arizona Coyotes", 33.532, -112.261),
    "ATL": ("Atlanta Thrashers", 33.757, -84.396), "BOS": ("Boston Bruins", 42.366, -71.062),
    "BUF": ("Buffalo Sabres", 42.875, -78.876), "CAR": ("Carolina Hurricanes", 35.803, -78.722),
    "CBJ": ("Columbus Blue Jackets", 39.969, -83.006), "CGY": ("Calgary Flames", 51.037, -114.052),
    "CHI": ("Chicago Blackhawks", 41.881, -87.674), "COL": ("Colorado Avalanche", 39.749, -105.008),
    "DAL": ("Dallas Stars", 32.790, -96.810), "DET": ("Detroit Red Wings", 42.341, -83.055),
    "EDM": ("Edmonton Oilers", 53.547, -113.498), "FLA": ("Florida Panthers", 26.158, -80.326),
    "LAK": ("Los Angeles Kings", 34.043, -118.267), "MIN": ("Minnesota Wild", 44.945, -93.101),
    "MTL": ("Montréal Canadiens", 45.496, -73.569), "NJD": ("New Jersey Devils", 40.734, -74.171),
    "NSH": ("Nashville Predators", 36.159, -86.779), "NYI": ("New York Islanders", 40.712, -73.726),
    "NYR": ("New York Rangers", 40.751, -73.993), "OTT": ("Ottawa Senators", 45.297, -75.927),
    "PHI": ("Philadelphia Flyers", 39.901, -75.172), "PHX": ("Phoenix Coyotes", 33.532, -112.261),
    "PIT": ("Pittsburgh Penguins", 40.440, -79.989), "SEA": ("Seattle Kraken", 47.622, -122.354),
    "SJS": ("San Jose Sharks", 37.333, -121.901), "STL": ("St. Louis Blues", 38.627, -90.203),
    "TBL": ("Tampa Bay Lightning", 27.943, -82.452), "TOR": ("Toronto Maple Leafs", 43.643, -79.379),
    "UTA": ("Utah Mammoth", 40.768, -111.901), "VAN": ("Vancouver Canucks", 49.278, -123.109),
    "VGK": ("Vegas Golden Knights", 36.103, -115.178), "WPG": ("Winnipeg Jets", 49.893, -97.144),
    "WSH": ("Washington Capitals", 38.898, -77.021),
}

#: former tri-code -> the code the same roster plays under today
_SUCCESSOR: Dict[str, str] = {"ATL": "WPG", "PHX": "UTA", "ARI": "UTA"}

#: earlier names of a continuing team, oldest first (registered as aliases)
_FORMER_NAMES: Dict[str, Tuple[str, ...]] = {
    "WPG": ("Atlanta Thrashers",),
    "UTA": ("Phoenix Coyotes", "Arizona Coyotes", "Utah Hockey Club"),
}


def canonical_code(code: str) -> str:
    """Tri-code of the continuing team (``ARI`` -> ``UTA``)."""
    return _SUCCESSOR.get(code, code)


def team_names(code: str, fallback: Optional[str] = None) -> List[str]:
    """Every name the continuing team has carried, oldest first; the last one
    is the current name.  An unlisted tri-code keeps the source's own name."""
    canon = canonical_code(code)
    current = _TEAMS[canon][0] if canon in _TEAMS else (fallback or canon)
    return [*_FORMER_NAMES.get(canon, ()), current]


def team_coords(code: Optional[str]) -> Tuple[float, float]:
    """``(lat, lon)`` of the home arena city for the tri-code a club played
    under at the time (``ATL`` is Atlanta, not Winnipeg); NaNs when unknown."""
    if code in _TEAMS:
        return _TEAMS[code][1], _TEAMS[code][2]
    return float("nan"), float("nan")
