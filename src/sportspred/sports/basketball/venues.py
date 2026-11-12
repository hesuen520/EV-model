"""City coordinates used only to measure travel between consecutive games.

Latitude/longitude of the cities NBA and WNBA games are played in (public
geography, city-level precision).  A city that is not listed yields a missing
travel value rather than a guess.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ...core.ids import norm_text

__all__ = ["city_coords"]

_CITIES: Dict[str, Tuple[float, float]] = {
    "atlanta": (33.757, -84.396), "boston": (42.366, -71.062), "brooklyn": (40.683, -73.975),
    "new york": (40.751, -73.994), "charlotte": (35.225, -80.839), "chicago": (41.881, -87.674),
    "cleveland": (41.497, -81.688), "dallas": (32.790, -96.810), "denver": (39.749, -105.008),
    "detroit": (42.341, -83.055), "auburn hills": (42.697, -83.246), "san francisco": (37.768, -122.388),
    "oakland": (37.750, -122.203), "houston": (29.751, -95.362), "indianapolis": (39.764, -86.156),
    "los angeles": (34.043, -118.267), "inglewood": (33.945, -118.343), "memphis": (35.138, -90.051),
    "miami": (25.781, -80.188), "milwaukee": (43.045, -87.917), "minneapolis": (44.979, -93.276),
    "new orleans": (29.949, -90.082), "oklahoma city": (35.463, -97.515), "orlando": (28.539, -81.384),
    "lake buena vista": (28.337, -81.556), "philadelphia": (39.901, -75.172), "phoenix": (33.446, -112.071),
    "portland": (45.532, -122.667), "sacramento": (38.580, -121.500), "san antonio": (29.427, -98.438),
    "seattle": (47.622, -122.354), "toronto": (43.643, -79.379), "salt lake city": (40.768, -111.901),
    "washington": (38.898, -77.021), "east rutherford": (40.812, -74.068), "newark": (40.734, -74.171),
    "vancouver": (49.278, -123.109), "tampa": (27.943, -82.452), "mexico city": (19.497, -99.176),
    "london": (51.503, 0.003), "paris": (48.839, 2.378), "las vegas": (36.103, -115.178),
    "tokyo": (35.713, 139.752), "abu dhabi": (24.467, 54.603), "manchester": (53.488, -2.244),
    "berlin": (52.506, 13.444), "uncasville": (41.491, -72.089), "arlington": (32.740, -97.108),
    "college park": (33.642, -84.449), "everett": (47.978, -122.202), "white plains": (41.033, -73.775),
    "henderson": (36.040, -114.982), "st paul": (44.945, -93.101), "st petersburg": (27.771, -82.653),
    "anaheim": (33.807, -117.877), "honolulu": (21.294, -157.858), "bradenton": (27.440, -82.460),
}


def city_coords(city: Optional[str]) -> Tuple[float, float]:
    """``(lat, lon)`` for a venue city, or ``(nan, nan)`` when unknown."""
    key = norm_text(city)
    if key.startswith("st "):
        key = "st " + key[3:]
    return _CITIES.get(key, (float("nan"), float("nan")))
