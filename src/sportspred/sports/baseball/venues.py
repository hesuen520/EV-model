"""City coordinates used only to measure travel between consecutive games.

Latitude/longitude of the cities major-league games were played in from 2000
(public geography, city-level precision: two parks in one city are the same
point).  Retrosheet's ballpark table names each park's city; a city that is
not listed here yields a missing travel value rather than a guess.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ...core.ids import norm_text

__all__ = ["city_coords"]

_CITIES: Dict[str, Tuple[float, float]] = {
    "anaheim": (33.800, -117.883), "arlington": (32.747, -97.083), "atlanta": (33.750, -84.390),
    "baltimore": (39.284, -76.622), "birmingham": (33.520, -86.810), "boston": (42.346, -71.097),
    "bristol": (36.520, -82.260), "buffalo": (42.880, -78.870), "chicago": (41.880, -87.630),
    "cincinnati": (39.097, -84.507), "cleveland": (41.496, -81.685), "denver": (39.756, -104.994),
    "detroit": (42.339, -83.049), "dunedin": (28.000, -82.790), "dyersville": (42.480, -91.120),
    "fort bragg": (35.140, -79.000), "houston": (29.757, -95.355), "kansas city": (39.051, -94.480),
    "lake buena vista": (28.337, -81.556), "london": (51.540, -0.020), "los angeles": (34.074, -118.240),
    "mexico city": (19.400, -99.090), "miami": (25.778, -80.220), "milwaukee": (43.028, -87.971),
    "minneapolis": (44.982, -93.278), "monterrey": (25.720, -100.310), "montreal": (45.558, -73.552),
    "new york": (40.790, -73.890), "oakland": (37.752, -122.201), "omaha": (41.267, -95.932),
    "philadelphia": (39.906, -75.166), "phoenix": (33.445, -112.067), "pittsburgh": (40.447, -80.006),
    "sacramento": (38.580, -121.514), "san diego": (32.707, -117.157), "seattle": (47.591, -122.332),
    "seoul": (37.498, 126.867), "san francisco": (37.778, -122.389), "san juan": (18.417, -66.073),
    "st louis": (38.623, -90.193), "st petersburg": (27.768, -82.653), "sydney": (-33.892, 151.225),
    "tampa": (27.980, -82.507), "tokyo": (35.706, 139.752), "toronto": (43.641, -79.389),
    "washington": (38.873, -77.007), "williamsport": (41.240, -77.050),
}


def city_coords(city: Optional[str]) -> Tuple[float, float]:
    """``(lat, lon)`` for a ballpark city, or ``(nan, nan)`` when unknown."""
    return _CITIES.get(norm_text(city), (float("nan"), float("nan")))
