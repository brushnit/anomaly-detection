import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


# ---------------------------------------------------------------------------
# Geographic utilities
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ = radians(lat2 - lat1)
    dλ = radians(lon2 - lon1)
    a = sin(dφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(dλ / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_neighbor_index(
    airports: pd.DataFrame,   # columns: airport_id, lat, lon
    radius_km: float = 300.0,
) -> dict[str, list[str]]:
    """
    Precompute neighbor lists once at startup (or after re-cluster).
    Returns {airport_id: [neighbor_ids within radius]}.
    """

    ### NEIGHBOR INDEX — precompute once, reuse daily
    neighbors = {}
    records = airports.to_dict("records")
    for a in records:
        nbrs = [
            b["airport_id"] for b in records
            if b["airport_id"] != a["airport_id"]
            and haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) <= radius_km
        ]
        neighbors[a["airport_id"]] = nbrs
    return neighbors


# ---------------------------------------------------------------------------
# Suppression logic
# ---------------------------------------------------------------------------

def neighbor_correlation(
    airport_id: str,
    neighbor_index: dict[str, list[str]],
    flagged_today: set[str],
) -> float:
    """Fraction of geographic neighbors also flagged today."""
    nbrs = neighbor_index.get(airport_id, [])
    if not nbrs:
        return 0.0
    return sum(n in flagged_today for n in nbrs) / len(nbrs)


def is_regime_event(
    flagged_today: set[str],
    total_airports: int,
    global_threshold: float = 0.15,
) -> bool:
    """Global suppression: too many airports flagging = systemic event, not anomalies."""
    return len(flagged_today) / max(total_airports, 1) > global_threshold


def apply_suppression(
    records: list[dict],
    neighbor_index: dict[str, list[str]],
    holiday_dates: dict[str, set],   # {country_code: {date, ...}}
    total_airports: int,
    neighbor_threshold: float = 0.4,
    global_threshold: float = 0.15,
) -> list[dict]:
    """
    Annotate anomaly records with suppression context.
    Does NOT remove records — preserves them for downstream review.
    """

    flagged_today = {r["airport_id"] for r in records if r.get("is_hard_flag")}
    global_regime = is_regime_event(flagged_today, total_airports, global_threshold)

    for r in records:
        aid = r["airport_id"]
        date = r["date"]

        r["neighbor_correlation"] = neighbor_correlation(aid, neighbor_index, flagged_today)
        r["likely_weather_event"] = r["neighbor_correlation"] >= neighbor_threshold

        country = r.get("country_code", "")
        r["is_public_holiday"] = date in holiday_dates.get(country, set())

        ### REGIME EVENT — suppress but preserve; let downstream decide
        r["regime_event"] = global_regime

        r["suppressed"] = (
            r["regime_event"]
            or r["likely_weather_event"]
            or r["is_public_holiday"]
        )

    return records