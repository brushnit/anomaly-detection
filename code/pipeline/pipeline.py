"""
pipeline.py — daily orchestrator
Run once per day via cron. Processes all airports in parallel.
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool
from datetime import date

from anomaly_scoring import score_airport_day
from baseline_models import AirportBaseline, retrain_baseline_cells
from suppression import apply_suppression, build_neighbor_index
from output_records import AirportAnomaly, write_anomalies

import holidays as hdays  # pip install holidays


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANOMALY_STORE_PATH  = "data/anomalies.parquet"
BASELINE_STORE_PATH = "data/baselines/"      # one pickle per airport_id
HISTORY_PATH        = "data/history.parquet"
NEIGHBOR_RADIUS_KM  = 300.0
HARD_FLAG_THRESHOLD = 0.7
COMPOSITION_SHIFT_THRESHOLD = 0.3
MIN_HISTORY_DAYS    = 28


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_baseline(airport_id: str) -> AirportBaseline:
    import pickle, os
    path = f"{BASELINE_STORE_PATH}{airport_id}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return AirportBaseline(airport_id=airport_id)


def save_baseline(baseline: AirportBaseline) -> None:
    import pickle
    path = f"{BASELINE_STORE_PATH}{baseline.airport_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump(baseline, f)


def build_holiday_lookup(country_codes: list[str], year: int) -> dict[str, set]:
    """Map country_code → set of holiday dates for given year."""
    lookup = {}
    for cc in country_codes:
        try:
            lookup[cc] = set(hdays.country_holidays(cc, years=year).keys())
        except Exception:
            lookup[cc] = set()
    return lookup


def dominant_type(type_dist: dict[str, float]) -> str:
    return max(type_dist, key=type_dist.get) if type_dist else "unknown"


# ---------------------------------------------------------------------------
# Per-airport worker
# ---------------------------------------------------------------------------

def process_airport(args: tuple) -> AirportAnomaly | None:
    airport_row, today_obs, history_slice, holiday_dates = args

    aid          = airport_row["airport_id"]
    lat, lon     = airport_row["lat"], airport_row["lon"]
    country      = airport_row["country_code"]
    cluster_id   = airport_row.get("cluster_id", -1)

    if today_obs is None or len(history_slice) < MIN_HISTORY_DAYS:
        return None  # insufficient data — skip, don't flag

    ### LOAD BASELINE STATE
    baseline = load_baseline(aid)

    count        = today_obs["count"]
    cloud_cover  = today_obs["cloud_cover"]
    dow          = pd.Timestamp(today_obs["date"]).dayofweek
    stream_counts = np.array(today_obs["stream_counts"])
    stream_confs  = np.array(today_obs["stream_confidences"])
    today_types   = today_obs.get("aircraft_types", {})
    mean_conf     = float(np.nanmean(stream_confs))
    low_conf_data = bool(np.sum(~np.isnan(stream_counts)) < 2)

    ### BASELINE TYPE DISTRIBUTION from history
    baseline_types = (
        history_slice["aircraft_types"]
        .apply(lambda d: pd.Series(d) if isinstance(d, dict) else pd.Series())
        .mean()
        .to_dict()
    )

    scores = score_airport_day(
        baseline=baseline,
        count=count,
        dow=dow,
        cloud_cover=cloud_cover,
        today_types=today_types,
        baseline_types=baseline_types,
        stream_counts=stream_counts,
        stream_confidences=stream_confs,
    )

    ### UPDATE BASELINE — incremental O(1) update
    if not low_conf_data:
        baseline.update(count, dow, cloud_cover)
        save_baseline(baseline)

    return AirportAnomaly(
        airport_id=aid,
        date=pd.Timestamp(today_obs["date"]),
        lat=lat, lon=lon,
        country_code=country,
        cluster_id=cluster_id,
        observed_count=count,
        expected_count=scores["expected_count"],
        cloud_cover=cloud_cover,
        mean_stream_confidence=mean_conf,
        stream_disagreement=scores["stream_anomaly_score"],
        low_confidence_data=low_conf_data,
        deviation_sigma=scores["deviation_sigma"],
        count_anomaly_score=scores["count_anomaly_score"],
        composition_anomaly_score=scores["composition_anomaly_score"],
        stream_anomaly_score=scores["stream_anomaly_score"],
        ensemble_confidence=scores["ensemble_confidence"],
        detectors_fired=scores["detectors_fired"],
        dominant_type_today=dominant_type(today_types),
        dominant_type_baseline=dominant_type(baseline_types),
        composition_shift=scores["composition_anomaly_score"] > COMPOSITION_SHIFT_THRESHOLD,
        is_public_holiday=False,        # filled by suppression layer
        neighbor_correlation=0.0,       # filled by suppression layer
        likely_weather_event=False,     # filled by suppression layer
        regime_event=False,             # filled by suppression layer
        suppressed=False,               # filled by suppression layer
        is_hard_flag=scores["ensemble_confidence"] >= HARD_FLAG_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Daily run
# ---------------------------------------------------------------------------

def run_daily_pipeline(
    airports: pd.DataFrame,         # airport_id, lat, lon, country_code, cluster_id
    today_observations: dict,       # {airport_id: obs_dict}
    history: pd.DataFrame,          # full history store
    n_workers: int = 8,
) -> None:

    today = date.today()
    holiday_lookup = build_holiday_lookup(airports["country_code"].unique().tolist(), today.year)
    neighbor_index = build_neighbor_index(airports, radius_km=NEIGHBOR_RADIUS_KM)

    ### BUILD WORKER ARGS
    args = []
    for _, airport_row in airports.iterrows():
        aid = airport_row["airport_id"]
        obs = today_observations.get(aid)
        hist = history[history["airport_id"] == aid].tail(90)
        args.append((airport_row.to_dict(), obs, hist, holiday_lookup))

    ### PARALLEL SCORING — embarrassingly parallel
    with Pool(n_workers) as pool:
        results = pool.map(process_airport, args)

    records = [r._asdict() for r in results if r is not None]

    ### SUPPRESSION LAYER
    records = apply_suppression(
        records,
        neighbor_index=neighbor_index,
        holiday_dates=holiday_lookup,
        total_airports=len(airports),
    )

    anomalies = [AirportAnomaly(**r) for r in records]

    ### WRITE OUTPUT
    write_anomalies(anomalies, ANOMALY_STORE_PATH)

    hard_flags = [a for a in anomalies if a.is_hard_flag and not a.suppressed]
    print(f"[{today}] {len(anomalies)} scored | {len(hard_flags)} hard flags emitted")


if __name__ == "__main__":
    ### LOAD REAL DATA HERE
    # airports   = pd.read_parquet("data/airports.parquet")
    # today_obs  = load_today_observations()
    # history    = pd.read_parquet("data/history.parquet")
    # run_daily_pipeline(airports, today_obs, history)
    pass