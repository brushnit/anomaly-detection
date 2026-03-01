import numpy as np
import pandas as pd
from time import time_ns
from adtk.data import validate_series

from utils.time_series_generator import create_time_series_ensemble


def aggregate_streams(
    stream_counts: np.ndarray,
    stream_confidences: np.ndarray,
    min_valid: int = 2,
) -> tuple[float, float, bool]:
    """
    Confidence-weighted mean across streams.
    Returns (aggregated_count, mean_confidence, low_confidence_flag).
    """
    valid = ~np.isnan(stream_counts)
    if valid.sum() < min_valid:
        return np.nan, np.nan, True

    w = stream_confidences[valid]
    w = w / (w.sum() + 1e-9)
    count = float((stream_counts[valid] * w).sum())
    mean_conf = float(stream_confidences[valid].mean())
    return count, mean_conf, valid.sum() < len(stream_counts)


def stream_divergence(
    stream_counts: np.ndarray,
) -> float:
    """Coefficient of variation across streams — sensor disagreement signal."""
    valid = stream_counts[~np.isnan(stream_counts)]
    if len(valid) < 2:
        return 0.0
    return float(valid.std() / (valid.mean() + 1e-9))


def nan_audit(stream_counts: np.ndarray) -> str:
    """
    Classify NaN pattern for downstream handling.
    Returns: 'ok' | 'low_confidence' | 'no_data'
    """
    n_valid = int(np.sum(~np.isnan(stream_counts)))
    if n_valid == 0:
        return "no_data"
    if n_valid < 2:
        return "low_confidence"
    return "ok"


def type_composition_vector(
    type_counts: dict[str, float],
    aircraft_types: list[str] = ("commercial", "cargo", "GA", "military", "other"),
) -> dict[str, float]:
    """Normalize raw type counts to a probability distribution."""
    total = sum(type_counts.values()) + 1e-9
    return {t: type_counts.get(t, 0.0) / total for t in aircraft_types}


def preprocess_time_series() -> tuple[list[str], pd.DataFrame, pd.DataFrame]:

    ts_raw = create_time_series_ensemble(
        start_date="2024-06-01",
        magnitude=15,
        noise_scale=0.4,
        floor=2,
        log_rates=[0.5, 0.3, 0.2],
        seasonal_cycles={1: 0.3, 7: 0.45, 365.25: 0.4},
        seed=int(time_ns()),
        y_label="source",
        plot=True,
    )
    ### INGEST DATA ABOVE ^^^

    # -Keep relevant/common fields (airport_id, date, stream counts, confidences)
    # -Enrich with static fields (country_code, lat, lon, airport name)
    #
    # Per-row (per airport-day) steps:
    # -nan_audit(stream_counts)             → skip 'no_data' rows entirely
    # -aggregate_streams(counts, confs)     → weighted count + low_confidence flag
    # -stream_divergence(counts)            → sensor disagreement score
    # -type_composition_vector(type_counts) → normalized aircraft type distribution
    # -attach cloud_cover from stream or external source (0–100 float)
    ### DATA PREPROCESS ABOVE ^^^

    source_cols = [col for col in ts_raw.columns if col != "time"]

    ts_sparse = ts_raw.set_index("time")
    ts_sparse.index = pd.DatetimeIndex(ts_sparse.index)
    ts_sparse = validate_series(ts_sparse)

    daily_index = pd.date_range(ts_sparse.index.min(), ts_sparse.index.max(), freq="D")

    ts_daily = (
        ts_sparse
        .reindex(daily_index)
        .interpolate(method="time")
        .ffill()
        .bfill()
    )
    ts_daily.index.name = "time"
    ts_daily = validate_series(ts_daily)

    return source_cols, ts_sparse, ts_daily