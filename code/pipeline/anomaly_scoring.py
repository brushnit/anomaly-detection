import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from baseline_models import AirportBaseline


DETECTOR_WEIGHTS = {
    "zscore":      1.5,
    "seasonal":    1.2,
    "composition": 1.0,
    "divergence":  0.6,
}

AIRCRAFT_TYPES = ["commercial", "cargo", "GA", "military", "other"]


def score_count_anomaly(
    baseline: AirportBaseline,
    count: float,
    dow: int,
    cloud_cover: float,
) -> tuple[float, float]:
    """Returns (z_score, expected_count)."""
    mean, _ = baseline.expected(dow, cloud_cover)
    z = baseline.zscore(count, dow, cloud_cover)
    return z, mean


def score_composition_anomaly(
    today_types: dict[str, float],
    baseline_types: dict[str, float],
) -> float:
    """
    Jensen-Shannon divergence between today's and baseline aircraft type distributions.
    Returns score in [0, 1]. Higher = more anomalous.
    """

    def to_vec(d: dict) -> np.ndarray:
        v = np.array([d.get(t, 0.0) for t in AIRCRAFT_TYPES], dtype=float)
        s = v.sum()
        return v / s if s > 0 else np.ones(len(AIRCRAFT_TYPES)) / len(AIRCRAFT_TYPES)

    ### COMPOSITION SIGNAL — catches type-mix shifts invisible to count detectors
    return float(jensenshannon(to_vec(today_types), to_vec(baseline_types)))


def score_stream_divergence(
    stream_counts: np.ndarray,
    stream_confidences: np.ndarray,
    cloud_cover: float,
    cloud_threshold: float = 60.0,
) -> float:
    """
    Coefficient of variation across streams.
    High divergence on a clear day = sensor issue.
    High divergence on a cloudy day = expected — downweight accordingly.
    """
    valid = ~np.isnan(stream_counts)
    if valid.sum() < 2:
        return 0.0
    cv = stream_counts[valid].std() / (stream_counts[valid].mean() + 1e-6)
    cloud_penalty = max(0.0, 1.0 - cloud_cover / cloud_threshold)
    return float(cv * cloud_penalty)


def ensemble_score(signal_scores: dict[str, float]) -> float:
    """Weighted sum of detector signals, normalized to [0, 1]."""
    total_weight = sum(DETECTOR_WEIGHTS[k] for k in signal_scores)
    raw = sum(
        DETECTOR_WEIGHTS[k] * min(abs(v), 3.0) / 3.0   # clip z-scores at 3σ
        for k, v in signal_scores.items()
        if k in DETECTOR_WEIGHTS
    )
    return float(raw / total_weight) if total_weight > 0 else 0.0


def score_airport_day(
    baseline: AirportBaseline,
    count: float,
    dow: int,
    cloud_cover: float,
    today_types: dict[str, float],
    baseline_types: dict[str, float],
    stream_counts: np.ndarray,
    stream_confidences: np.ndarray,
    seasonal_anomaly: bool = False,
) -> dict:
    """
    Single entry point for daily per-airport scoring.
    Returns raw signal scores + ensemble confidence.
    """

    z, expected = score_count_anomaly(baseline, count, dow, cloud_cover)

    signals = {
        "zscore":      z,
        "seasonal":    3.0 if seasonal_anomaly else 0.0,
        "composition": score_composition_anomaly(today_types, baseline_types),
        "divergence":  score_stream_divergence(stream_counts, stream_confidences, cloud_cover),
    }

    confidence = ensemble_score(signals)

    return {
        "expected_count":           expected,
        "deviation_sigma":          z,
        "count_anomaly_score":      signals["zscore"],
        "composition_anomaly_score": signals["composition"],
        "stream_anomaly_score":     signals["divergence"],
        "ensemble_confidence":      confidence,
        "detectors_fired":          [k for k, v in signals.items() if abs(v) > 1.0],
        "is_hard_flag":             confidence >= 0.7,
    }