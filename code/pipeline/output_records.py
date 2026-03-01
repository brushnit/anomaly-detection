import pandas as pd
from typing import NamedTuple


class AirportAnomaly(NamedTuple):
    # Identity
    airport_id:               str
    date:                     pd.Timestamp
    lat:                      float
    lon:                      float
    country_code:             str
    cluster_id:               int

    # Observed state
    observed_count:           float
    expected_count:           float
    cloud_cover:              float
    mean_stream_confidence:   float
    stream_disagreement:      float
    low_confidence_data:      bool

    # Anomaly signals
    deviation_sigma:          float
    count_anomaly_score:      float
    composition_anomaly_score: float
    stream_anomaly_score:     float
    ensemble_confidence:      float
    detectors_fired:          list[str]

    # Aircraft composition
    dominant_type_today:      str
    dominant_type_baseline:   str
    composition_shift:        bool       # JS divergence > threshold

    # Suppression context
    is_public_holiday:        bool
    neighbor_correlation:     float
    likely_weather_event:     bool
    regime_event:             bool
    suppressed:               bool

    # Final verdict
    is_hard_flag:             bool       # confidence >= 0.7 and not suppressed


def to_dataframe(records: list[AirportAnomaly]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=AirportAnomaly._fields)


def write_anomalies(records: list[AirportAnomaly], path: str) -> None:
    """Append today's records to persistent Parquet store."""
    df = to_dataframe(records)

    ### STORAGE — append to partitioned parquet; partition by date for cheap time queries
    try:
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_parquet(path, index=False)


def load_anomalies(path: str, since: pd.Timestamp | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if since is not None:
        df = df[df["date"] >= since]
    return df