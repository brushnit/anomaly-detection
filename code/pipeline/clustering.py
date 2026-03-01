import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# from hdbscan import HDBSCAN  # preferred if available


def compute_airport_features(airport_id: str, ts: pd.Series) -> dict:
    """
    Build behavioral feature vector for a single airport.
    ts: daily aggregated count series (pd.Series, DatetimeIndex)
    """

    ### FEATURE EXTRACTION — extend with real seasonal_strength() impl
    weekday = ts.groupby(ts.index.dayofweek).mean()

    features = {
        "airport_id":       airport_id,
        "log_mean_count":   np.log1p(ts.mean()),
        "cv":               ts.std() / (ts.mean() + 1e-6),
        "weekly_amplitude": ts.resample("W").mean().std(),   # placeholder
        "annual_amplitude": ts.resample("ME").mean().std(),  # placeholder
        "peak_dow":         weekday.idxmax(),
        "weekend_ratio":    weekday.iloc[[5, 6]].mean() / (weekday.iloc[:5].mean() + 1e-6),
        "zero_rate":        (ts == 0).mean(),
        "p99_p50_ratio":    ts.quantile(0.99) / (ts.median() + 1e-6),
    }

    ### ENRICH WITH CLOUD SENSITIVITY — correlate cloud_cover with count if available
    # features["cloud_sensitivity"] = ts.corr(cloud_cover_series)

    return features


def cluster_airports(
    feature_records: list[dict],
    n_clusters: int = 8,
    min_history_days: int = 30,
) -> pd.DataFrame:
    """
    Cluster airports by behavioral archetype.
    Returns DataFrame with airport_id → cluster_id mapping.
    Cadence: monthly batch job.
    """

    df = pd.DataFrame(feature_records).set_index("airport_id")

    ### FILTER — drop airports without enough history
    df = df.dropna()

    feature_cols = [c for c in df.columns if c != "peak_dow"]
    X = StandardScaler().fit_transform(df[feature_cols])

    ### CLUSTER — swap KMeans for HDBSCAN for better density handling
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(X)
    # labels = HDBSCAN(min_cluster_size=10).fit_predict(X)

    df["cluster_id"] = labels
    return df[["cluster_id"]]


def assign_cluster(
    airport_features: dict,
    cluster_model,           # fitted KMeans or centroid store
    scaler: StandardScaler,
    feature_cols: list[str],
) -> int:
    """
    Assign a single airport (e.g. cold-start) to nearest cluster centroid.
    """

    x = np.array([airport_features[c] for c in feature_cols]).reshape(1, -1)
    x_scaled = scaler.transform(x)

    ### COLD START — new airports inherit nearest cluster's base model
    return int(cluster_model.predict(x_scaled)[0])