import numpy as np
import pandas as pd
from adtk.detector import SeasonalAD
from adtk.data import validate_series
from dataclasses import dataclass, field


CLOUD_BINS = [0, 25, 50, 75, 100]  # cloud cover percentile buckets


def _cloud_bin(cloud_cover: float) -> int:
    return int(np.digitize(cloud_cover, CLOUD_BINS[1:-1]))


@dataclass
class AirportBaseline:
    """
    Stratified rolling baseline conditioned on (DOW, cloud_bin).
    State is serializable — store as JSON/pickle per airport.
    Update daily (O(1)), retrain weekly from trailing 90-day window.
    """
    airport_id: str
    # rolling stats: key = (dow, cloud_bin) → {"mean": float, "std": float, "n": int}
    cells: dict = field(default_factory=dict)
    seasonal_model: SeasonalAD | None = None

    def _key(self, dow: int, cloud_bin: int) -> str:
        return f"{dow}_{cloud_bin}"

    def update(self, count: float, dow: int, cloud_cover: float) -> None:
        """Incremental O(1) update with a new observation."""
        key = self._key(dow, _cloud_bin(cloud_cover))
        cell = self.cells.get(key, {"mean": count, "std": 0.0, "n": 0})
        n = cell["n"] + 1
        delta = count - cell["mean"]
        cell["mean"] += delta / n
        cell["std"] = np.sqrt(((n - 1) * cell["std"] ** 2 + delta * (count - cell["mean"])) / n)
        cell["n"] = n
        self.cells[key] = cell

    def expected(self, dow: int, cloud_cover: float) -> tuple[float, float]:
        """Return (mean, std) for given conditions. Falls back to global stats."""
        key = self._key(dow, _cloud_bin(cloud_cover))
        cell = self.cells.get(key)
        if cell and cell["n"] >= 7:
            return cell["mean"], max(cell["std"], 1.0)
        # fallback: average across all cells
        all_means = [c["mean"] for c in self.cells.values()]
        all_stds  = [c["std"]  for c in self.cells.values()]
        return np.mean(all_means), max(np.mean(all_stds), 1.0)

    def zscore(self, count: float, dow: int, cloud_cover: float) -> float:
        mean, std = self.expected(dow, cloud_cover)
        return (count - mean) / std


def retrain_seasonal(ts_daily: pd.Series, period: int = 7) -> SeasonalAD:
    """
    Weekly retrain of SeasonalAD on trailing 90-day window.
    ts_daily: validated daily pd.Series
    """
    ### RETRAIN — call weekly per airport in batch job
    model = SeasonalAD(period)
    model.fit(validate_series(ts_daily.to_frame()))
    return model


def retrain_baseline_cells(
    baseline: AirportBaseline,
    history: pd.DataFrame,   # columns: count, dow, cloud_cover
) -> AirportBaseline:
    """
    Rebuild cell stats from trailing window (weekly cadence).
    history: last 90 days of observations for this airport.
    """
    baseline.cells = {}
    for _, row in history.iterrows():
        baseline.update(row["count"], int(row["dow"]), row["cloud_cover"])
    return baseline