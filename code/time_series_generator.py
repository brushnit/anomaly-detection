import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

def create_time_series(
    start_date,
    magnitude=10.0,
    floor=0.0,
    log_rate=[1.0],
    seasonal_cycles={},
    seed=None,
    y_label="y",
    plot=False,
):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)

    def seasonal_component(t, cycles):
        return sum((a * np.sin(2 * np.pi * t / p) for p, a in cycles.items()), np.zeros(len(t)))

    duration_days = (date.today() - start.date()).days

    n_samples = min(rng.poisson(log_rate * duration_days), duration_days)
    time_numeric = np.sort(rng.uniform(0, duration_days, n_samples))
    time = (start + pd.to_timedelta(time_numeric, unit="D")).normalize()

    seasonal = seasonal_component(time_numeric, seasonal_cycles)
    noise = rng.standard_normal(n_samples)

    y = ((seasonal + noise) * magnitude + floor).round().astype(int)

    mask = y >= floor
    time, y = time[mask], y[mask]

    df = (
        pd.DataFrame({"time": time, y_label: y})
        .groupby("time", as_index=False)[y_label]
        .mean()
        .assign(**{y_label: lambda x: x[y_label].round().astype(int)})
    )

    if plot:
        plt.figure(figsize=(10, 6))
        plt.title("Simulated Time Series Data")
        plt.xlabel("Time")
        plt.ylabel(y_label)
        plt.plot(df["time"], df[y_label], "o")
        plt.tight_layout()
        plt.show()

    return df

def create_time_series_ensemble(
    start_date,
    magnitude=10.0,
    floor=0.0,
    log_rates=[1.0],
    seasonal_cycles={},
    seed=None,
    y_label="source",
    plot=False,
):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)

    def seasonal_component(t, cycles):
        return sum((a * np.sin(2 * np.pi * t / p) for p, a in cycles.items()), np.zeros(len(t)))

    duration_days = (date.today() - start.date()).days

    # Shared ground truth signal
    gt_t = np.arange(duration_days, dtype=float)
    gt_signal = seasonal_component(gt_t, seasonal_cycles)

    dfs = []
    for i, log_rate in enumerate(log_rates):
        col = f"{y_label}_{i + 1}"

        n_samples = min(rng.poisson(log_rate * duration_days), duration_days)
        time_numeric = np.sort(rng.choice(gt_t, size=n_samples, replace=False))
        time = (start + pd.to_timedelta(time_numeric, unit="D")).normalize()

        seasonal = gt_signal[time_numeric.astype(int)]
        noise = rng.standard_normal(n_samples)

        y = ((seasonal + noise) * magnitude + floor).round().astype(int)
        mask = y >= floor
        time, y = time[mask], y[mask]

        df = (
            pd.DataFrame({"time": time, col: y})
            .groupby("time", as_index=False)[col]
            .mean()
            .assign(**{col: lambda x: x[col].round().astype(int)})
        )
        dfs.append(df)

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.merge(df, on="time", how="outer")
    combined = combined.sort_values("time").reset_index(drop=True)

    if plot:
        fig, ax = plt.subplots(figsize=(14, 4))
        for i in range(len(log_rates)):
            col = f"{y_label}_{i + 1}"
            ax.plot(combined["time"], combined[col], ".", markersize=3, label=col)
        ax.set_title("Simulated Sensor Ensemble")
        ax.set_xlabel("Time")
        ax.set_ylabel("Detections")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return combined