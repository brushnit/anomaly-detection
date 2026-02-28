import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date


def _seasonal_component(t, cycles):
    """Compute the sum of sinusoidal seasonal components over time array `t`."""
    if not cycles:
        return np.zeros(len(t))
    return sum((a * np.sin(2 * np.pi * t / p) for p, a in cycles.items()), np.zeros(len(t)))


def _round_int_series(series):
    """Round a float Series to the nearest integer dtype."""
    return series.round().astype(int)


def create_time_series(
    start_date,
    end_date=None,
    magnitude=10.0,
    noise_scale=1.0,
    floor=0.0,
    log_rate=[1.0],
    seasonal_cycles={},
    seed=None,
    y_label="y",
    plot=False,
):
    """
    Generate a single simulated time series.

    Parameters
    ----------
    start_date : str or date-like
        Start of the simulation window.
    end_date : str or date-like, optional
        End of the simulation window. Defaults to today.
    magnitude : float
        Scaling factor applied to the signal + noise.
    floor : float
        Minimum allowed value; observations below this are dropped.
    noise_scale : float
        Standard deviation of the i.i.d. Gaussian noise added to each
        observation. Lower values (e.g. 0.2) reduce day-to-day jumps;
        higher values increase volatility. Defaults to 1.0.
    log_rate : list[float]
        Poisson rate multiplier(s) for sample density.
    seasonal_cycles : dict[float, float]
        Mapping of {period_days: amplitude} for sinusoidal components.
    seed : int or None
        Random seed for reproducibility.
    y_label : str
        Column name for the output values.
    plot : bool
        Whether to display a plot of the result.

    Returns
    -------
    pd.DataFrame
        Columns: ["time", y_label]
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(date.today())

    duration_days = (end.date() - start.date()).days
    if duration_days <= 0:
        raise ValueError(f"end_date ({end.date()}) must be after start_date ({start.date()}).")

    n_samples = min(rng.poisson(log_rate * duration_days), duration_days)
    time_numeric = np.sort(rng.uniform(0, duration_days, n_samples))
    time = (start + pd.to_timedelta(time_numeric, unit="D")).normalize()

    seasonal = _seasonal_component(time_numeric, seasonal_cycles)
    noise = rng.standard_normal(n_samples) * noise_scale

    # Keep as float; single round+cast happens after groupby aggregation
    y_float = (seasonal + noise) * magnitude + floor

    mask = y_float >= floor
    time, y_float = time[mask], y_float[mask]

    df = (
        pd.DataFrame({"time": time, y_label: y_float})
        .groupby("time", as_index=False)[y_label]
        .mean()
        .assign(**{y_label: lambda x: _round_int_series(x[y_label])})
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
    end_date=None,
    magnitude=10.0,
    noise_scale=1.0,
    floor=0.0,
    log_rates=[1.0],
    seasonal_cycles={},
    seed=None,
    y_label="source",
    plot=False,
):
    """
    Generate an ensemble of simulated time series sharing a common ground-truth signal.

    Parameters
    ----------
    start_date : str or date-like
        Start of the simulation window.
    end_date : str or date-like, optional
        End of the simulation window. Defaults to today.
    magnitude : float
        Scaling factor applied to the signal + noise.
    floor : float
        Minimum allowed value; observations below this are dropped.
    noise_scale : float
        Standard deviation of the i.i.d. Gaussian noise added to each
        observation. Lower values (e.g. 0.2) reduce day-to-day jumps;
        higher values increase volatility. Defaults to 1.0.
    log_rates : list[float]
        One Poisson rate multiplier per series in the ensemble.
    seasonal_cycles : dict[float, float]
        Mapping of {period_days: amplitude} for sinusoidal components.
    seed : int or None
        Random seed for reproducibility.
    y_label : str
        Prefix for each series column (e.g. "source" → "source_1", "source_2", …).
    plot : bool
        Whether to display a plot of the ensemble.

    Returns
    -------
    pd.DataFrame
        Columns: ["time", "{y_label}_1", "{y_label}_2", …]
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(date.today())

    duration_days = (end.date() - start.date()).days
    if duration_days <= 0:
        raise ValueError(f"end_date ({end.date()}) must be after start_date ({start.date()}).")

    # Shared ground-truth signal on integer day grid
    gt_t = np.arange(duration_days, dtype=float)
    gt_signal = _seasonal_component(gt_t, seasonal_cycles)

    dfs = []
    for i, log_rate in enumerate(log_rates):
        col = f"{y_label}_{i + 1}"

        n_samples = min(rng.poisson(log_rate * duration_days), duration_days)
        time_numeric = np.sort(rng.choice(gt_t, size=n_samples, replace=False))
        time = (start + pd.to_timedelta(time_numeric, unit="D")).normalize()

        seasonal = gt_signal[time_numeric.astype(int)]
        noise = rng.standard_normal(n_samples) * noise_scale

        # Keep as float; single round+cast happens after groupby aggregation
        y_float = (seasonal + noise) * magnitude + floor

        mask = y_float >= floor
        time, y_float = time[mask], y_float[mask]

        df = (
            pd.DataFrame({"time": time, col: y_float})
            .groupby("time", as_index=False)[col]
            .mean()
            .assign(**{col: lambda x: _round_int_series(x[col])})
        )
        dfs.append(df)

    # Merge all series on time using pd.concat + pivot to avoid O(n²) chained merges
    long = pd.concat(
        [df.melt(id_vars="time", var_name="series", value_name="value") for df in dfs],
        ignore_index=True,
    )
    combined = (
        long.pivot_table(index="time", columns="series", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values("time")
        .reset_index(drop=True)
    )
    # Restore integer dtype on all series columns (pivot_table promotes to float due to NaNs)
    series_cols = [f"{y_label}_{i + 1}" for i in range(len(log_rates))]
    combined[series_cols] = combined[series_cols].round().astype("Int64")  # nullable int to handle NaNs

    if plot:
        fig, ax = plt.subplots(figsize=(14, 4))
        for col in series_cols:
            ax.plot(combined["time"], combined[col], ".", markersize=3, label=col)
        ax.set_title("Simulated Time Series Data Ensemble")
        ax.set_xlabel("Time")
        ax.set_ylabel("Detections")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return combined