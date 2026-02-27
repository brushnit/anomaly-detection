import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

def create_time_series(
    start_date,
    magnitude=10.0,
    log_rate=1.0,
    seasonal_cycles={},
    seed=None,
    y_label="y",
    plot=False,
):
    """
    Creates a time series dataset with a seasonal component.

    Parameters
    ----------
    start_date : str or datetime-like
        The starting date for the time series (data runs until today).
    magnitude : float
        Scales the amplitude of the entire signal.
    log_rate : float
        Average number of readings per day.
    seasonal_cycles : dict or None
        Dict of {period_in_days: amplitude}, e.g. {7: 0.5, 365.25: 1.0}
    seed : int or None
        Random seed for reproducibility (None means no seed is set).
    y_label : str, optional
        Column name for the value column in the returned DataFrame (default: "y").
    plot : bool, optional
        If True, displays a matplotlib plot of the time series (default: False).

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ["time", y_label], where "time" contains
        normalized DatetimeIndex values and y_label contains the simulated values.
        Only one entry per day is returned (mean of any same-day values, rounded).
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)

    # Fourier transform-based seasonal component generator, for week and year cycles (or any 2 defined seaosonal cycles)
    def seasonal_component(t, cycles):
        return sum((a * np.sin(2 * np.pi * t / p) for p, a in cycles.items()), np.zeros(len(t)))

    # Simulate data from start date to today
    duration_days = (date.today() - start.date()).days
    
    actual_samples = min(rng.poisson(log_rate * duration_days), duration_days) # n_samples based on log_rate and duration
    time_numeric = np.sort(rng.uniform(0, duration_days, actual_samples))
    time = (start + pd.to_timedelta(time_numeric, unit="D")).normalize()

    seasonal = seasonal_component(time_numeric, seasonal_cycles)
    noise = rng.standard_normal(actual_samples)
    y = np.clip((seasonal + noise) * magnitude, 0, None).round().astype(int)

    df = (
    pd.DataFrame({"time": time, y_label: y})
    .loc[y > 0]
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