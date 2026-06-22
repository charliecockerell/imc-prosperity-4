"""
Reusable statistical utilities for hypothesis testing and analysis.

Extracted from book_analysis.py and extended with generic tools.
"""

from scipy import stats
import numpy as np
import pandas as pd


def adf_test(series: np.ndarray, alpha: float = 0.05) -> tuple[float, float, bool]:
    """
    Augmented Dickey-Fuller test for stationarity.

    Tests H0: unit root (non-stationary) vs H1: stationary.
    Rejects → series is stationary (mean-reverting).

    Returns:
        (adf_stat, p_value_approx, is_stationary)
    """
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    if len(series) < 20:
        return 0.0, 1.0, False

    y = np.diff(series)
    y_lag = series[:-1]
    n = len(y)

    X = np.column_stack([np.ones(n), y_lag])

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0, False

    y_hat = X @ beta
    resid = y - y_hat
    s2 = np.sum(resid ** 2) / (n - 2)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 0.0, 1.0, False
    se_beta1 = np.sqrt(s2 * XtX_inv[1, 1])

    if se_beta1 == 0:
        return 0.0, 1.0, False

    adf_stat = beta[1] / se_beta1

    # Approximate p-value from MacKinnon critical values (constant, no trend, n→∞)
    if adf_stat < -3.43:
        p_approx = 0.005
    elif adf_stat < -2.86:
        p_approx = 0.03
    elif adf_stat < -2.57:
        p_approx = 0.07
    elif adf_stat < -1.94:
        p_approx = 0.15
    elif adf_stat < -1.62:
        p_approx = 0.25
    else:
        p_approx = 0.50

    return adf_stat, p_approx, p_approx < alpha


def half_life(series: np.ndarray | pd.Series) -> float:
    """
    Estimate mean-reversion half-life via OU process fitting.

    Fits: dx_t = theta * (mu - x_t) * dt + sigma * dW
    Half-life = ln(2) / theta

    Interpretation:
      - Small half-life (1-5 ticks) → fast mean reversion → quote tight, hold briefly
      - Large half-life (50+ ticks) → slow reversion → quote wide, hold longer
      - Infinite → random walk (no reversion)

    Returns:
        half-life in ticks (same units as input series)
    """
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]

    if len(series) < 20:
        return np.inf

    # Regress dx on x_{t-1}: dx_t = alpha + beta * x_{t-1}
    # For OU process, beta = -theta (mean reversion speed)
    y = np.diff(series)
    x = series[:-1]

    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.inf

    theta = -beta[1]

    if theta <= 0:
        return np.inf  # Not mean-reverting

    return np.log(2) / theta


def chi2_uniform(counts: np.ndarray, alpha: float = 0.05) -> tuple[float, float, bool]:
    """
    Chi-squared test for uniformity.

    Args:
        counts: array of bin counts
        alpha: significance level

    Returns:
        (chi2_stat, p_value, is_uniform) where is_uniform = (p_value > alpha)
    """
    expected = np.full_like(counts, counts.sum() / len(counts), dtype=float)    ## Generates a full array in same shape as counts, with expected for uniform dist.
    chi2_stat = np.sum((counts - expected) ** 2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2_stat, len(counts) - 1)    ## Does this work when degrees of freedom changes due to expected being too low is this an issue? 
    return chi2_stat, p_value, p_value > alpha


def z_test_proportion(successes: int, total: int, p_null: float = 0.5, alpha: float = 0.05) -> tuple[float, float, bool]:
    """
    Two-tailed z-test for a proportion.

    Tests H0: p = p_null vs H1: p != p_null

    Args:
        successes: number of successes
        total: total trials
        p_null: null hypothesis proportion
        alpha: significance level

    Returns:
        (z_stat, p_value, reject_null) where reject_null = (p_value < alpha)
    """
    p_hat = successes / total
    se = np.sqrt(p_null * (1 - p_null) / total)     ## Standard Error
    z_stat = (p_hat - p_null) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value, p_value < alpha


def bootstrap_ci(
    statistic_fn,
    data: np.ndarray | pd.Series,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Nonparametric bootstrap confidence interval.

    Args:
        statistic_fn: function that takes data array and returns a scalar
        data: 1D array of observations
        n_bootstrap: number of bootstrap samples
        alpha: significance level (returns (alpha/2, 1-alpha/2) quantiles)

    Returns:
        (ci_lower, ci_upper)
    """
    data = np.asarray(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_fn(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    ci_lower = np.quantile(bootstrap_stats, alpha / 2)
    ci_upper = np.quantile(bootstrap_stats, 1 - alpha / 2)
    return ci_lower, ci_upper


def autocorr_profile(series: pd.Series | np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Autocorrelation function.

    Args:
        series: 1D time series
        max_lag: maximum lag to compute

    Returns:
        array of autocorrelations at lags 1..max_lag
    """
    series = np.asarray(series)
    acf_vals = []
    mean = series.mean()
    c0 = np.mean((series - mean) ** 2)

    for lag in range(1, max_lag + 1):
        c_lag = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
        acf_vals.append(c_lag / c0 if c0 != 0 else 0)

    return np.array(acf_vals)


def variance_ratio(series: pd.Series | np.ndarray, lag: int = 2) -> float:
    """
    Variance ratio test: Var(X_t + X_{t+1} + ... + X_{t+lag-1}) / (lag * Var(X_t))

    VR ~ 1: random walk
    VR < 1: mean-reverting
    VR > 1: trending/momentum

    Args:
        series: 1D price or return series
        lag: the lag at which to compute VR

    Returns:
        variance ratio value
    """
    series = np.asarray(series)
    returns = np.diff(series)

    var_1 = np.var(returns, ddof=1)

    if lag > 1:
        # Compute lagged differences: X_t - X_{t-lag}
        lagged_returns = series[lag:] - series[:-lag]
        var_lag = np.var(lagged_returns, ddof=1)
    else:
        var_lag = var_1

    if var_1 == 0:
        return 1.0
    return var_lag / (lag * var_1)


def rolling_variance_ratio(series: pd.Series, lag: int = 2, window: int = 50) -> pd.Series:
    """
    Rolling variance ratio (mean reversion indicator).

    Returns:
        Series of VR values, indexed same as input
    """
    vr_vals = []
    for i in range(len(series) - window):
        vr = variance_ratio(series.iloc[i : i + window], lag=lag)
        vr_vals.append(vr)

    # Pad with NaN to match input length
    vr_vals = [np.nan] * (window - 1) + vr_vals + [np.nan] * (len(series) - len(vr_vals) - window + 1)
    return pd.Series(vr_vals, index=series.index)


def hurst_exponent(series: np.ndarray | pd.Series, lags: list[int] | None = None) -> float:
    """
    Estimate Hurst exponent using rescaled range analysis.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending

    Args:
        series: 1D time series
        lags: lags at which to compute rescaled range (default: [10, 20, 30, ..., min(len/2, 200)])

    Returns:
        estimated Hurst exponent
    """
    series = np.asarray(series)
    if lags is None:
        lags = [10, 20, 30, 50, 100, min(len(series) // 2, 200)]

    tau = []
    for lag in lags:
        # Divide series into chunks of size lag
        chunks = [series[i : i + lag] for i in range(0, len(series) - lag + 1, lag)]
        if len(chunks) < 2:
            continue

        # Compute rescaled range for each chunk
        rs_list = []
        for chunk in chunks:
            mean_centered = chunk - chunk.mean()
            Y = np.cumsum(mean_centered)
            R = Y.max() - Y.min()
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            tau.append(np.mean(rs_list))

    if len(tau) < 2:
        return 0.5

    # Log-log regression to estimate H
    log_lags = np.log(lags[: len(tau)])
    log_tau = np.log(tau)
    H = np.polyfit(log_lags, log_tau, 1)[0]
    return H
