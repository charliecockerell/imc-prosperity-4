"""
Generic fair value estimation using multiple methods.

"""

import pandas as pd
import numpy as np
from .stat_utils import variance_ratio, hurst_exponent, adf_test


def fv_weighted_mid(prices_df: pd.DataFrame) -> pd.Series:
    """
    Volume-weighted midpoint.

    wmid = (best_bid * ask_vol_1 + best_ask * bid_vol_1) / (bid_vol_1 + ask_vol_1)

    Corrects for order book imbalance. Better than unweighted mid when one side has more depth.

    """
    bid1 = prices_df["bid_price_1"]
    ask1 = prices_df["ask_price_1"]
    bid_vol1 = prices_df["bid_volume_1"]
    ask_vol1 = prices_df["ask_volume_1"]

    total_vol = bid_vol1 + ask_vol1
    total_vol = total_vol.replace(0, np.nan)  # Avoid division by zero

    wmid = (bid1 * ask_vol1 + ask1 * bid_vol1) / total_vol
    return wmid


def fv_microprice(prices_df: pd.DataFrame) -> pd.Series:
    """
    Stoikov micro-price: imbalance-adjusted midpoint.

    microprice = ask_1 * imbalance + bid_1 * (1 - imbalance)
    where imbalance = bid_vol_1 / (bid_vol_1 + ask_vol_1)

    Better than wmid when book is asymmetric.
    """
    bid1 = prices_df["bid_price_1"]
    ask1 = prices_df["ask_price_1"]
    bid_vol1 = prices_df["bid_volume_1"]
    ask_vol1 = prices_df["ask_volume_1"]

    total_vol = bid_vol1 + ask_vol1
    total_vol = total_vol.replace(0, np.nan)

    imbalance = bid_vol1 / total_vol
    microprice = ask1 * imbalance + bid1 * (1 - imbalance)
    return microprice


def fv_regression(prices_df: pd.DataFrame, window: int = 200, ground_truth_fv: pd.Series = None) -> pd.Series | None:
    """
    Multi-level regression FV estimate — REQUIRES a ground-truth target.

    Rolling OLS of a known fair value on the six book levels (bid1-3, ask1-3):
    it learns how the true value loads on the visible book, then predicts FV
    from the current book.

    This only does anything when `ground_truth_fv` is supplied (e.g. the
    hold-one settlement series from the backtester). When no target is given 
    we return None and let the caller skip it.

    Args:
        prices_df: price data with bid/ask levels
        window: rolling window size
        ground_truth_fv: REQUIRED Series of true settlement prices. If None, the
                         estimator is not applicable and returns None.
    """
    if ground_truth_fv is None:
        return None

    bid_cols = ["bid_price_1", "bid_price_2", "bid_price_3"]
    ask_cols = ["ask_price_1", "ask_price_2", "ask_price_3"]
    all_level_cols = bid_cols + ask_cols
    target = ground_truth_fv

    fv_vals = [np.nan] * (window - 1)  # Start with NaN padding

    for i in range(window - 1, len(prices_df)):
        X = prices_df[all_level_cols].iloc[i - window + 1 : i + 1].values
        y = target.iloc[i - window + 1 : i + 1].values

        # Skip if any NaN
        if np.isnan(X).any() or np.isnan(y).any():
            fv_vals.append(np.nan)
            continue

        # Use numpy lstsq: regress y on X (multivariate OLS)
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
            # FV estimate for current point
            X_current = np.column_stack([np.ones(1), prices_df[all_level_cols].iloc[i : i + 1].values])
            fv_hat = (X_current @ coeffs)[0]
            fv_vals.append(fv_hat)
        except np.linalg.LinAlgError:
            fv_vals.append(np.nan)

    return pd.Series(fv_vals, index=prices_df.index)


def fv_trade_anchor(trades_df: pd.DataFrame, prices_df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Uses recent trade prices as a noisy but unbiased FV anchor.
    For each price timestamp, uses the mean of recent trades before that time.

    Args:
        trades_df: trades dataframe with global_ts and price columns
        prices_df: prices dataframe with global_ts index
        lookback: lookback window parameter

    Returns:
        Series of FV estimates indexed by prices_df
    """
    if trades_df.empty:
        return pd.Series(np.nan, index=prices_df.index)

    lookback = max(10, lookback) if alpha > 0 else 20
    fv_vals = []

    for global_ts in prices_df["global_ts"]:
        # Trades before this timestamp
        recent_trades = trades_df[trades_df["global_ts"] <= global_ts].tail(lookback)
        if len(recent_trades) > 0:
            fv_vals.append(recent_trades["price"].mean())
        else:
            fv_vals.append(np.nan)

    return pd.Series(fv_vals, index=prices_df.index)


def fv_cross_validate(prices_df: pd.DataFrame, trades_df: pd.DataFrame, ground_truth_fv: pd.Series = None) -> pd.DataFrame:
    """
    Run the FV estimators and return cross-validation results.

    Args:
        prices_df: price data with bid/ask levels
        trades_df: trade data
        ground_truth_fv: optional Series of true settlement prices from hold-one simulation

    Returns:
        DataFrame with columns: fv_wmid, fv_micro, fv_trade, (fv_regression only if
        a ground truth was given,) fv_consensus, fv_disagreement
    """
    result = pd.DataFrame(index=prices_df.index)

    result["fv_wmid"] = fv_weighted_mid(prices_df)
    result["fv_micro"] = fv_microprice(prices_df)
    result["fv_trade"] = fv_trade_anchor(trades_df, prices_df, lookback=20)

    # Regression only contributes when it has a real target to learn from
    reg = fv_regression(prices_df, window=200, ground_truth_fv=ground_truth_fv)
    if reg is not None:
        result["fv_regression"] = reg

    # Consensus / disagreement over whichever estimators are available
    method_cols = [c for c in ["fv_wmid", "fv_micro", "fv_regression", "fv_trade"]
                   if c in result.columns]
    result["fv_consensus"] = result[method_cols].mean(axis=1)
    result["fv_disagreement"] = result[method_cols].std(axis=1)

    return result


def characterize_fv(fv_series: pd.Series) -> dict:
    """
    Characterize a FV series: determine if static, random walk, mean-reverting, or trending.

    Uses three independent tests and majority-votes:
      1. Coefficient of variation (CV) — detects static products (price stays near a fixed level)
      2. ADF stationarity test — rejects unit root → mean-reverting
      3. Variance ratio at lag 2 — VR<1 mean-reverting, VR≈1 random walk, VR>1 trending
      4. Hurst exponent on RETURNS (not levels) — H<0.5 mean-reverting, H≈0.5 RW, H>0.5 trending

    Args:
        fv_series: 1D series of FV estimates

    Returns:
        dict with fv_type, sigma, cv, vr_2, hurst, adf_pvalue, adf_stationary
    """
    fv_clean = fv_series.dropna()
    returns = fv_clean.diff().dropna()

    if len(returns) < 50:
        return {"fv_type": "unknown", "sigma": np.nan, "cv": np.nan,
                "vr_2": np.nan, "hurst": np.nan, "adf_pvalue": np.nan, "adf_stationary": False}

    sigma = returns.std()
    mean_price = fv_clean.mean()
    price_std = fv_clean.std()

    # Coefficient of variation
    cv = (price_std / abs(mean_price)) * 100 if abs(mean_price) > 0 else np.inf

    # ADF test on levels: rejects null (unit root) → series is stationary → mean-reverting
    adf_stat, adf_p, adf_stationary = adf_test(fv_clean.values)

    # Variance ratio at lag 2 (computed on levels, uses returns internally)
    vr_2 = variance_ratio(fv_clean, lag=2)

    # Hurst on RETURNS (not levels) — avoids inflation from bounded price series
    H = hurst_exponent(returns.values)

    # --- Classification: vote across indicators ---
    if cv < 0.5:
        # Price stays within 0.5% of its mean → static product
        fv_type = "static"
    elif adf_stationary and vr_2 < 1.05:
        # ADF rejects unit root AND VR doesn't suggest trending → mean-reverting
        fv_type = "mean_reverting"
    elif vr_2 > 1.15 and H > 0.55:
        # Both VR and Hurst agree on trending
        fv_type = "trending"
    elif vr_2 < 0.85 or H < 0.45:
        # Either indicator says mean-reverting
        fv_type = "mean_reverting"
    elif 0.9 < vr_2 < 1.1 and 0.4 < H < 0.6:
        fv_type = "random_walk"
    else:
        # Default: VR is most reliable single indicator
        if vr_2 < 1.0:
            fv_type = "mean_reverting"
        else:
            fv_type = "trending"

    return {
        "fv_type": fv_type,
        "sigma": sigma,
        "cv": cv,
        "vr_2": vr_2,
        "hurst": H,
        "adf_pvalue": adf_p,
        "adf_stationary": adf_stationary,
    }
