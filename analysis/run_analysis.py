"""
One-command entry point for analysis: discover structure and alpha in any product.

Usage:
    python analysis/run_analysis.py --data-dir data/
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Support both: python -m analysis.run_analysis and python analysis/run_analysis.py
try:
    from .loader import load_round, products_in_round, load_ground_truth_fv
    from .fv_estimator import fv_weighted_mid, fv_microprice, fv_cross_validate, characterize_fv
    from .alpha_signals import (
        signal_ofi,
        signal_book_pressure,
        signal_spread_change,
        signal_trade_flow,
        signal_rolling_variance_ratio,
        signal_volume_imbalance,
        signal_spread_regime,
    )
    from .stat_utils import half_life
except ImportError:
    from loader import load_round, products_in_round, load_ground_truth_fv
    from fv_estimator import fv_weighted_mid, fv_microprice, fv_cross_validate, characterize_fv
    from alpha_signals import (
        signal_ofi,
        signal_book_pressure,
        signal_spread_change,
        signal_trade_flow,
        signal_rolling_variance_ratio,
        signal_volume_imbalance,
        signal_spread_regime,
    )
    from stat_utils import half_life


def compute_price_stats(product_prices: pd.DataFrame) -> dict:
    """Price distribution, spread regime, autocorrelation, and mean-reversion speed."""
    mid = (product_prices["bid_price_1"] + product_prices["ask_price_1"]) / 2
    spread = product_prices["ask_price_1"] - product_prices["bid_price_1"]
    returns = mid.diff().dropna()

    # Spread regime: unique values reveal bot configurations
    spread_unique = sorted(spread.unique().tolist())

    # Return autocorrelation at key lags (negative = mean-reverting)
    autocorrs = {}
    for lag in [1, 2, 5, 10]:
        autocorrs[lag] = returns.autocorr(lag) if len(returns) > lag else 0.0

    # Mean reversion half-life (ticks)
    hl = half_life(mid.values)

    return {
        "mid_mean": mid.mean(),
        "mid_std": mid.std(),
        "mid_min": mid.min(),
        "mid_max": mid.max(),
        "mid_range": mid.max() - mid.min(),
        "spread_mean": spread.mean(),
        "spread_median": spread.median(),
        "spread_min": spread.min(),
        "spread_max": spread.max(),
        "spread_std": spread.std(),
        "spread_unique": spread_unique,
        "return_mean": returns.mean(),
        "return_std": returns.std(),
        "return_skew": returns.skew(),
        "return_kurtosis": returns.kurtosis(),
        "autocorr": autocorrs,
        "half_life": hl,
    }


def compute_fv_agreement(product_prices: pd.DataFrame, trades_df: pd.DataFrame, ground_truth_fv_series: pd.Series = None) -> dict:
    """How much do the 4 FV estimators agree?"""
    fv_results = fv_cross_validate(product_prices, trades_df, ground_truth_fv=ground_truth_fv_series)
    # fv_regression is only present when a ground truth was supplied (see fv_cross_validate)
    fv_cols = [c for c in ["fv_wmid", "fv_micro", "fv_regression", "fv_trade"]
               if c in fv_results.columns]

    # Mean absolute disagreement between estimators
    pairwise_diffs = []
    for i in range(len(fv_cols)):
        for j in range(i + 1, len(fv_cols)):
            diff = (fv_results[fv_cols[i]] - fv_results[fv_cols[j]]).abs().mean()
            pairwise_diffs.append((fv_cols[i], fv_cols[j], diff))

    # Which estimator has lowest variance (most stable)?
    estimator_std = {col: fv_results[col].std() for col in fv_cols}
    most_stable = min(estimator_std, key=estimator_std.get)

    return {
        "consensus": fv_results["fv_consensus"],
        "disagreement_mean": fv_results["fv_disagreement"].mean(),
        "disagreement_max": fv_results["fv_disagreement"].max(),
        "pairwise_diffs": pairwise_diffs,
        "estimator_std": estimator_std,
        "most_stable": most_stable,
        "fv_results": fv_results,
    }


def compute_signal_quality(
    signals: dict[str, pd.Series], fv_consensus: pd.Series, lookaheads: list[int] = [1, 5, 10, 20]
) -> dict:
    """
    For each signal, compute correlation with forward returns at multiple horizons.
    This tells you which signals are actually predictive and at what timescale.
    """
    fv_clean = fv_consensus.dropna()
    quality = {}

    for sig_name, sig in signals.items():
        sig_quality = {"horizons": {}}

        for h in lookaheads:
            fwd_return = fv_clean.shift(-h) - fv_clean
            corr = sig.corr(fwd_return)
            sig_quality["horizons"][h] = corr if pd.notna(corr) else 0.0

        # Best horizon: which lookahead gives strongest correlation?
        best_h = max(sig_quality["horizons"], key=lambda h: abs(sig_quality["horizons"][h]))
        sig_quality["best_horizon"] = best_h
        sig_quality["best_corr"] = sig_quality["horizons"][best_h]

        # Signal rating
        abs_corr = abs(sig_quality["best_corr"])
        if abs_corr > 0.10:
            sig_quality["rating"] = "STRONG"
        elif abs_corr > 0.05:
            sig_quality["rating"] = "MODERATE"
        elif abs_corr > 0.02:
            sig_quality["rating"] = "WEAK"
        else:
            sig_quality["rating"] = "NOISE"

        quality[sig_name] = sig_quality

    return quality


def analyze_product(product: str, prices_df: pd.DataFrame, trades_df: pd.DataFrame, ground_truth_fv_dict: dict = None) -> dict:
    """Full analysis of a single product."""
    product_prices = prices_df[prices_df["product"] == product].copy()

    if product_prices.empty:
        return {"product": product, "status": "empty", "message": "No price data found"}

    # Price distribution
    price_stats = compute_price_stats(product_prices)

    # FV estimation and agreement (with optional ground truth from backtester)
    ground_truth_fv_series = None
    if ground_truth_fv_dict and product in ground_truth_fv_dict:
        ground_truth_fv_series = ground_truth_fv_dict[product]

    fv_info = compute_fv_agreement(product_prices, trades_df, ground_truth_fv_series=ground_truth_fv_series)
    fv_consensus = fv_info["consensus"]

    # Characterize FV
    fv_char = characterize_fv(fv_consensus)

    # Generate alpha signals (including new ones)
    # Reset indices so all signals + mid price align for correlation
    signals_raw = {
        "Vol Imbalance": signal_volume_imbalance(product_prices),
        "Spread Regime": signal_spread_regime(product_prices),
        "OFI": signal_ofi(product_prices),
        "Pressure": signal_book_pressure(product_prices),
        "Spread Chg": signal_spread_change(product_prices),
        "Trade Flow": signal_trade_flow(trades_df, product_prices),
        "Regime": signal_rolling_variance_ratio(fv_consensus),
    }
    signals = {k: v.reset_index(drop=True) for k, v in signals_raw.items()}

    # Signal quality: correlate against RAW MID returns (not fv_consensus which is noisy)
    mid_price = ((product_prices["bid_price_1"] + product_prices["ask_price_1"]) / 2).reset_index(drop=True)
    signal_quality = compute_signal_quality(signals, mid_price)

    # Recommendation
    fv_type = fv_char["fv_type"]
    if fv_type == "static":
        recommendation = f"STATIC product. Use StaticTrader pattern. FV ~ {fv_consensus.mean():.1f}"
    elif fv_type == "mean_reverting":
        recommendation = "MEAN-REVERTING. Quote aggressively around FV, trade inventory reversions."
    elif fv_type == "random_walk":
        recommendation = "RANDOM WALK. Use DynamicTrader (wall mid). Focus on microstructure edges."
    else:
        recommendation = "TRENDING. Lean into momentum, consider directional bias."

    return {
        "product": product,
        "status": "ok",
        "price_stats": price_stats,
        "fv_characterization": fv_char,
        "fv_agreement": {
            "disagreement_mean": fv_info["disagreement_mean"],
            "disagreement_max": fv_info["disagreement_max"],
            "most_stable": fv_info["most_stable"],
            "estimator_std": fv_info["estimator_std"],
        },
        "signal_quality": signal_quality,
        "recommendation": recommendation,
        "price_data_points": len(product_prices),
    }


# ── Printing ─────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n  ┌─ {title}")

def print_kv(key: str, value, indent=4):
    prefix = " " * indent
    print(f"{prefix}{key:20s}: {value}")

def print_analysis(analysis: dict):
    """Pretty-print a full product analysis."""
    if analysis["status"] != "ok":
        print(f"  Status: {analysis['message']}")
        return

    ps = analysis["price_stats"]
    fv = analysis["fv_characterization"]
    fa = analysis["fv_agreement"]
    sq = analysis["signal_quality"]

    # Price Distribution
    print_section("PRICE DISTRIBUTION")
    print_kv("Mid price", f"{ps['mid_mean']:.2f}  (range: {ps['mid_min']:.0f} – {ps['mid_max']:.0f}, σ={ps['mid_std']:.2f})")
    print_kv("Spread", f"mean={ps['spread_mean']:.2f}  median={ps['spread_median']:.1f}  max={ps['spread_max']:.0f}")
    print_kv("Spread values", f"{ps['spread_unique']}")
    print_kv("Returns", f"μ={ps['return_mean']:.4f}  σ={ps['return_std']:.4f}  skew={ps['return_skew']:.3f}  kurt={ps['return_kurtosis']:.1f}")

    # Microstructure
    print_section("MICROSTRUCTURE")
    ac = ps["autocorr"]
    ac_str = "  ".join(f"lag{k}={v:+.3f}" for k, v in ac.items())
    print_kv("Return autocorr", ac_str)
    hl = ps["half_life"]
    hl_str = f"{hl:.1f} ticks" if hl < 1e6 else "∞ (no reversion)"
    print_kv("Half-life", hl_str)
    if ac.get(1, 0) < -0.3:
        print(f"      → FAST mean reversion (lag-1 AC = {ac[1]:+.3f}). Quote tight, fills revert quickly.")

    # FV Characterization
    print_section("FAIR VALUE CHARACTERIZATION")
    type_str = fv["fv_type"].upper()
    print_kv("Type", type_str)
    print_kv("Coeff of Variation", f"{fv['cv']:.4f}%")
    print_kv("Variance Ratio (2)", f"{fv['vr_2']:.4f}  {'< 1 → mean-reverting' if fv['vr_2'] < 0.95 else '≈ 1 → random walk' if fv['vr_2'] < 1.05 else '> 1 → trending'}")
    print_kv("Hurst (returns)", f"{fv['hurst']:.4f}  {'< 0.5 → mean-reverting' if fv['hurst'] < 0.45 else '≈ 0.5 → RW' if fv['hurst'] < 0.55 else '> 0.5 → persistent'}")
    print_kv("ADF test", f"p={fv['adf_pvalue']:.4f}  {'→ STATIONARY (mean-reverting)' if fv['adf_stationary'] else '→ non-stationary'}")

    # FV Estimator Agreement
    print_section("FV ESTIMATOR AGREEMENT")
    print_kv("Mean disagreement", f"{fa['disagreement_mean']:.2f}")
    print_kv("Max disagreement", f"{fa['disagreement_max']:.2f}")
    print_kv("Most stable", fa["most_stable"])
    for est, std in sorted(fa["estimator_std"].items(), key=lambda x: x[1]):
        marker = " ★" if est == fa["most_stable"] else ""
        print_kv(f"  {est}", f"σ={std:.4f}{marker}")

    # Signal Quality
    print_section("ALPHA SIGNAL QUALITY")
    print(f"    {'Signal':15s} {'Rating':10s} {'Best H':>8s} {'Corr':>8s}   {'1-step':>8s} {'5-step':>8s} {'10-step':>8s} {'20-step':>8s}")
    print(f"    {'─'*15} {'─'*10} {'─'*8} {'─'*8}   {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for sig_name, sq_info in sq.items():
        h = sq_info["horizons"]
        rating = sq_info["rating"]
        best_h = sq_info["best_horizon"]
        best_c = sq_info["best_corr"]
        rating_color = {"STRONG": "***", "MODERATE": "**", "WEAK": "*", "NOISE": ""}[rating]
        print(f"    {sig_name:15s} {rating:10s} {best_h:>8d} {best_c:>+8.4f}   {h.get(1,0):>+8.4f} {h.get(5,0):>+8.4f} {h.get(10,0):>+8.4f} {h.get(20,0):>+8.4f}  {rating_color}")

    # Recommendation
    print_section("RECOMMENDATION")
    print(f"    {analysis['recommendation']}")

    # Actionable signals
    usable = [name for name, info in sq.items() if info["rating"] in ("STRONG", "MODERATE")]
    if usable:
        print(f"    Actionable signals: {', '.join(usable)}")
    else:
        print(f"    No signals above noise threshold — rely on pure market making.")

    print(f"\n    Data points: {analysis['price_data_points']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze IMC Prosperity round data for structure and alpha")
    parser.add_argument("--data-dir", type=str, default="Data", help="Directory containing prices_*.csv and trades_*.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print(f"\n{'═'*80}")
    print(f"  IMC PROSPERITY ANALYSIS — {data_dir}")
    print(f"{'═'*80}")

    try:
        prices_df, trades_df = load_round(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load ground truth FV from backtester if available
    ground_truth_fv_dict = load_ground_truth_fv(prices_df=prices_df)
    if ground_truth_fv_dict:
        print(f"\n  Loaded ground_truth_fv for: {', '.join(ground_truth_fv_dict.keys())}")
    else:
        print(f"\n  No ground_truth_fv found (OK if running on raw CSV data)")

    products = products_in_round(prices_df)
    print(f"\n  Products found: {', '.join(products)}")
    print(f"  Price rows: {len(prices_df):,}  |  Trade rows: {len(trades_df):,}")

    results = {}
    for product in products:
        print(f"\n{'─'*80}")
        print(f"  {product}")
        print(f"{'─'*80}")

        analysis = analyze_product(product, prices_df, trades_df, ground_truth_fv_dict=ground_truth_fv_dict)
        results[product] = analysis
        print_analysis(analysis)

    # Cross-product correlation (if 2+ products)
    if len(products) >= 2:
        print(f"\n{'─'*80}")
        print(f"  CROSS-PRODUCT CORRELATION")
        print(f"{'─'*80}")
        mids = {}
        for product in products:
            pp = prices_df[prices_df["product"] == product]
            mids[product] = ((pp["bid_price_1"] + pp["ask_price_1"]) / 2).reset_index(drop=True).diff()

        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                p1, p2 = products[i], products[j]
                r1, r2 = mids[p1], mids[p2]
                min_len = min(len(r1), len(r2))
                r1, r2 = r1.iloc[:min_len], r2.iloc[:min_len]
                contemp = r1.corr(r2)
                print(f"\n    {p1} vs {p2}:")
                print(f"      Contemporaneous: {contemp:+.4f}")
                for lag in [1, 2, 5, 10]:
                    lead_1 = r1.corr(r2.shift(lag))
                    lead_2 = r2.corr(r1.shift(lag))
                    print(f"      {p1} leads by {lag}: {lead_1:+.4f}   {p2} leads by {lag}: {lead_2:+.4f}")
                if abs(contemp) < 0.05:
                    print(f"      → INDEPENDENT. No cross-product edge.")
                else:
                    print(f"      → CORRELATED. Consider cross-product signals.")

    print(f"\n{'═'*80}\n")


if __name__ == "__main__":
    main()
