"""
Adverse selection analyzer: measures post-fill price movement.

For each fill, looks at where the mid price goes 1/5/10/20 ticks after.
If price moves against you after a fill, that fill was "adversely selected"
(you got picked off by someone with better information).

Usage:
    python analysis/adverse_selection.py --data-dir data/ --fill-log visualisation/fill_log.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .loader import load_round
except ImportError:
    from loader import load_round


def load_fills(fill_log_path: str) -> list[dict]:
    with open(fill_log_path) as f:
        return json.load(f)


def analyze_adverse_selection(
    fills: list[dict],
    prices_df: pd.DataFrame,
    horizons: list[int] = [1, 5, 10, 20, 50],
) -> pd.DataFrame:
    """
    For each fill, compute the mid price move at various horizons after the fill.

    A BUY fill is good if mid goes UP after (we bought cheap).
    A SELL fill is good if mid goes DOWN after (we sold dear).

    Returns DataFrame with one row per fill, columns:
      side, price, qty, ts, mid_at_fill, edge_at_fill,
      pnl_1, pnl_5, pnl_10, pnl_20, pnl_50 (signed PnL per unit)
    """
    # Build mid price lookup by (day, timestamp)
    mid = (prices_df["bid_price_1"] + prices_df["ask_price_1"]) / 2
    prices_df = prices_df.copy()
    prices_df["mid"] = mid

    results = []

    for product in prices_df["product"].unique():
        product_prices = prices_df[prices_df["product"] == product].reset_index(drop=True)
        product_fills = [f for f in fills if f["product"] == product]

        if not product_fills:
            continue

        # Build timestamp → index mapping for fast lookup
        ts_list = product_prices["global_ts"].values
        mid_list = product_prices["mid"].values

        TICKS_PER_DAY = 1_000_000
        for fill in product_fills:
            day = fill["day"]
            day_offset = (day + 2) * TICKS_PER_DAY
            global_ts = fill["ts"] + day_offset

            # Find nearest index
            idx = np.searchsorted(ts_list, global_ts)
            if idx >= len(ts_list):
                idx = len(ts_list) - 1

            mid_at_fill = mid_list[idx]
            fill_price = fill["price"]
            side = fill["side"]

            # Edge at fill: how far from mid did we trade?
            # Positive = good (bought below mid or sold above mid)
            if side == "BUY":
                edge = mid_at_fill - fill_price
            else:
                edge = fill_price - mid_at_fill

            row = {
                "product": product,
                "side": side,
                "price": fill_price,
                "qty": fill["qty"],
                "ts": global_ts,
                "mid_at_fill": mid_at_fill,
                "edge_at_fill": edge,
            }

            # Forward PnL at each horizon
            for h in horizons:
                future_idx = idx + h
                if future_idx < len(mid_list):
                    future_mid = mid_list[future_idx]
                    if side == "BUY":
                        # Bought at fill_price, now worth future_mid
                        pnl = future_mid - fill_price
                    else:
                        # Sold at fill_price, now worth future_mid (short)
                        pnl = fill_price - future_mid
                else:
                    pnl = np.nan

                row[f"pnl_{h}"] = pnl

            results.append(row)

    return pd.DataFrame(results)


def print_report(df: pd.DataFrame, horizons: list[int] = [1, 5, 10, 20, 50]):
    """Print adverse selection report."""

    for product in sorted(df["product"].unique()):
        pdf = df[df["product"] == product]

        print(f"\n{'─'*80}")
        print(f"  {product} — ADVERSE SELECTION REPORT")
        print(f"{'─'*80}")

        for side in ["BUY", "SELL"]:
            sdf = pdf[pdf["side"] == side]
            if sdf.empty:
                continue

            print(f"\n  {side}S ({len(sdf)} fills)")
            print(f"    Avg edge at fill: {sdf['edge_at_fill'].mean():+.2f} ticks")

            print(f"\n    {'Horizon':>10s} {'Avg PnL':>10s} {'Median':>10s} {'%Profitable':>12s} {'Toxic%':>8s}")
            print(f"    {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*8}")

            for h in horizons:
                col = f"pnl_{h}"
                vals = sdf[col].dropna()
                if vals.empty:
                    continue
                avg = vals.mean()
                med = vals.median()
                pct_profit = (vals > 0).mean() * 100
                pct_toxic = (vals < 0).mean() * 100

                marker = ""
                if pct_toxic > 60:
                    marker = " !! TOXIC"
                elif pct_profit > 60:
                    marker = " ** GOOD"

                print(f"    {h:>10d} {avg:>+10.2f} {med:>+10.2f} {pct_profit:>11.1f}% {pct_toxic:>7.1f}%{marker}")

        # Worst fills: largest adverse moves at 10-tick horizon
        print(f"\n  WORST 5 FILLS (by 10-tick adverse move):")
        worst = pdf.nsmallest(5, "pnl_10")
        for _, row in worst.iterrows():
            print(f"    {row['side']:4s} {row['qty']:.0f}x @ {row['price']:.0f}  "
                  f"edge={row['edge_at_fill']:+.1f}  10-tick PnL={row['pnl_10']:+.1f}")

        # Best fills
        print(f"\n  BEST 5 FILLS (by 10-tick profit):")
        best = pdf.nlargest(5, "pnl_10")
        for _, row in best.iterrows():
            print(f"    {row['side']:4s} {row['qty']:.0f}x @ {row['price']:.0f}  "
                  f"edge={row['edge_at_fill']:+.1f}  10-tick PnL={row['pnl_10']:+.1f}")


def main():
    parser = argparse.ArgumentParser(description="Adverse selection analysis")
    parser.add_argument("--data-dir", type=str, default="Data")
    parser.add_argument("--fill-log", type=str, default="visualisation/fill_log.json")
    args = parser.parse_args()

    print(f"\n{'═'*80}")
    print(f"  ADVERSE SELECTION ANALYSIS")
    print(f"{'═'*80}")

    prices_df, trades_df = load_round(args.data_dir)
    fills = load_fills(args.fill_log)
    print(f"  Loaded {len(fills)} fills from {args.fill_log}")

    df = analyze_adverse_selection(fills, prices_df)
    print_report(df)

    print(f"\n{'═'*80}\n")


if __name__ == "__main__":
    main()
