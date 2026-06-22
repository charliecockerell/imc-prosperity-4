"""
Generic CSV data loader for any round/product structure.

Auto-discovers prices and trades CSVs, detects products, adds global timestamps.
Works with any round naming convention (prices_round_X_day_Y.csv).
Optionally loads ground_truth_fv from backtester fill_log.json.
"""

from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import json


def load_round(data_dir: str | Path = "Data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all price and trade CSVs from a directory.

    Auto-discovers any prices_round_*_day_*.csv and trades_round_*_day_*.csv files.
    Concatenates across days, adds global_ts column (day offset applied).

    Returns:
        (prices_df, trades_df) — indexed by global_ts
    """
    data_dir = Path(data_dir)

    # Load prices
    price_files = sorted(data_dir.glob("prices_round_*_day_*.csv"))
    if not price_files:
        raise FileNotFoundError(f"No price CSVs found in {data_dir}")

    prices = []
    max_ts = 0

    for fpath in price_files:
        df = pd.read_csv(fpath, sep=";")
        if "day" not in df.columns:
            # Extract day from filename: prices_round_0_day_-1.csv -> -1
            parts = fpath.stem.split("_")
            day = int(parts[-1])
            df["day"] = day

        # Global timestamp: offset by day (assuming 1M ticks per day)
        TICKS_PER_DAY = 1_000_000
        day_offset = (df["day"].max() + 2) * TICKS_PER_DAY  # Assume day -2 is day 0
        df["global_ts"] = df["timestamp"] + day_offset
        max_ts = max(max_ts, df["global_ts"].max())
        prices.append(df)

    prices_df = pd.concat(prices, ignore_index=True).sort_values("global_ts").reset_index(drop=True)

    # Load trades
    trade_files = sorted(data_dir.glob("trades_round_*_day_*.csv")) ## .glob returns directories/files matching a pattern
    trades = []

    ## We do this load process twice, I think this should be a function.

    for fpath in trade_files:
        df = pd.read_csv(fpath, sep=";")
        if "day" not in df.columns:
            parts = fpath.stem.split("_")
            day = int(parts[-1])
            df["day"] = day

        # Apply same day offset
        TICKS_PER_DAY = 1_000_000
        day_offset = (df["day"].max() + 2) * TICKS_PER_DAY
        df["global_ts"] = df["timestamp"] + day_offset
        trades.append(df)

    if trades:
        trades_df = pd.concat(trades, ignore_index=True).sort_values("global_ts").reset_index(drop=True)
    else:
        trades_df = pd.DataFrame()

    return prices_df, trades_df


def products_in_round(prices_df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique products in the prices dataframe."""
    if "product" in prices_df.columns:
        return sorted(prices_df["product"].unique())
    return []


def load_ground_truth_fv(fill_log_path: str | Path = "visualisation/fill_log.json",
                         prices_df: pd.DataFrame = None) -> Dict[str, pd.Series]:
    """
    Load ground_truth_fv from backtester fill_log.json and align with prices_df.

    Returns:
        Dict mapping product -> Series of ground_truth_fv values indexed by global_ts.
        Empty dict if file not found or no ground_truth_fv data.

    Args:
        fill_log_path: path to backtester output (relative to repo root)
        prices_df: prices dataframe (to align indices and extract day/ts)
    """
    fill_log_path = Path(fill_log_path)

    # If relative path doesn't exist, try from repo root
    if not fill_log_path.exists() and not fill_log_path.is_absolute():
        # Try to find it relative to repo root
        repo_root = Path(__file__).parent.parent
        alt_path = repo_root / fill_log_path
        if alt_path.exists():
            fill_log_path = alt_path

    if not fill_log_path.exists():
        return {}

    try:
        with open(fill_log_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}

    # Handle old format (list of fills) vs new format (dict with fills + ground_truth_fv)
    if isinstance(data, list):
        # Old format, no ground_truth_fv
        return {}

    ground_truth_fv_raw = data.get("ground_truth_fv", {})
    if not ground_truth_fv_raw:
        return {}

    # Reconstruct ground_truth_fv Series for each product
    ground_truth_fv = {}

    for product, ts_dict in ground_truth_fv_raw.items():
        # ts_dict is {day_ts: price}, e.g., {"-2_0": 10000.5, "-2_1": 10001.2}
        fv_values = []
        fv_indices = []

        for day_ts_str, price in ts_dict.items():
            parts = day_ts_str.split('_')
            day, ts = int(parts[0]), int(parts[1])

            # Compute global_ts same way as loader
            TICKS_PER_DAY = 1_000_000
            day_offset = (day + 2) * TICKS_PER_DAY
            global_ts = ts + day_offset

            fv_indices.append(global_ts)
            fv_values.append(price)

        # Create Series indexed by global_ts
        if fv_indices:
            series = pd.Series(fv_values, index=fv_indices, name=f"{product}_ground_truth_fv")
            ground_truth_fv[product] = series.sort_index()

    return ground_truth_fv
