"""
Alpha signal generation: order flow, pressure, regime detection.

Each signal returns a Series normalized ~[-1, 1].
Positive = buy signal, negative = sell signal.
"""

import pandas as pd
import numpy as np


def signal_volume_imbalance(prices_df: pd.DataFrame) -> pd.Series:
    """
    Raw volume imbalance at L1: bid_vol / (bid_vol + ask_vol).

    """
    bid_vol = prices_df["bid_volume_1"]
    ask_vol = prices_df["ask_volume_1"]
    total = bid_vol + ask_vol
    total = total.replace(0, np.nan)

    # Center at 0: imbalance of 0 means balanced book
    imbalance = (bid_vol - ask_vol) / total
    return imbalance.fillna(0)


def signal_spread_regime(prices_df: pd.DataFrame) -> pd.Series:
    """
    Spread regime indicator: normalised spread relative to its median.

    """
    spread = prices_df["ask_price_1"] - prices_df["bid_price_1"]
    median_spread = spread.median()

    if median_spread == 0:
        return pd.Series(0, index=prices_df.index)

    # Normalise: 0 = median spread, positive = wider, negative = tighter
    regime = (spread - median_spread) / median_spread
    return regime


def signal_ofi(prices_df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Order Flow Imbalance: sum of bid/ask volume changes.
    OFI = sum(change in bid_vol - change in ask_vol) over rolling window
    Positive OFI = buying pressure

    """
    bid_change = prices_df["bid_volume_1"].diff()
    ask_change = prices_df["ask_volume_1"].diff()
    raw_ofi = (bid_change - ask_change).rolling(window).sum()

    # Normalize to ~[-1, 1]
    ofi_std = raw_ofi.std()
    if ofi_std > 0:
        return raw_ofi / (3 * ofi_std)  # 3-sigma normalization
    else:
        return pd.Series(0, index=prices_df.index)


def signal_book_pressure(prices_df: pd.DataFrame) -> pd.Series:
    """
    Book Pressure Asymmetry: (total bid depth - total ask depth) / total depth
    Positive = more bid liquidity = upward pressure

    """
    bid_depth = prices_df["bid_volume_1"] + prices_df["bid_volume_2"] + prices_df["bid_volume_3"]
    ask_depth = prices_df["ask_volume_1"] + prices_df["ask_volume_2"] + prices_df["ask_volume_3"]

    total_depth = bid_depth + ask_depth
    total_depth = total_depth.replace(0, np.nan)

    pressure = (bid_depth - ask_depth) / total_depth
    return pressure.fillna(0)


def signal_trade_flow(
    trades_df: pd.DataFrame, prices_df: pd.DataFrame, window: int = 20
) -> pd.Series:
    """
    Trade flow signal: classify trades as buys/sells, compute running ratio.
    Positive = more buying pressure

    """
    if trades_df.empty:
        return pd.Series(0, index=prices_df.index)

    # Classify trades: a trade at price P is:
    # - buy if P >= ask_1
    # - sell if P <= bid_1
    # - indeterminate otherwise (executed inside spread)

    # Merge trades with order book
    buy_count = 0
    sell_count = 0
    trade_signal = []

    for i, row in prices_df.iterrows():
        ts = row["global_ts"]
        bid1 = row["bid_price_1"]
        ask1 = row["ask_price_1"]

        # Trades near this timestamp
        ts_trades = trades_df[trades_df["global_ts"] == ts]
        for _, trade in ts_trades.iterrows():
            trade_price = trade["price"]
            if trade_price >= ask1:
                buy_count += 1
            elif trade_price <= bid1:
                sell_count += 1

        # Compute rolling ratio
        total = buy_count + sell_count
        if total >= window:
            buy_ratio = buy_count / total
            sell_ratio = sell_count / total
            # Signal: buy_ratio - 0.5 (centered at 0 when 50/50)
            signal_val = buy_ratio - 0.5
            trade_signal.append(signal_val * 2)  # Scale to ~[-1, 1]
        else:
            trade_signal.append(0)

    return pd.Series(trade_signal, index=prices_df.index).fillna(0)


def composite_alpha(
    prices_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    
    """
    Composite alpha signal: weighted sum of OFI, book pressure, and trade flow.

    Args:
        prices_df: order book prices
        trades_df: trade data
        weights: dict mapping signal name to weight. Default: equal weight.
                 Signals: ofi, pressure, trade
    Returns:
        Composite signal Series

    """
    if weights is None:
        weights = {"ofi": 1/3, "pressure": 1/3, "trade": 1/3}

    signals = {
        "ofi":      signal_ofi(prices_df),
        "pressure": signal_book_pressure(prices_df),
        "trade":    signal_trade_flow(trades_df, prices_df),
    }

    composite = pd.Series(0.0, index=prices_df.index)
    total_weight = sum(weights.values())
    for name, sig in signals.items():
        w = weights.get(name, 0)
        composite += (sig * w / total_weight).fillna(0)

    return composite
