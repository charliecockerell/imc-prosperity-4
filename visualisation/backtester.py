"""
IMC Prosperity backtester.

Fill model:
  1. Aggressive fills — our order crosses the current book immediately.
  2. Trade-based passive fills — when bots trade (market_trades CSV) at a price
     that crosses our passive limit, we fill at our limit price.
     e.g. bot trade at P <= our_bid → we fill as buyer at our_bid price.
     e.g. bot trade at P >= our_ask → we fill as seller at our_ask price.
     This is the main fill mechanism for inside-spread passive quoting.

Orders do NOT persist across ticks — each tick we place fresh orders.
"""

import pandas as pd
import json
import os
import sys
from collections import defaultdict

# ── MOCK DATA MODEL ────────────────────────────────────────────────────────────

class Order:
    def __init__(self, symbol, price, quantity):
        self.symbol   = symbol
        self.price    = price
        self.quantity = quantity

class OrderDepth:
    def __init__(self):
        self.buy_orders  = {}   # price -> volume (positive)
        self.sell_orders = {}   # price -> volume (negative, Prosperity format)

class TradingState:
    def __init__(self, timestamp, order_depths, position, traderData=""):
        self.timestamp    = timestamp
        self.order_depths = order_depths
        self.position     = position
        self.traderData   = traderData
        self.market_trades = {}
        self.own_trades    = {}
        self.observations  = None


# ── LOAD STRATEGY ──────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Model'))
import Model as _M
# Round 4 historical: 3 days starting TTE=4 (vouchers expire end of round 7).
if hasattr(_M, "INITIAL_TTE"):
    _M.INITIAL_TTE = 4.0
from Model import Trader


# ── CONFIG ─────────────────────────────────────────────────────────────────────

STRIKES_R3 = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
PRODUCTS = {
    "HYDROGEL_PACK":       200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in STRIKES_R3},
}

_repo_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER_PATH = os.path.join(_repo_root, "data/round4")

DAYS = [
    (os.path.join(FOLDER_PATH, "prices_round_4_day_1.csv"),
     os.path.join(FOLDER_PATH, "trades_round_4_day_1.csv"), 1),
    (os.path.join(FOLDER_PATH, "prices_round_4_day_2.csv"),
     os.path.join(FOLDER_PATH, "trades_round_4_day_2.csv"), 2),
    (os.path.join(FOLDER_PATH, "prices_round_4_day_3.csv"),
     os.path.join(FOLDER_PATH, "trades_round_4_day_3.csv"), 3),
]


# ── LOAD DATA ──────────────────────────────────────────────────────────────────

price_frames = []
trade_frames = []
for price_file, trade_file, day_id in DAYS:
    df = pd.read_csv(price_file, sep=';')
    df['day'] = day_id
    price_frames.append(df)
    if os.path.exists(trade_file):
        tf = pd.read_csv(trade_file, sep=';')
        tf['day'] = day_id
        trade_frames.append(tf)

prices_raw = pd.concat(price_frames, ignore_index=True)
prices_raw = prices_raw[prices_raw['product'].isin(PRODUCTS)]

trades_raw = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

print("Loaded data:")
for product in PRODUCTS:
    n = len(prices_raw[prices_raw['product'] == product])
    if n: print(f"  {product}: {n} price rows")

# Build trade lookup: (day, timestamp) -> {symbol -> [(price, qty)]}
# AND: (day, timestamp) -> {symbol -> [Trade(...)]} for state.market_trades
class _Trade:
    def __init__(self, symbol, price, quantity, buyer, seller, timestamp):
        self.symbol = symbol; self.price = price; self.quantity = quantity
        self.buyer = buyer; self.seller = seller; self.timestamp = timestamp

trade_groups = defaultdict(lambda: defaultdict(list))
market_trade_groups = defaultdict(lambda: defaultdict(list))
if not trades_raw.empty:
    for _, row in trades_raw.iterrows():
        sym = row['symbol']
        if sym in PRODUCTS:
            key = (row['day'], row['timestamp'])
            trade_groups[key][sym].append(
                (float(row['price']), int(row['quantity']))
            )
            buyer  = row.get('buyer', '')  if 'buyer' in row.index  else ''
            seller = row.get('seller', '') if 'seller' in row.index else ''
            if pd.isna(buyer):  buyer = ''
            if pd.isna(seller): seller = ''
            market_trade_groups[key][sym].append(_Trade(
                sym, float(row['price']), int(row['quantity']),
                str(buyer), str(seller), int(row['timestamp']),
            ))


# ── HELPERS ────────────────────────────────────────────────────────────────────

def build_order_depth(row):
    od = OrderDepth()
    for i in range(1, 4):
        bp = row.get(f"bid_price_{i}");  bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}");  av = row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv): od.buy_orders[int(bp)]  =  int(bv)
        if pd.notna(ap) and pd.notna(av): od.sell_orders[int(ap)] = -int(av)
    return od


all_keys = sorted(
    prices_raw[['day', 'timestamp']].drop_duplicates().itertuples(index=False, name=None)
)
price_groups = defaultdict(dict)
for _, row in prices_raw.iterrows():
    price_groups[(row['day'], row['timestamp'])][row['product']] = row


# ── BACKTEST ───────────────────────────────────────────────────────────────────

def run_backtest(verbose=True):
    trader     = Trader()
    position   = {p: 0 for p in PRODUCTS}
    cash       = 0.0
    traderData = ""
    fill_log   = []
    day_results = []

    days_ordered = sorted(set(day for day, _ in all_keys))

    for current_day in days_ordered:
        day_keys    = [(d, t) for d, t in all_keys if d == current_day]
        pnl_history = []

        for day, timestamp in day_keys:

            # ── BUILD BOOK ─────────────────────────────────────────────────────
            order_depths = {}
            for product, price_row in price_groups[(day, timestamp)].items():
                order_depths[product] = build_order_depth(price_row)

            # ── RUN MODEL ──────────────────────────────────────────────────────
            state = TradingState(
                timestamp=timestamp,
                order_depths=order_depths,
                position=position.copy(),
                traderData=traderData,
            )
            state.market_trades = dict(market_trade_groups.get((day, timestamp), {}))
            result, _, traderData = trader.run(state)

            # ── FILL ORDERS ────────────────────────────────────────────────────
            tick_trades = trade_groups.get((day, timestamp), {})

            for product, lim in PRODUCTS.items():
                orders = result.get(product, [])
                if not orders:
                    continue

                od = order_depths.get(product)
                ask_levels = {}
                bid_levels = {}
                if od:
                    for p, v in od.sell_orders.items(): ask_levels[p] = abs(v)
                    for p, v in od.buy_orders.items():  bid_levels[p] = abs(v)

                # Aggregate to single order per price
                agg = defaultdict(int)
                for o in orders: agg[o.price] += o.quantity
                buy_orders  = sorted([(p, q) for p, q in agg.items() if q > 0], key=lambda x: -x[0])
                sell_orders = sorted([(p, q) for p, q in agg.items() if q < 0], key=lambda x:  x[0])

                # ── AGGRESSIVE BUYS ──
                for price, qty in buy_orders:
                    remaining = qty
                    for ask_p in sorted(ask_levels):
                        if remaining <= 0 or price < ask_p: break
                        avail    = ask_levels[ask_p]
                        fill_qty = min(remaining, avail, lim - position[product])
                        if fill_qty > 0:
                            position[product] += fill_qty
                            cash -= fill_qty * price
                            ask_levels[ask_p] -= fill_qty
                            remaining -= fill_qty
                            fill_log.append({'day': int(day), 'ts': int(timestamp),
                                'product': product, 'side': 'BUY',
                                'price': float(price), 'qty': float(fill_qty)})

                    # ── PASSIVE BUY: trade crossed our bid ──
                    # Bot traded at price <= our limit → we buy at our limit
                    if remaining > 0:
                        for trade_price, trade_qty in tick_trades.get(product, []):
                            if trade_price <= price and lim - position[product] > 0:
                                fill_qty = min(remaining, trade_qty, lim - position[product])
                                if fill_qty > 0:
                                    position[product] += fill_qty
                                    cash -= fill_qty * price
                                    remaining -= fill_qty
                                    fill_log.append({'day': int(day), 'ts': int(timestamp),
                                        'product': product, 'side': 'BUY',
                                        'price': float(price), 'qty': float(fill_qty)})

                # ── AGGRESSIVE SELLS ──
                for price, qty in sell_orders:
                    remaining = abs(qty)
                    for bid_p in sorted(bid_levels, reverse=True):
                        if remaining <= 0 or price > bid_p: break
                        avail    = bid_levels[bid_p]
                        fill_qty = min(remaining, avail, lim + position[product])
                        if fill_qty > 0:
                            position[product] -= fill_qty
                            cash += fill_qty * price
                            bid_levels[bid_p] -= fill_qty
                            remaining -= fill_qty
                            fill_log.append({'day': int(day), 'ts': int(timestamp),
                                'product': product, 'side': 'SELL',
                                'price': float(price), 'qty': float(fill_qty)})

                    # ── PASSIVE SELL: trade crossed our ask ──
                    # Bot traded at price >= our limit → we sell at our limit
                    if remaining > 0:
                        for trade_price, trade_qty in tick_trades.get(product, []):
                            if trade_price >= price and lim + position[product] > 0:
                                fill_qty = min(remaining, trade_qty, lim + position[product])
                                if fill_qty > 0:
                                    position[product] -= fill_qty
                                    cash += fill_qty * price
                                    remaining -= fill_qty
                                    fill_log.append({'day': int(day), 'ts': int(timestamp),
                                        'product': product, 'side': 'SELL',
                                        'price': float(price), 'qty': float(fill_qty)})

            # ── MARK TO MARKET ─────────────────────────────────────────────────
            mtm = cash
            for product, od in order_depths.items():
                if od.buy_orders and od.sell_orders:
                    mid = (max(od.buy_orders) + min(od.sell_orders)) / 2
                else:
                    mid = 0
                mtm += position[product] * mid
            pnl_history.append(mtm)

        if verbose:
            print(f"\n--- Day {current_day} ---")
            for p in PRODUCTS:
                if position[p] != 0: print(f"  {p} final pos: {position[p]}")
            print(f"  Cash: {round(cash, 2)}")
            if pnl_history: print(f"  PnL (mark-to-market): {round(pnl_history[-1], 2)}")

        day_results.append({
            'day': current_day, 'cash': cash,
            'pnl': pnl_history[-1] if pnl_history else 0,
            'positions': dict(position),
        })

    # Final MTM is the true total P&L (not sum of daily MTMs which double-counts)
    total_pnl = day_results[-1]['pnl'] if day_results else 0.0

    # Per-day realized P&L (incremental)
    daily_realized = []
    prev = 0.0
    for d in day_results:
        daily_realized.append(d['pnl'] - prev)
        prev = d['pnl']

    if verbose:
        print("\n========== AGGREGATE RESULTS ==========")
        print(f"Total PnL (final MTM): {round(total_pnl, 2)}")
        print(f"Per-day realized:      {[round(x, 2) for x in daily_realized]}")
        for d in day_results:
            print(f"  Day {d['day']}: MTM {round(d['pnl'], 2)}, positions {d['positions']}")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fill_log.json')
        with open(out, 'w') as f: json.dump({"fills": fill_log}, f)
        print(f"\nExported {len(fill_log)} fills to {out}")

    return total_pnl, day_results, fill_log


if __name__ == "__main__":
    run_backtest(verbose=True)
