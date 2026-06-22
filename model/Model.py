"""
IMC Prosperity 4 — Round 4 trading agent (live submission).

Architecture: market-make by default, swing aggressive on a confirmed
informed-counterparty signal.

Six product groups (10 vouchers + 2 delta-1 underlyings):

  HYDROGEL_PACK          passive MM inside-spread, slow-EMA mean-reversion skew,
                         kill the side opposing an active signal.
  VELVETFRUIT_EXTRACT    passive MM, delta-hedge sink for the voucher book,
                         capped directional sweep on confirmed signal.
  MM vouchers
    (5200, 5300, 5400, 5500)
                         BS-priced inside-spread MM with IV mean-reversion takes,
                         hard inventory caps per strike.
  Signal vouchers
    (5000, 5100)         No MM (spreads too wide). Edge-take only, blocked from
                         taking against the active signal direction.
  Deep-ITM (4000, 4500)  Acts as synthetic underlying. Tight edge thresholds,
                         small directional tilt on confirmed signal.
  Deep-OTM (6000, 6500)  Lottery: long only on LONG signal *confirmed* by both
                         the calibrated table and the generic bot detector.

Signal generation runs two layers:
  - Calibrated table: known informed counterparties (Mark X) hand-mapped to
    long/short from prior-round analysis. High precision.
  - Generic detector: any "Mark X" with one-sided imbalance ≥ 65% over a
    rolling window. Confirmation gate for the calibrated signal.

Both must fire before a full directional swing executes — limits the variance
of the directional book.

Live R4 result: -20,123 Xirecs. The directional long-vol position bled theta
on a stationary underlying while waiting for realised vol that didn't arrive.
See README for the full post-mortem.
"""
from datamodel import TradingState, Order
import json
import math

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ── PRODUCTS ───────────────────────────────────────────────────────────────────
UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL   = "HYDROGEL_PACK"
STRIKES        = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
MM_STRIKES     = [5200, 5300, 5400, 5500]
SIGNAL_STRIKES = [5000, 5100]
DEEP_ITM       = [4000, 4500]
DEEP_OTM       = [6000, 6500]
VOUCHER = lambda k: f"VEV_{k}"

POS_LIMITS = {
    HYDROGEL:   200,
    UNDERLYING: 200,
    **{VOUCHER(k): 300 for k in STRIKES},
}

# ── TIME ───────────────────────────────────────────────────────────────────────
INITIAL_TTE   = 3.0     # live R4 = 3
TICKS_PER_DAY = 1_000_000
FALLBACK_IV   = 0.0121

# ── HG ─────────────────────────────────────────────────────────────────────────
HG_EMA_ALPHA  = 0.02
HG_SLOW_ALPHA = 0.002
HG_DEV_TRIG   = 5
HG_MR_BOOST   = 2.0
HG_SOFT_CAP   = 100
HG_MAX_QTY    = 50
HG_EMA_BAND   = 2.0

# ── VE ─────────────────────────────────────────────────────────────────────────
VE_OFFSET     = 1
VE_MAX_QTY    = 40
VE_SWING_TGT  = 150     # capped vs 519's 200 → lower VaR
DELTA_BAND    = 40

# ── VOUCHERS ───────────────────────────────────────────────────────────────────
VOUCHER_SIZE     = 20
SIGNAL_SIZE      = 15
SIGNAL_EDGES     = {4000: 0.15, 4500: 0.15, 5000: 1.0, 5100: 1.0}
VOUCHER_INV_CAPS = {4000: 25, 4500: 25, 5000: 60, 5100: 60,
                    5200: 60, 5300: 60, 5400: 50, 5500: 40,
                    6000: 80, 6500: 80}
VOUCHER_INV_CAP  = 60
MIN_SPREAD       = 1
IVMR_EDGE        = 1.0
IVMR_SIZE        = 10

# ── SIGNAL SWING (when informed hot) ───────────────────────────────────────────
VOUCHER_SWING_TGT_MM     = 180   # 519 used 250 — pull back
VOUCHER_SWING_TGT_SIGNAL = 150
DEEP_OTM_SWING_TGT       = 150   # 519 used 300 — half it
DEEP_ITM_SWING_TGT       = 25    # tiny — already delta-1, hedge bandwidth

# ── INFORMED DETECTION ─────────────────────────────────────────────────────────
NEUTRAL, LONG, SHORT = 0, 1, -1
TRADER_SIGNAL = {
    UNDERLYING: {"Mark 67": +1, "Mark 49": -1, "Mark 14": -1, "Mark 22": -1},
    HYDROGEL:   {"Mark 14": +1, "Mark 38": -1},
    "VEV_4000": {"Mark 38": +1, "Mark 14": -1},
}
SIGNAL_HOT_TICKS     = 500
GENERIC_MIN_FIRES    = 4
GENERIC_IMBALANCE    = 0.65
GENERIC_HOT_TICKS    = 800


# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES
# ══════════════════════════════════════════════════════════════════════════════
def _N(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0: return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / v
    return S * _N(d1) - K * _N(d1 - v)

def bs_delta(S, K, T, sigma):
    if T <= 0 or sigma <= 0: return 1.0 if S > K else 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / v
    return _N(d1)

def implied_vol(price, S, K, T, tol=1e-6):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-9 or T <= 0 or price >= S: return None
    lo, hi = 1e-6, 5.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if bs_call(S, K, T, m) > price: hi = m
        else: lo = m
        if hi - lo < tol: return 0.5 * (lo + hi)
    return 0.5 * (lo + hi)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL ENGINES — calibrated table + generic Mark-X imbalance detector
# ══════════════════════════════════════════════════════════════════════════════
def calibrated_signal(symbol, state, td):
    """Use calibrated TRADER_SIGNAL table — high precision."""
    table = TRADER_SIGNAL.get(symbol, {})
    if not table: return NEUTRAL
    key = f"sig_{symbol}"
    last_long, last_short = td.get(key, [None, None])
    trades = (state.market_trades.get(symbol, []) or []) + \
             (state.own_trades.get(symbol, []) or [])
    for t in trades:
        b_w = table.get(getattr(t, "buyer", "") or "", 0)
        s_w = table.get(getattr(t, "seller", "") or "", 0)
        if b_w == +1: last_long  = t.timestamp if last_long  is None else max(last_long,  t.timestamp)
        elif b_w == -1: last_short = t.timestamp if last_short is None else max(last_short, t.timestamp)
        if s_w == +1: last_short = t.timestamp if last_short is None else max(last_short, t.timestamp)
        elif s_w == -1: last_long  = t.timestamp if last_long  is None else max(last_long,  t.timestamp)
    td[key] = [last_long, last_short]
    now = state.timestamp
    long_hot  = last_long  is not None and now - last_long  <= SIGNAL_HOT_TICKS
    short_hot = last_short is not None and now - last_short <= SIGNAL_HOT_TICKS
    if long_hot and short_hot:
        return LONG if last_long >= last_short else SHORT
    if long_hot:  return LONG
    if short_hot: return SHORT
    return NEUTRAL


def generic_bot_signal(state, td):
    """Backup: any 'Mark X' with imbalance ≥ threshold over enough fires.
    Returns (long_hot, short_hot)."""
    bots = td.get("bots", {})
    for t in state.market_trades.get(UNDERLYING, []) or []:
        ts = t.timestamp
        for name, side in ((getattr(t, "buyer", "") or "", "b"),
                           (getattr(t, "seller", "") or "", "s")):
            if not name.startswith("Mark "): continue
            rec = bots.setdefault(name, {"b": 0, "s": 0, "lb": -10**9, "ls": -10**9})
            rec[side] += 1
            rec["l" + side] = max(rec["l" + side], ts)
    td["bots"] = bots
    now = state.timestamp
    lh = sh = False
    for rec in bots.values():
        tot = rec["b"] + rec["s"]
        if tot < GENERIC_MIN_FIRES: continue
        imb = (rec["b"] - rec["s"]) / tot
        if imb >  GENERIC_IMBALANCE and now - rec["lb"] <= GENERIC_HOT_TICKS: lh = True
        if imb < -GENERIC_IMBALANCE and now - rec["ls"] <= GENERIC_HOT_TICKS: sh = True
    return lh, sh


def combined_signal(state, td):
    """LONG/SHORT only when calibrated table fires; STRONG when also confirmed
    by generic detector. Returns (direction, confirmed)."""
    cal = calibrated_signal(UNDERLYING, state, td)
    lh, sh = generic_bot_signal(state, td)
    if cal == LONG:
        return LONG, lh
    if cal == SHORT:
        return SHORT, sh
    # No calibrated signal — only act if BOTH generic sides agree (i.e. only one is true) AND it's strong.
    # Conservative: never swing on generic alone for VaR.
    return NEUTRAL, False


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY LAYERS — base class + one function per product group
# ══════════════════════════════════════════════════════════════════════════════

# ── Base ───────────────────────────────────────────────────────────────────────
class P:
    def __init__(self, sym, state, td):
        self.sym = sym; self.state = state; self.td = td; self.orders = []
        d = state.order_depths.get(sym)
        if d and d.buy_orders and d.sell_orders:
            self.bid = max(d.buy_orders); self.ask = min(d.sell_orders)
            self.bid_q = abs(d.buy_orders[self.bid]); self.ask_q = abs(d.sell_orders[self.ask])
            self.mid = (self.bid + self.ask) / 2.0; self.depth = d
        else:
            self.bid = self.ask = self.mid = None; self.bid_q = self.ask_q = 0; self.depth = d
        self.pos = state.position.get(sym, 0); self.lim = POS_LIMITS.get(sym, 0)
        self.max_buy  = self.lim - self.pos
        self.max_sell = self.lim + self.pos

    def buy(self, px, q):
        q = min(int(q), self.max_buy)
        if q > 0: self.orders.append(Order(self.sym, int(px), q)); self.max_buy -= q

    def sell(self, px, q):
        q = min(int(q), self.max_sell)
        if q > 0: self.orders.append(Order(self.sym, int(px), -q)); self.max_sell -= q

    def sweep_buy(self, target_pos):
        """Walk asks up to target_pos."""
        if not self.depth or not self.depth.sell_orders: return
        budget = max(0, target_pos - self.pos)
        for px in sorted(self.depth.sell_orders):
            if budget <= 0: break
            q = min(abs(self.depth.sell_orders[px]), budget, self.max_buy)
            if q > 0: self.buy(px, q); budget -= q

    def sweep_sell(self, target_pos):
        if not self.depth or not self.depth.buy_orders: return
        budget = max(0, self.pos - target_pos)
        for px in sorted(self.depth.buy_orders, reverse=True):
            if budget <= 0: break
            q = min(abs(self.depth.buy_orders[px]), budget, self.max_sell)
            if q > 0: self.sell(px, q); budget -= q


# ── HYDROGEL ───────────────────────────────────────────────────────────────────
def trade_hydrogel(state, td):
    p = P(HYDROGEL, state, td)
    if p.mid is None: return p.orders
    sig = calibrated_signal(HYDROGEL, state, td)

    ema = td.get("hg_ema", p.mid)
    ema = HG_EMA_ALPHA * p.mid + (1 - HG_EMA_ALPHA) * ema; td["hg_ema"] = ema
    slow = td.get("hg_slow", p.mid)
    slow = HG_SLOW_ALPHA * p.mid + (1 - HG_SLOW_ALPHA) * slow; td["hg_slow"] = slow

    my_bid = p.bid + 1; my_ask = p.ask - 1
    if my_ask <= my_bid: my_bid, my_ask = p.bid, p.ask

    taper = max(0.0, 1.0 - abs(p.pos) / HG_SOFT_CAP)
    buy_size  = HG_MAX_QTY if p.pos <= 0 else int(HG_MAX_QTY * taper)
    sell_size = HG_MAX_QTY if p.pos >= 0 else int(HG_MAX_QTY * taper)
    if p.mid > ema + HG_EMA_BAND:  buy_size  = min(buy_size, HG_MAX_QTY // 3)
    elif p.mid < ema - HG_EMA_BAND: sell_size = min(sell_size, HG_MAX_QTY // 3)

    if sig == LONG:  sell_size = 0
    elif sig == SHORT: buy_size = 0

    dev = p.mid - slow
    if dev > HG_DEV_TRIG:    buy_size, sell_size = 0, int(HG_MAX_QTY * HG_MR_BOOST)
    elif dev < -HG_DEV_TRIG: buy_size, sell_size = int(HG_MAX_QTY * HG_MR_BOOST), 0

    p.buy(my_bid, buy_size); p.sell(my_ask, sell_size)
    return p.orders


# ── VOUCHER MM ─────────────────────────────────────────────────────────────────
def trade_mm_voucher(state, td, k, S, T, direction, confirmed):
    p = P(VOUCHER(k), state, td)
    if p.mid is None or S is None: return p.orders, 0.0
    cap = VOUCHER_INV_CAPS.get(k, VOUCHER_INV_CAP)

    # IV bookkeeping
    iv_key = f"iv_{k}"
    live_iv = implied_vol(p.mid, S, float(k), T)
    if live_iv is not None:
        ema_iv = td.get(iv_key, live_iv)
        ema_iv = 0.05 * live_iv + 0.95 * ema_iv
        td[iv_key] = ema_iv
        iv, fair_iv = live_iv, ema_iv
    else:
        iv = td.get(iv_key, FALLBACK_IV); fair_iv = iv

    # Signal swing — only on confirmed direction (calibrated + bot agreement)
    if direction == LONG and confirmed:
        p.sweep_buy(VOUCHER_SWING_TGT_MM)
        return p.orders, bs_delta(S, float(k), T, iv) * (p.pos + sum(o.quantity for o in p.orders))
    if direction == SHORT and confirmed:
        p.sweep_sell(-VOUCHER_SWING_TGT_MM)
        return p.orders, bs_delta(S, float(k), T, iv) * (p.pos + sum(o.quantity for o in p.orders))

    # MM around BS theo
    if (p.ask - p.bid) > MIN_SPREAD:
        theo = bs_call(S, float(k), T, iv)
        fair = bs_call(S, float(k), T, fair_iv)
        r = int(round(theo))

        # IV-MR take
        if p.bid - fair > IVMR_EDGE and p.pos > -cap:
            q = min(IVMR_SIZE, cap + p.pos, p.bid_q, p.max_sell)
            if q > 0 and direction != LONG: p.sell(p.bid, q)
        elif fair - p.ask > IVMR_EDGE and p.pos < cap:
            q = min(IVMR_SIZE, cap - p.pos, p.ask_q, p.max_buy)
            if q > 0 and direction != SHORT: p.buy(p.ask, q)

        my_bid = max(r - 1, p.bid); my_ask = min(r + 1, p.ask)
        if my_ask > my_bid:
            taper = max(0.0, 1.0 - abs(p.pos) / cap)
            sz = int(VOUCHER_SIZE * taper)
            if direction != SHORT: p.buy(my_bid,  sz)
            if direction != LONG:  p.sell(my_ask, sz)

    return p.orders, bs_delta(S, float(k), T, iv) * p.pos


# ── SIGNAL VOUCHERS (5000, 5100) ───────────────────────────────────────────────
def trade_signal_voucher(state, td, k, S, T, direction, confirmed):
    p = P(VOUCHER(k), state, td)
    if p.mid is None or S is None: return p.orders, 0.0
    iv = td.get(f"iv_{k}", FALLBACK_IV)
    cap = VOUCHER_INV_CAPS.get(k, VOUCHER_INV_CAP)
    edge = SIGNAL_EDGES.get(k, 1.0)

    if direction == LONG and confirmed:
        p.sweep_buy(VOUCHER_SWING_TGT_SIGNAL); return p.orders, bs_delta(S, float(k), T, iv) * p.pos
    if direction == SHORT and confirmed:
        p.sweep_sell(-VOUCHER_SWING_TGT_SIGNAL); return p.orders, bs_delta(S, float(k), T, iv) * p.pos

    # Edge takes only — no MM (spreads too wide)
    theo = bs_call(S, float(k), T, iv)
    if p.pos < cap and theo - p.ask > edge and direction != SHORT:
        p.buy(p.ask, min(SIGNAL_SIZE, cap - p.pos, p.ask_q, p.max_buy))
    if p.bid - theo > edge and p.pos > -cap and direction != LONG:
        p.sell(p.bid, min(SIGNAL_SIZE, cap + p.pos, p.bid_q, p.max_sell))

    # Drift residual back to zero passively
    if direction == NEUTRAL and abs(p.pos) > cap // 2:
        if p.pos > 0:
            ask = p.ask - 1 if p.ask - 1 > p.bid else p.ask
            p.sell(ask, min(p.pos - cap // 2, 20))
        else:
            bid = p.bid + 1 if p.bid + 1 < p.ask else p.bid
            p.buy(bid, min(-p.pos - cap // 2, 20))

    return p.orders, bs_delta(S, float(k), T, iv) * p.pos


# ── DEEP ITM (4000, 4500) — synthetic underlying ───────────────────────────────
def trade_deep_itm(state, td, k, S, T, direction, confirmed):
    p = P(VOUCHER(k), state, td)
    if p.mid is None or S is None: return p.orders, 0.0
    iv = td.get(f"iv_{k}", FALLBACK_IV)
    own_dir = calibrated_signal(VOUCHER(k), state, td)
    eff_dir = own_dir if own_dir != NEUTRAL else direction
    cap = VOUCHER_INV_CAPS.get(k, 25)
    edge = SIGNAL_EDGES.get(k, 0.15)

    if eff_dir == LONG and confirmed:
        p.sweep_buy(DEEP_ITM_SWING_TGT); return p.orders, bs_delta(S, float(k), T, iv) * p.pos
    if eff_dir == SHORT and confirmed:
        p.sweep_sell(-DEEP_ITM_SWING_TGT); return p.orders, bs_delta(S, float(k), T, iv) * p.pos

    theo = bs_call(S, float(k), T, iv)
    edge_b = edge - 0.05 if eff_dir == LONG  else edge
    edge_s = edge - 0.05 if eff_dir == SHORT else edge
    if p.pos < cap and theo - p.ask > edge_b and eff_dir != SHORT:
        p.buy(p.ask, min(SIGNAL_SIZE, cap - p.pos, p.ask_q, p.max_buy))
    if p.bid - theo > edge_s and p.pos > -cap and eff_dir != LONG:
        p.sell(p.bid, min(SIGNAL_SIZE, cap + p.pos, p.bid_q, p.max_sell))

    return p.orders, bs_delta(S, float(k), T, iv) * p.pos


# ── DEEP OTM (6000, 6500) — long-only lottery on confirmed long ────────────────
def trade_deep_otm(state, td, k, direction, confirmed):
    p = P(VOUCHER(k), state, td)
    if p.mid is None: return p.orders
    if direction == LONG and confirmed:
        p.sweep_buy(DEEP_OTM_SWING_TGT)
        return p.orders
    # Cold: drift toward zero passively
    if p.pos > 0:
        ask = p.ask - 1 if p.ask is not None and p.ask > 1 else (p.ask or 1)
        p.sell(ask, min(p.pos, 20))
    return p.orders


# ── VE ─────────────────────────────────────────────────────────────────────────
def trade_ve(state, td, agg_delta, direction, confirmed):
    p = P(UNDERLYING, state, td)
    if p.bid is None: return p.orders

    # 1) MM inside spread — kill side opposing signal
    my_bid = p.bid + VE_OFFSET; my_ask = p.ask - VE_OFFSET
    if my_ask > my_bid:
        bq = 0 if direction == SHORT else min(VE_MAX_QTY, p.max_buy)
        sq = 0 if direction == LONG  else min(VE_MAX_QTY, p.max_sell)
        if bq > 0: p.buy(my_bid, bq)
        if sq > 0: p.sell(my_ask, sq)

    # 2) Directional swing — capped, only when signal CONFIRMED
    if direction == LONG and confirmed:
        target = VE_SWING_TGT
        if p.pos < target:
            for px in sorted(p.depth.sell_orders):
                if p.pos + sum(o.quantity for o in p.orders if o.quantity > 0) >= target: break
                q = min(abs(p.depth.sell_orders[px]), p.max_buy, target - p.pos)
                if q > 0: p.buy(px, q)
    elif direction == SHORT and confirmed:
        target = -VE_SWING_TGT
        if p.pos > target:
            for px in sorted(p.depth.buy_orders, reverse=True):
                if p.pos + sum(o.quantity for o in p.orders if o.quantity < 0) <= target: break
                q = min(abs(p.depth.buy_orders[px]), p.max_sell, p.pos - target)
                if q > 0: p.sell(px, q)

    # 3) Delta hedge — never fights an active confirmed signal
    target_hedge = int(-agg_delta)
    needed = target_hedge - p.pos
    if abs(needed) > DELTA_BAND:
        if needed > 0 and not (direction == SHORT and confirmed):
            p.buy(p.ask, needed)
        elif needed < 0 and not (direction == LONG and confirmed):
            p.sell(p.bid, -needed)

    return p.orders


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Trader.run() dispatches to the strategy layers each tick
# ══════════════════════════════════════════════════════════════════════════════
class Trader:
    def _tte(self, state, td):
        last_ts = td.get("_ts", -1); day = td.get("_day", 0)
        if last_ts >= 0 and state.timestamp < last_ts: day += 1
        td["_ts"] = state.timestamp; td["_day"] = day
        return max(INITIAL_TTE - day - state.timestamp / TICKS_PER_DAY, 1e-6)

    def run(self, state: TradingState):
        try: td = json.loads(state.traderData) if state.traderData else {}
        except Exception: td = {}

        T = self._tte(state, td)
        ve_d = state.order_depths.get(UNDERLYING)
        S = (max(ve_d.buy_orders) + min(ve_d.sell_orders)) / 2.0 \
            if ve_d and ve_d.buy_orders and ve_d.sell_orders else None

        direction, confirmed = combined_signal(state, td)

        result = {HYDROGEL: trade_hydrogel(state, td)}

        agg_delta = 0.0
        if S is not None:
            for k in MM_STRIKES:
                ords, dlt = trade_mm_voucher(state, td, k, S, T, direction, confirmed)
                result[VOUCHER(k)] = ords; agg_delta += dlt
            for k in SIGNAL_STRIKES:
                ords, dlt = trade_signal_voucher(state, td, k, S, T, direction, confirmed)
                result[VOUCHER(k)] = ords; agg_delta += dlt
            for k in DEEP_ITM:
                ords, dlt = trade_deep_itm(state, td, k, S, T, direction, confirmed)
                result[VOUCHER(k)] = ords; agg_delta += dlt
            for k in DEEP_OTM:
                result[VOUCHER(k)] = trade_deep_otm(state, td, k, direction, confirmed)

        result[UNDERLYING] = trade_ve(state, td, agg_delta, direction, confirmed)

        try: out = json.dumps(td)
        except Exception: out = ""
        return result, 0, out
