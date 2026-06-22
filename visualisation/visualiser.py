import pandas as pd
import numpy as np
import json
import sys
import os

# Add analysis to path for signal computation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
from fv_estimator import fv_weighted_mid, fv_microprice
from alpha_signals import signal_ofi, signal_book_pressure, signal_trade_flow, signal_volume_imbalance

# =====================
# CONFIG
# =====================

PRODUCTS   = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]   # all products you want tabs for
FILL_LOG   = "fill_log.json"

# Resolve paths relative to repo root (not visualisation directory)
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_repo_root = _os.path.dirname(_script_dir)
FOLDER_PATH = _os.path.join(_repo_root, "data/round2/")
if not _os.path.exists(FILL_LOG):
    FILL_LOG = _os.path.join(_repo_root, "visualisation/fill_log.json")

DAYS = [
    (_os.path.join(FOLDER_PATH, "prices_round_2_day_-1.csv"), _os.path.join(FOLDER_PATH, "trades_round_2_day_-1.csv"), -1),
    (_os.path.join(FOLDER_PATH, "prices_round_2_day_0.csv"),  _os.path.join(FOLDER_PATH, "trades_round_2_day_0.csv"),  0),
    (_os.path.join(FOLDER_PATH, "prices_round_2_day_1.csv"),  _os.path.join(FOLDER_PATH, "trades_round_2_day_1.csv"),  1),
]

# Position limits per product — used to draw limit lines and detect missed trades
POS_LIMITS = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}

# Each day spans 0-999900. Offset later days so the x-axis is continuous.
DAY_OFFSET = {day_id: i * 1_000_000 for i, (_, __, day_id) in enumerate(DAYS)}


# =====================
# LOAD FILLS
# =====================

with open(FILL_LOG) as f:
    data = json.load(f)

# Handle both old format (list of fills) and new format (dict with "fills" key)
if isinstance(data, list):
    all_fills = data
else:
    all_fills = data.get("fills", [])

# Give every fill a global timestamp using the same offset as prices
for f in all_fills:
    f['global_ts'] = f['ts'] + DAY_OFFSET[f['day']]


# =====================
# BUILD DATA PER PRODUCT
# =====================
# We loop over each product and build a self-contained dict of lists.
# That dict gets embedded as JSON in the HTML, one entry per product.

all_data = {}   # { product: { labels, bid1, ask1, ..., fills, stats } }

for PRODUCT in PRODUCTS:

    # --- Load and combine prices for this product ---
    frames = []
    for price_file, trade_file, day_id in DAYS:
        df = pd.read_csv(price_file, sep=';')
        df = df[df['product'] == PRODUCT].reset_index(drop=True)
        df['global_ts'] = df['timestamp'] + DAY_OFFSET[day_id]
        frames.append(df)

    prices = pd.concat(frames, ignore_index=True).sort_values('global_ts').reset_index(drop=True)

    # Downsample so the chart stays fast (every 5th row)
    sampled = prices.iloc[::5].reset_index(drop=True)

    # Helper: convert a column to a plain Python list, replacing NaN with None
    def col(name):
        return [None if pd.isna(v) else round(float(v), 1) for v in sampled[name]]

    labels = sampled['global_ts'].tolist()
    mid    = col('mid_price')

    # --- Compute FV estimates and alpha signals on full data, then downsample ---
    fv_wmid_full = fv_weighted_mid(prices)
    fv_micro_full = fv_microprice(prices)
    ofi_full = signal_ofi(prices, window=10)

    # Load trades for trade flow signal
    _trade_frames = []
    for _, trade_file, day_id in DAYS:
        tdf = pd.read_csv(trade_file, sep=';')
        tdf['day'] = day_id
        tdf['global_ts'] = tdf['timestamp'] + DAY_OFFSET[day_id]
        _trade_frames.append(tdf)
    _trades_for_product = pd.concat(_trade_frames, ignore_index=True)
    _trades_for_product = _trades_for_product[_trades_for_product['symbol'] == PRODUCT]

    tflow_full = signal_trade_flow(_trades_for_product, prices, window=20)
    pressure_full = signal_book_pressure(prices)
    vol_imb_full = signal_volume_imbalance(prices)

    # Downsample signals (every 5th row to match sampled prices)
    fv_wmid_s   = [None if pd.isna(v) else round(float(v), 2) for v in fv_wmid_full.iloc[::5]]
    fv_micro_s  = [None if pd.isna(v) else round(float(v), 2) for v in fv_micro_full.iloc[::5]]
    ofi_s       = [None if pd.isna(v) else round(float(v), 4) for v in ofi_full.iloc[::5]]
    tflow_s     = [None if pd.isna(v) else round(float(v), 4) for v in tflow_full.iloc[::5]]
    pressure_s  = [None if pd.isna(v) else round(float(v), 4) for v in pressure_full.iloc[::5]]
    vol_imb_s   = [None if pd.isna(v) else round(float(v), 4) for v in vol_imb_full.iloc[::5]]

    # --- Filter fills for this product ---
    our_fills = [f for f in all_fills if f['product'] == PRODUCT]
    buys  = [f for f in our_fills if f['side'] == 'BUY']
    sells = [f for f in our_fills if f['side'] == 'SELL']

    # Map each fill to the nearest sampled timestamp on the x-axis
    def nearest_label(gts):
        return min(labels, key=lambda t: abs(t - gts))

    buy_points  = [{'x': nearest_label(f['global_ts']), 'y': f['price'], 'price': f['price'], 'qty': f['qty']} for f in buys]
    sell_points = [{'x': nearest_label(f['global_ts']), 'y': f['price'], 'price': f['price'], 'qty': f['qty']} for f in sells]

    # --- Replay fills to get position and PnL at every sampled timestamp ---
    sorted_fills = sorted(our_fills, key=lambda f: f['global_ts'])
    pos_series, pnl_series = [], []
    running_pos, running_cash, fill_idx = 0, 0, 0

    for i, label_ts in enumerate(labels):
        while fill_idx < len(sorted_fills) and sorted_fills[fill_idx]['global_ts'] <= label_ts:
            f = sorted_fills[fill_idx]
            if f['side'] == 'BUY':
                running_cash -= f['qty'] * f['price']
                running_pos  += f['qty']
            else:
                running_cash += f['qty'] * f['price']
                running_pos  -= f['qty']
            fill_idx += 1
        current_mid = mid[i] if mid[i] else 0
        pos_series.append(round(running_pos, 1))
        pnl_series.append(round(running_cash + running_pos * current_mid, 1))

    # --- Compute summary stats (done in Python so HTML stays simple) ---
    # Average edge = how far from mid our fills were on average.
    # Use a DENSE mid lookup keyed by every tick (not the downsampled labels),
    # otherwise snapping fills to the nearest plotted label can be ±200 ticks
    # off — on a mean-reverting product that flips the sign of the edge.
    mid_map      = {labels[i]: mid[i] for i in range(len(labels))}           # sparse, for chart use
    dense_mid_map = dict(zip(prices['global_ts'], prices['mid_price']))       # exact per-tick mid

    def avg_edge(fills, side):
        edges = []
        for f in fills:
            m = dense_mid_map.get(f['global_ts'])
            if m is not None and pd.notna(m):
                # Positive edge means we bought below mid or sold above mid — good
                edge = (m - f['price']) if side == 'BUY' else (f['price'] - m)
                edges.append(edge)
        return round(sum(edges) / len(edges), 2) if edges else 0

    stats = {
        'total_fills':    len(our_fills),
        'n_buys':         len(buys),
        'n_sells':        len(sells),
        'final_pnl':      pnl_series[-1] if pnl_series else 0,
        'final_pos':      pos_series[-1] if pos_series else 0,
        'avg_buy_edge':   avg_edge(buys,  'BUY'),
        'avg_sell_edge':  avg_edge(sells, 'SELL'),
        'missed_count':   0,   # filled in after missed_trades is computed below
    }

    # --- Day boundaries: list of (day_id, offset) for N-day support ---
    # Used by the JS tooltip to label each sample's day correctly.
    day_list = [{"id": int(d_id), "offset": int(DAY_OFFSET[d_id])} for _, __, d_id in DAYS]

    # --- Load market trades for this product ---
    trade_frames = []
    for price_file, trade_file, day_id in DAYS:
        df = pd.read_csv(trade_file, sep=';')
        df['day'] = day_id
        trade_frames.append(df)
    market_trades = pd.concat(trade_frames, ignore_index=True)
    market_trades = market_trades[market_trades['symbol'] == PRODUCT].copy()
    market_trades['global_ts'] = market_trades['timestamp'] + market_trades['day'].map(DAY_OFFSET)

    # --- Find missed trades: market trade happened but we were at the position limit ---
    # We replay position fill-by-fill, then at each market trade check if we were maxed out.
    limit = POS_LIMITS.get(PRODUCT, 80)
    missed_trades = []  # list of {x, side, qty, price} to plot as markers on pos chart

    pos_replay = 0
    fill_replay_idx = 0

    for _, trade in market_trades.sort_values('global_ts').iterrows():
        # Apply fills up to and including this trade's timestamp
        while (fill_replay_idx < len(sorted_fills) and
               sorted_fills[fill_replay_idx]['global_ts'] <= trade['global_ts']):
            f = sorted_fills[fill_replay_idx]
            pos_replay += f['qty'] if f['side'] == 'BUY' else -f['qty']
            fill_replay_idx += 1

        # A trade below mid means a taker was selling — we'd want to BUY
        # A trade above mid means a taker was buying  — we'd want to SELL
        # Use the nearest mid price as reference
        approx_mid = mid_map.get(nearest_label(trade['global_ts']), None)
        if approx_mid is None:
            continue

        wanted_side = 'BUY' if trade['price'] < approx_mid else 'SELL'

        was_maxed = (wanted_side == 'BUY'  and pos_replay >= limit) or \
                    (wanted_side == 'SELL' and pos_replay <= -limit)

        if was_maxed:
            missed_trades.append({
                'x':     nearest_label(trade['global_ts']),
                'y':     pos_replay,   # plot at current position level
                'side':  wanted_side,
                'qty':   float(trade['quantity']),
                'price': float(trade['price']),
            })

    print(f"  Missed trades due to position limit: {len(missed_trades)}")

    # Now we know the count, patch it into stats
    stats['missed_count'] = len(missed_trades)

    all_data[PRODUCT] = {
        'labels':       labels,
        # Order book levels — L1 is tightest, L3 is widest
        'bid1': col('bid_price_1'), 'bvol1': col('bid_volume_1'),
        'bid2': col('bid_price_2'), 'bvol2': col('bid_volume_2'),
        'bid3': col('bid_price_3'), 'bvol3': col('bid_volume_3'),
        'ask1': col('ask_price_1'), 'avol1': col('ask_volume_1'),
        'ask2': col('ask_price_2'), 'avol2': col('ask_volume_2'),
        'ask3': col('ask_price_3'), 'avol3': col('ask_volume_3'),
        'mid':          mid,
        'buy_points':   buy_points,
        'sell_points':  sell_points,
        'pos_series':   pos_series,
        'pnl_series':   pnl_series,
        'day_list':     day_list,
        'stats':        stats,
        'pos_limit':    limit,
        'missed_trades': missed_trades,
        # FV estimates and alpha signals
        'fv_wmid':    fv_wmid_s,
        'fv_micro':   fv_micro_s,
        'ofi':        ofi_s,
        'tflow':      tflow_s,
        'pressure':   pressure_s,
        'vol_imb':    vol_imb_s,
    }

    print(f"{PRODUCT}: {len(labels)} price rows | {len(buys)} buys, {len(sells)} sells | final PnL: {stats['final_pnl']:,.0f}")


# =====================
# EMBED DATA AS JSON
# =====================
# One big JSON object keyed by product name.
# The HTML just does:  const ALL_DATA = <this>;  then ALL_DATA['EMERALDS'].labels etc.

data_json = json.dumps(all_data)

# Build tab buttons dynamically from PRODUCTS list
tab_buttons = '\n  '.join(
    f'<button class="tab{" active" if i == 0 else ""}" onclick="switchProduct(\'{p}\')">{p}</button>'
    for i, p in enumerate(PRODUCTS)
)


# =====================
# WRITE HTML
# =====================

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Visualiser</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  body {{
    background: #0d0f14;
    color: #aaa;
    font-family: monospace;
    font-size: 12px;
    padding: 16px;
  }}

  /* ---- header ---- */
  h2 {{ font-size: 14px; letter-spacing: 2px; color: #00d4aa; margin-bottom: 4px; }}
  .sub {{ font-size: 10px; color: #444; margin-bottom: 12px; }}

  /* ---- tabs: one button per product ---- */
  .tabs {{ display: flex; gap: 4px; margin-bottom: 10px; }}
  .tab {{
    padding: 5px 18px;
    border: 1px solid #222;
    background: #111;
    color: #555;
    cursor: pointer;
    font-family: monospace;
    font-size: 11px;
  }}
  .tab.active {{ border-color: #00d4aa; color: #00d4aa; }}

  /* ---- toggle buttons ---- */
  .controls {{ display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; align-items: center; }}
  .ctrl-label {{ color: #444; font-size: 10px; }}
  .toggle {{
    padding: 3px 10px;
    border: 1px solid #222;
    background: #111;
    color: #555;
    cursor: pointer;
    font-family: monospace;
    font-size: 10px;
  }}
  .toggle.on {{ border-color: #4f8ef7; color: #4f8ef7; }}

  /* ---- stats bar ---- */
  .stats {{
    display: flex;
    gap: 24px;
    padding: 8px 12px;
    background: #111;
    border: 1px solid #1c2230;
    margin-bottom: 10px;
    flex-wrap: wrap;
  }}
  .stat {{ display: flex; flex-direction: column; gap: 2px; }}
  .stat-label {{ font-size: 9px; color: #444; letter-spacing: 1px; }}
  .stat-value {{ font-size: 13px; color: #ccc; }}
  .pos {{ color: #00d4aa; }}
  .neg {{ color: #ff4d6a; }}

  /* ---- chart containers ---- */
  .chart-box {{
    background: #0d0f14;
    border: 1px solid #1a2030;
    padding: 6px;
    margin-bottom: 6px;
  }}
  .chart-label {{ font-size: 9px; color: #444; letter-spacing: 1px; margin-bottom: 4px; }}

  /* ---- legend row ---- */
  .legend {{ display: flex; gap: 14px; margin-bottom: 8px; font-size: 10px; color: #555; flex-wrap: wrap; }}
  .leg {{ display: flex; align-items: center; gap: 4px; }}
  .dot {{ width: 12px; height: 2px; }}
</style>
</head>
<body>

<h2>PROSPERITY VISUALISER</h2>
<div class="sub">order book · fills · signals · position · pnl &nbsp;|&nbsp; {len(DAYS)} days</div>

<!-- One tab per product -->
<div class="tabs" id="tabs">
  {tab_buttons}
</div>

<!-- Zoom range slider -->
<div class="controls" style="gap: 12px; margin-bottom: 8px; border-top: 1px solid #1a2030; padding-top: 10px;">
  <span class="ctrl-label">ZOOM:</span>
  <input type="range" id="zoomStart" min="0" max="100" value="0" style="flex: 1; max-width: 200px; cursor: pointer;"
         oninput="updateZoom()" title="Drag to zoom into time range">
  <input type="range" id="zoomEnd" min="0" max="100" value="100" style="flex: 1; max-width: 200px; cursor: pointer;"
         oninput="updateZoom()" title="Drag to zoom into time range">
  <button class="toggle" onclick="resetZoom()" style="padding: 4px 12px; font-size: 10px;">RESET</button>
  <span class="ctrl-label" style="color: #666; font-size: 10px; margin-left: 8px; border-left: 1px solid #1a2030; padding-left: 8px;">PAN:</span>
  <button class="toggle" onclick="panLeft()" style="padding: 4px 8px; font-size: 10px;">◀</button>
  <button class="toggle" onclick="panRight()" style="padding: 4px 8px; font-size: 10px;">▶</button>
  <span class="ctrl-label" id="zoomLabel" style="color: #666; font-size: 10px; min-width: 80px; margin-left: 12px;">100% view</span>
</div>

<!-- Toggle buttons to show/hide layers -->
<div class="controls">
  <span class="ctrl-label">SHOW:</span>
  <button class="toggle on" id="btn-l2"    onclick="toggle('l2')">DEPTH L2</button>
  <button class="toggle on" id="btn-l3"    onclick="toggle('l3')">DEPTH L3</button>
  <button class="toggle on" id="btn-mid"   onclick="toggle('mid')">MID</button>
  <button class="toggle on" id="btn-fills" onclick="toggle('fills')">FILLS</button>
  <button class="toggle" id="btn-fv" onclick="toggle('fv')">FV ESTIMATES</button>
  <span class="ctrl-label" style="margin-left:10px">VIEW:</span>
  <button class="toggle" id="btn-norm" onclick="toggleNorm()">NORMALIZE TO MID</button>
</div>

<!-- Stats bar: values filled by updateStats() in JS -->
<div class="stats">
  <div class="stat"><span class="stat-label">FILLS</span>    <span class="stat-value" id="s-fills">—</span></div>
  <div class="stat"><span class="stat-label">BUYS</span>     <span class="stat-value pos" id="s-buys">—</span></div>
  <div class="stat"><span class="stat-label">SELLS</span>    <span class="stat-value neg" id="s-sells">—</span></div>
  <div class="stat"><span class="stat-label">FINAL PNL</span><span class="stat-value" id="s-pnl">—</span></div>
  <div class="stat"><span class="stat-label">FINAL POS</span><span class="stat-value" id="s-pos">—</span></div>
  <div class="stat"><span class="stat-label">AVG BUY EDGE</span> <span class="stat-value" id="s-be">—</span></div>
  <div class="stat"><span class="stat-label">AVG SELL EDGE</span><span class="stat-value" id="s-se">—</span></div>
  <div class="stat"><span class="stat-label">MISSED (LIMIT)</span><span class="stat-value" id="s-missed">—</span></div>
</div>

<!-- Legend -->
<div class="legend">
  <div class="leg"><div class="dot" style="background:#ff4d6a"></div> Ask L1</div>
  <div class="leg"><div class="dot" style="background:rgba(255,77,106,0.35)"></div> Ask L2/L3</div>
  <div class="leg"><div class="dot" style="background:#00d4aa"></div> Bid L1</div>
  <div class="leg"><div class="dot" style="background:rgba(0,212,170,0.35)"></div> Bid L2/L3</div>
  <div class="leg"><div class="dot" style="background:rgba(79,142,247,0.4); border-top: 1px dashed rgba(79,142,247,0.4)"></div> Mid</div>
  <div class="leg"><span style="color:#00d4aa">▲</span> Buy fill</div>
  <div class="leg"><span style="color:#ff4d6a">▼</span> Sell fill</div>
  <div class="leg"><span style="color:#ff4d6a">●</span> Missed (pos limit)</div>
  <div class="leg"><div class="dot" style="background:#f0c040"></div> FV wmid</div>
  <div class="leg"><div class="dot" style="background:#c77dff"></div> FV microprice</div>
</div>

<!-- Main price + fills chart -->
<div class="chart-box" style="height:360px">
  <div class="chart-label">ORDER BOOK + FILLS</div>
  <canvas id="mainChart"></canvas>
</div>

<!-- Alpha signals -->
<div class="chart-box" style="height:140px">
  <div class="chart-label">ALPHA SIGNALS &nbsp;
    <span style="color:#ff6b6b; font-size:8px">■</span><span style="font-size:8px"> Vol Imbalance</span> &nbsp;
    <span style="color:#00d4aa; font-size:8px">■</span><span style="font-size:8px"> OFI</span> &nbsp;
    <span style="color:#f0c040; font-size:8px">■</span><span style="font-size:8px"> Trade Flow</span> &nbsp;
    <span style="color:#c77dff; font-size:8px">■</span><span style="font-size:8px"> Pressure</span>
  </div>
  <canvas id="signalChart"></canvas>
</div>

<!-- Position over time -->
<div class="chart-box" style="height:200px">
  <div class="chart-label">NET POSITION &nbsp;<span id="pos-pct" style="color:#555"></span></div>
  <canvas id="posChart"></canvas>
</div>

<!-- PnL over time -->
<div class="chart-box" style="height:120px">
  <div class="chart-label">MARK-TO-MARKET PNL</div>
  <canvas id="pnlChart"></canvas>
</div>


<script>
// All price + fill data, keyed by product name.
// Produced by Python — don't edit this by hand.
const ALL_DATA = {data_json};

// ---- Chart.js plugin: vertical day-divider lines + D-1 / D0 / D1 labels ----
const dayDividerPlugin = {{
  id: 'dayDivider',
  afterDraw(chart) {{
    const D = ALL_DATA[currentProduct];
    if (!D || !D.day_list) return;
    const vis = chart.data.labels;
    if (!vis || vis.length === 0) return;
    const first = vis[0], last = vis[vis.length - 1];
    const {{ ctx, chartArea, scales }} = chart;
    ctx.save();
    ctx.strokeStyle = 'rgba(79,142,247,0.35)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3,3]);
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(79,142,247,0.85)';
    for (const d of D.day_list) {{
      if (d.offset < first || d.offset > last) continue;
      // Nearest index in currently visible labels
      let idx = 0, best = Math.abs(vis[0] - d.offset);
      for (let i = 1; i < vis.length; i++) {{
        const v = Math.abs(vis[i] - d.offset);
        if (v < best) {{ best = v; idx = i; }}
      }}
      const x = scales.x.getPixelForValue(idx);
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.fillText('D' + d.id, x + 4, chartArea.top + 11);
    }}
    ctx.restore();
  }},
}};
Chart.register(dayDividerPlugin);

// ---- State ----
let currentProduct = '{PRODUCTS[0]}';
let normalized     = false;

// Which layers are currently visible
const visible = {{ l2: true, l3: true, mid: true, fills: true, fv: false }};

// Chart.js instances (we destroy + recreate on product switch)
let mainChart = null, posChart = null, pnlChart = null, signalChart = null;


// ---- Helpers ----

// Resolve (day_id, in-day ts) for a given global timestamp using day_list
function dayFor(ts) {{
  const D = ALL_DATA[currentProduct];
  // day_list is ordered by offset; find the last entry whose offset <= ts
  let entry = D.day_list[0];
  for (const e of D.day_list) {{
    if (ts >= e.offset) entry = e;
  }}
  return {{ id: entry.id, t: ts - entry.offset }};
}}

// Format a global timestamp as "D-1:42k" or "D0:7k"
function fmtTs(idx) {{
  const D   = ALL_DATA[currentProduct];
  const ts  = D.labels[idx];
  if (ts === undefined) return '';
  const {{ id, t }} = dayFor(ts);
  return 'D' + id + ':' + Math.round(t / 1000) + 'k';
}}

// Common x-axis config reused across all three charts
function xAxis() {{
  return {{
    ticks: {{
      color: '#444',
      maxTicksLimit: 14,
      font: {{ family: 'monospace', size: 9 }},
      callback: (_, idx) => fmtTs(idx),
    }},
    grid: {{ color: 'rgba(255,255,255,0.03)' }},
  }};
}}


// ---- Stats bar ----

function updateStats() {{
  const s = ALL_DATA[currentProduct].stats;

  document.getElementById('s-fills').textContent  = s.total_fills;
  document.getElementById('s-buys').textContent   = s.n_buys;
  document.getElementById('s-sells').textContent  = s.n_sells;
  document.getElementById('s-be').textContent     = s.avg_buy_edge;
  document.getElementById('s-se').textContent     = s.avg_sell_edge;

  const missedEl = document.getElementById('s-missed');
  missedEl.textContent = s.missed_count;
  missedEl.className   = 'stat-value ' + (s.missed_count > 0 ? 'neg' : 'pos');

  const pnlEl = document.getElementById('s-pnl');
  pnlEl.textContent  = (s.final_pnl >= 0 ? '+' : '') + s.final_pnl.toLocaleString();
  pnlEl.className    = 'stat-value ' + (s.final_pnl >= 0 ? 'pos' : 'neg');

  const posEl = document.getElementById('s-pos');
  posEl.textContent  = (s.final_pos > 0 ? '+' : '') + s.final_pos;
  posEl.className    = 'stat-value ' + (s.final_pos === 0 ? '' : s.final_pos > 0 ? 'pos' : 'neg');
}}


// ---- Main chart (order book + fills) ----

function buildMainChart() {{
  const D = ALL_DATA[currentProduct];

  // Apply zoom: slice all arrays to visible range
  const labels = zoomedSlice(D.labels);
  const bid1 = zoomedSlice(D.bid1);
  const bid2 = zoomedSlice(D.bid2);
  const bid3 = zoomedSlice(D.bid3);
  const ask1 = zoomedSlice(D.ask1);
  const ask2 = zoomedSlice(D.ask2);
  const ask3 = zoomedSlice(D.ask3);
  const mid  = zoomedSlice(D.mid);
  const fv_wmid = zoomedSlice(D.fv_wmid);
  const fv_micro = zoomedSlice(D.fv_micro);

  // If normalize is on, subtract mid from every price so the chart is flat at 0.
  // This makes it easy to see your edge (e.g. bid1 sitting at -7, ask1 at +7).
  const offsets = normalized ? mid.map(m => m || 0) : mid.map(() => 0);
  const shift   = (arr) => arr.map((v, i) => v === null ? null : v - offsets[i]);

  // Scatter fill points also need the same offset applied + zoom filtering
  const shiftPoint = (pts) => zoomedPoints(pts).map(p => {{
    // Find the offset at the nearest label to this fill's x position
    const idx = labels.indexOf(p.x);
    const off  = idx >= 0 ? offsets[idx] : 0;
    return {{ x: p.x, y: p.y - off, price: p.price, qty: p.qty }};
  }});

  const datasets = [
    // Ask levels — L3 (faintest) drawn first so L1 sits on top
    {{ type:'line', label:'Ask L3', data: shift(ask3), borderColor:'rgba(255,77,106,0.18)', borderWidth:1, pointRadius:0, tension:0, spanGaps:true, hidden:!visible.l3 }},
    {{ type:'line', label:'Ask L2', data: shift(ask2), borderColor:'rgba(255,77,106,0.38)', borderWidth:1, pointRadius:0, tension:0, spanGaps:true, hidden:!visible.l2 }},
    {{ type:'line', label:'Ask L1', data: shift(ask1), borderColor:'#ff4d6a',               borderWidth:1.5, pointRadius:0, tension:0, spanGaps:true }},
    // Bid levels
    {{ type:'line', label:'Bid L1', data: shift(bid1), borderColor:'#00d4aa',               borderWidth:1.5, pointRadius:0, tension:0, spanGaps:true }},
    {{ type:'line', label:'Bid L2', data: shift(bid2), borderColor:'rgba(0,212,170,0.38)', borderWidth:1, pointRadius:0, tension:0, spanGaps:true, hidden:!visible.l2 }},
    {{ type:'line', label:'Bid L3', data: shift(bid3), borderColor:'rgba(0,212,170,0.18)', borderWidth:1, pointRadius:0, tension:0, spanGaps:true, hidden:!visible.l3 }},
    // Mid price (dashed)
    {{ type:'line', label:'Mid', data: shift(mid), borderColor:'rgba(79,142,247,0.35)', borderWidth:1, borderDash:[3,3], pointRadius:0, tension:0, spanGaps:true, hidden:!visible.mid }},
    // Our fills
    {{ type:'scatter', label:'Buys',  data: shiftPoint(D.buy_points),  backgroundColor:'#00d4aa', pointStyle:'triangle',          pointRadius:7, hidden:!visible.fills }},
    {{ type:'scatter', label:'Sells', data: shiftPoint(D.sell_points), backgroundColor:'#ff4d6a', pointStyle:'triangle', rotation:180, pointRadius:7, hidden:!visible.fills }},
    // FV estimates (toggled off by default)
    {{ type:'line', label:'FV wmid',   data: shift(fv_wmid),  borderColor:'#f0c040', borderWidth:1.5, borderDash:[4,2], pointRadius:0, tension:0, spanGaps:true, hidden:!visible.fv }},
    {{ type:'line', label:'FV micro',  data: shift(fv_micro), borderColor:'#c77dff', borderWidth:1.5, borderDash:[4,2], pointRadius:0, tension:0, spanGaps:true, hidden:!visible.fv }},
  ];

  if (mainChart) mainChart.destroy();

  mainChart = new Chart(document.getElementById('mainChart'), {{
    data: {{ labels: labels, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#0a0f18',
          borderColor: '#1c2230',
          borderWidth: 1,
          titleColor: '#00d4aa',
          bodyColor: '#888',
          titleFont: {{ family:'monospace', size:10 }},
          bodyFont:  {{ family:'monospace', size:10 }},
          padding: 8,
          callbacks: {{
            title: ctx => {{
              // Show day + timestamp in tooltip header
              const ts  = D.labels[ctx[0].dataIndex];
              if (ts === undefined) return '';
              const {{ id, t }} = dayFor(ts);
              return 'Day ' + id + '  t=' + t;
            }},
            label: ctx => {{
              // For fill markers show side/qty/price
              if (ctx.dataset.label === 'Buys' || ctx.dataset.label === 'Sells') {{
                const p = ctx.raw;
                return ' ' + ctx.dataset.label.slice(0,-1).toUpperCase() + ' ' + p.qty + ' @ ' + p.price;
              }}
              // For lines show label + value
              const v = ctx.parsed.y;
              if (v === null) return null;
              return ' ' + ctx.dataset.label + ': ' + (normalized ? (v >= 0 ? '+' : '') + v.toFixed(1) : v);
            }},
            afterBody: ctx => {{
              // Show full order book snapshot at hovered timestamp
              const i = ctx[0].dataIndex;
              const lines = [];
              if (D.ask3[i] !== null) lines.push(' ASK3: ' + D.ask3[i] + '  x' + D.avol3[i]);
              if (D.ask2[i] !== null) lines.push(' ASK2: ' + D.ask2[i] + '  x' + D.avol2[i]);
              if (D.ask1[i] !== null) lines.push(' ASK1: ' + D.ask1[i] + '  x' + D.avol1[i]);
              lines.push(' ─────────────');
              if (D.bid1[i] !== null) lines.push(' BID1: ' + D.bid1[i] + '  x' + D.bvol1[i]);
              if (D.bid2[i] !== null) lines.push(' BID2: ' + D.bid2[i] + '  x' + D.bvol2[i]);
              if (D.bid3[i] !== null) lines.push(' BID3: ' + D.bid3[i] + '  x' + D.bvol3[i]);
              return lines;
            }},
          }},
        }},
      }},
      scales: {{
        x: xAxis(),
        y: {{ ticks: {{ color:'#444', font:{{ family:'monospace', size:9 }} }}, grid: {{ color:'rgba(255,255,255,0.03)' }} }},
      }},
    }},
  }});
}}


// ---- Position chart ----

function buildPosChart() {{
  const D    = ALL_DATA[currentProduct];
  const lim  = D.pos_limit;

  // Apply zoom
  const labels = zoomedSlice(D.labels);
  const pos_series = zoomedSlice(D.pos_series);
  const missed_trades = zoomedPoints(D.missed_trades);

  if (posChart) posChart.destroy();

  // Danger zone threshold — colour changes above 75% of limit
  const danger = lim * 0.75;

  // Colour each point on the position line based on how close to limit it is
  // green = safe, yellow = caution (>50%), red = danger (>75%)
  const pointColors = pos_series.map(p => {{
    const pct = Math.abs(p) / lim;
    if (pct >= 0.75) return 'rgba(255,77,106,0.8)';   // red
    if (pct >= 0.50) return 'rgba(240,192,64,0.7)';   // yellow
    return 'rgba(79,142,247,0.0)';                      // invisible (line colour handles it)
  }});

  posChart = new Chart(document.getElementById('posChart'), {{
    data: {{
      labels: labels,
      datasets: [
        // Solid limit lines
        {{
          type: 'line', label: '+limit',
          data: labels.map(() => lim),
          borderColor: 'rgba(255,77,106,0.5)', borderWidth: 1,
          borderDash: [4,3], pointRadius: 0, tension: 0,
        }},
        {{
          type: 'line', label: '-limit',
          data: labels.map(() => -lim),
          borderColor: 'rgba(255,77,106,0.5)', borderWidth: 1,
          borderDash: [4,3], pointRadius: 0, tension: 0,
        }},
        // 75% danger threshold lines
        {{
          type: 'line', label: '+danger',
          data: labels.map(() => danger),
          borderColor: 'rgba(240,192,64,0.25)', borderWidth: 1,
          borderDash: [2,4], pointRadius: 0, tension: 0,
        }},
        {{
          type: 'line', label: '-danger',
          data: labels.map(() => -danger),
          borderColor: 'rgba(240,192,64,0.25)', borderWidth: 1,
          borderDash: [2,4], pointRadius: 0, tension: 0,
        }},
        // Position line — colour shifts red when near limit
        {{
          type: 'line', label: 'Position',
          data: pos_series,
          borderColor: '#4f8ef7', borderWidth: 2,
          pointRadius: 0, tension: 0,
          fill: false,
          segment: {{
            borderColor: ctx => {{
              const v = ctx.p1.parsed.y;
              const pct = Math.abs(v) / lim;
              if (pct >= 0.75) return 'rgba(255,77,106,0.9)';
              if (pct >= 0.50) return 'rgba(240,192,64,0.8)';
              return '#4f8ef7';
            }},
          }},
        }},
        // Fill under the line, also colour-coded
        {{
          type: 'line', label: 'PosFill',
          data: pos_series,
          borderWidth: 0, pointRadius: 0, tension: 0,
          fill: 'origin',
          backgroundColor: ctx => {{
            const v = ctx.parsed ? ctx.parsed.y : 0;
            const pct = Math.abs(v) / lim;
            if (pct >= 0.75) return 'rgba(255,77,106,0.08)';
            if (pct >= 0.50) return 'rgba(240,192,64,0.06)';
            return 'rgba(79,142,247,0.05)';
          }},
        }},
        // Red dot for each missed trade (limit hit)
        {{
          type: 'scatter', label: 'Missed',
          data: missed_trades,
          backgroundColor: '#ff4d6a', pointStyle: 'circle', pointRadius: 7,
          borderColor: '#ff0000', borderWidth: 1,
        }},
      ],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#0a0f18',
          borderColor: '#1c2230', borderWidth: 1,
          titleColor: '#4f8ef7', bodyColor: '#888',
          titleFont: {{ family:'monospace', size:10 }},
          bodyFont:  {{ family:'monospace', size:10 }},
          padding: 8,
          callbacks: {{
            title: ctx => fmtTs(ctx[0].dataIndex),
            label: ctx => {{
              if (ctx.dataset.label === 'Missed') {{
                const p = ctx.raw;
                return [
                  ` ⚠ LIMIT HIT — missed ${{p.side}} ${{p.qty}} @ ${{p.price}}`,
                  ` position was: ${{p.y}} / ${{lim}} (${{Math.round(Math.abs(p.y)/lim*100)}}%)`,
                ];
              }}
              if (ctx.dataset.label === 'Position') {{
                const v = ctx.parsed.y;
                const pct = Math.round(Math.abs(v) / lim * 100);
                // Update the label above the chart too
                const el = document.getElementById('pos-pct');
                if (el) el.textContent = `${{v > 0 ? '+' : ''}}${{v}} / ${{lim}} (${{pct}}%)`;
                return ` pos: ${{v > 0 ? '+' : ''}}${{v}}  (${{pct}}% of limit)`;
              }}
              return null;
            }},
          }},
        }},
      }},
      scales: {{
        x: xAxis(),
        y: {{
          min: -(lim + 5),
          max:  (lim + 5),
          ticks: {{
            color: '#444',
            font: {{ family:'monospace', size:9 }},
            // Only label 0, ±50%, ±75%, ±100%
            callback: v => {{
              const marks = [0, lim*0.5, -lim*0.5, lim*0.75, -lim*0.75, lim, -lim];
              return marks.some(m => Math.abs(v - m) < 0.1) ? v : '';
            }},
          }},
          grid: {{
            color: ctx => {{
              const v = ctx.tick.value;
              if (Math.abs(Math.abs(v) - lim) < 0.1)       return 'rgba(255,77,106,0.2)';
              if (Math.abs(Math.abs(v) - lim*0.75) < 0.1)  return 'rgba(240,192,64,0.15)';
              return 'rgba(255,255,255,0.03)';
            }},
          }},
        }},
      }},
    }},
  }});
}}


// ---- Alpha signals chart ----

function buildSignalChart() {{
  const D = ALL_DATA[currentProduct];

  // Apply zoom
  const labels = zoomedSlice(D.labels);
  const vol_imb = zoomedSlice(D.vol_imb);
  const ofi = zoomedSlice(D.ofi);
  const tflow = zoomedSlice(D.tflow);
  const pressure = zoomedSlice(D.pressure);

  if (signalChart) signalChart.destroy();
  signalChart = new Chart(document.getElementById('signalChart'), {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [
        {{
          label: 'Vol Imbalance', data: vol_imb,
          borderColor: '#ff6b6b', borderWidth: 1.5, pointRadius: 0, tension: 0,
          spanGaps: true, fill: false,
        }},
        {{
          label: 'OFI', data: ofi,
          borderColor: '#00d4aa', borderWidth: 1.2, pointRadius: 0, tension: 0,
          spanGaps: true, fill: false,
        }},
        {{
          label: 'Trade Flow', data: tflow,
          borderColor: '#f0c040', borderWidth: 1, pointRadius: 0, tension: 0,
          spanGaps: true, fill: false,
        }},
        {{
          label: 'Pressure', data: pressure,
          borderColor: '#c77dff', borderWidth: 1, pointRadius: 0, tension: 0,
          spanGaps: true, fill: false,
        }},
        // Zero line
        {{
          label: 'Zero', data: labels.map(() => 0),
          borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
          borderDash: [3,3], pointRadius: 0, tension: 0,
        }},
      ],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#0a0f18',
          borderColor: '#1c2230', borderWidth: 1,
          titleColor: '#4f8ef7', bodyColor: '#888',
          titleFont: {{ family:'monospace', size:10 }},
          bodyFont:  {{ family:'monospace', size:10 }},
          padding: 8,
          callbacks: {{
            title: ctx => fmtTs(ctx[0].dataIndex),
            label: ctx => {{
              if (ctx.dataset.label === 'Zero') return null;
              const v = ctx.parsed.y;
              if (v === null) return null;
              const dir = v > 0.05 ? '↑ BUY' : v < -0.05 ? '↓ SELL' : '→ FLAT';
              return ` ${{ctx.dataset.label}}: ${{v >= 0 ? '+' : ''}}${{v.toFixed(4)}} ${{dir}}`;
            }},
          }},
        }},
      }},
      scales: {{
        x: xAxis(),
        y: {{
          ticks: {{ color:'#444', font:{{ family:'monospace', size:9 }} }},
          grid: {{ color:'rgba(255,255,255,0.03)' }},
        }},
      }},
    }},
  }});
}}


// ---- PnL chart ----

function buildPnlChart() {{
  const D = ALL_DATA[currentProduct];

  // Apply zoom
  const labels = zoomedSlice(D.labels);
  const pnl_series = zoomedSlice(D.pnl_series);

  if (pnlChart) pnlChart.destroy();
  pnlChart = new Chart(document.getElementById('pnlChart'), {{
    type: 'line',
    data: {{ labels: labels, datasets: [{{
      data: pnl_series,
      borderColor: '#00d4aa', borderWidth: 1.5,
      pointRadius: 0, tension: 0, fill: true,
      backgroundColor: 'rgba(0,212,170,0.05)',
    }}] }},
    options: {{
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }},
      scales: {{
        x: xAxis(),
        y: {{ ticks: {{ color:'#444', font:{{ family:'monospace', size:9 }} }}, grid: {{ color:'rgba(255,255,255,0.03)' }} }},
      }},
    }},
  }});
}}


// ---- Toggle handlers ----

function toggle(key) {{
  visible[key] = !visible[key];
  document.getElementById('btn-' + key).classList.toggle('on', visible[key]);
  // Update hidden flag on the relevant datasets without rebuilding the whole chart
  const map = {{ l2:['Ask L2','Bid L2'], l3:['Ask L3','Bid L3'], mid:['Mid'], fills:['Buys','Sells'], fv:['FV wmid','FV micro'] }};
  for (const label of map[key]) {{
    const ds = mainChart.data.datasets.find(d => d.label === label);
    if (ds) ds.hidden = !visible[key];
  }}
  mainChart.update();
}}

function toggleNorm() {{
  normalized = !normalized;
  document.getElementById('btn-norm').classList.toggle('on', normalized);
  buildMainChart();   // need full rebuild because all y-values change
}}


// ---- Switch product (tab click) ----

// ---- Zoom functionality ----

let zoomMinIdx = 0, zoomMaxIdx = null;

function updateZoom() {{
  const D = ALL_DATA[currentProduct];
  const n = D.labels.length;

  const startSlider = document.getElementById('zoomStart');
  const endSlider = document.getElementById('zoomEnd');

  zoomMinIdx = Math.floor((parseFloat(startSlider.value) / 100) * n);
  zoomMaxIdx = Math.ceil((parseFloat(endSlider.value) / 100) * n);

  // Ensure valid range
  zoomMaxIdx = Math.max(zoomMinIdx + 1, zoomMaxIdx);

  // Update label
  const range = zoomMaxIdx - zoomMinIdx;
  const pct = Math.round((range / n) * 100);
  document.getElementById('zoomLabel').textContent = pct + '% view';

  // Rebuild all charts with zoomed range
  buildMainChart();
  buildSignalChart();
  buildPosChart();
  buildPnlChart();
}}

function resetZoom() {{
  document.getElementById('zoomStart').value = 0;
  document.getElementById('zoomEnd').value = 100;
  zoomMinIdx = 0;
  zoomMaxIdx = null;
  document.getElementById('zoomLabel').textContent = '100% view';
  buildMainChart();
  buildSignalChart();
  buildPosChart();
  buildPnlChart();
}}

function panLeft() {{
  const D = ALL_DATA[currentProduct];
  const n = D.labels.length;
  const start = parseFloat(document.getElementById('zoomStart').value);
  const end = parseFloat(document.getElementById('zoomEnd').value);
  const width = end - start;
  const step = width * 0.2;  // pan by 20% of current zoom width
  const newStart = Math.max(0, start - step);
  const newEnd = Math.min(100, newStart + width);
  document.getElementById('zoomStart').value = newStart;
  document.getElementById('zoomEnd').value = newEnd;
  updateZoom();
}}

function panRight() {{
  const D = ALL_DATA[currentProduct];
  const n = D.labels.length;
  const start = parseFloat(document.getElementById('zoomStart').value);
  const end = parseFloat(document.getElementById('zoomEnd').value);
  const width = end - start;
  const step = width * 0.2;  // pan by 20% of current zoom width
  const newEnd = Math.min(100, end + step);
  const newStart = Math.max(0, newEnd - width);
  document.getElementById('zoomStart').value = newStart;
  document.getElementById('zoomEnd').value = newEnd;
  updateZoom();
}}

// Helper: slice array from zoomMinIdx to zoomMaxIdx
function zoomedSlice(arr) {{
  const D = ALL_DATA[currentProduct];
  const n = D.labels.length;
  const maxIdx = zoomMaxIdx === null ? n : zoomMaxIdx;
  return arr.slice(zoomMinIdx, maxIdx);
}}

// Helper: filter points by zoomed range
function zoomedPoints(points) {{
  const D = ALL_DATA[currentProduct];
  const labels = D.labels;
  const minLabel = labels[zoomMinIdx];
  const maxLabel = labels[(zoomMaxIdx === null ? labels.length : zoomMaxIdx) - 1];
  return points.filter(p => p.x >= minLabel && p.x <= maxLabel);
}}


function switchProduct(product) {{
  currentProduct = product;
  resetZoom();  // Reset zoom when switching products
  // Update tab highlight
  document.querySelectorAll('.tab').forEach(t => {{
    t.classList.toggle('active', t.textContent === product);
  }});
  updateStats();
  buildMainChart();
  buildSignalChart();
  buildPosChart();
  buildPnlChart();
}}


// ---- Initial render ----
updateStats();
buildMainChart();
buildSignalChart();
buildPosChart();
buildPnlChart();

</script>
</body>
</html>"""

# Write to visualisation/ directory (absolute path to ensure it works from any cwd)
output_path = _os.path.join(_script_dir, 'visualiser.html')
with open(output_path, 'w') as f:
    f.write(html)

print(f"Written {output_path} — open it in your browser")
