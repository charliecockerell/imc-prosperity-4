# Analysis Toolkit Philosophy

## Problem We Solve

The original analysis tools (fv_reconstruction.py, book_analysis.py, mc_simulator.py) were **hardcoded for Round 0** using known bot offsets. When Round 1 introduces new products, these tools become useless.

The new toolkit is **discovery-first**: it ingests raw order book data and automatically finds structure, fair value, and alpha signals. Zero prior knowledge required.

---

## Core Principles

### 1. Generic, Not Product-Specific

**Rule:** No hardcoded bot offsets, FV levels, or product behavior.

**Why:** Each round introduces unknown products. Tools must work on any product automatically.

**How it applies:** 
- Fair value is estimated via 4 independent methods that work on any book structure
- Signals are mechanical (OFI, pressure, regime) and apply universally
- Product characterization (static/RW/mean-reverting) happens algorithmically

### 2. Fair Value is Latent, Not Observable

**Rule:** FV is never directly observable. Estimate it from multiple sources and cross-validate.

**Why:** Bots quote at *offsets* from the true FV, not at FV itself. Multiple estimators converge when model is correct, diverge when something changes.

**How it applies:**
- `fv_weighted_mid`: volume-weighted best bid/ask (corrects for imbalance)
- `fv_microprice`: imbalance-adjusted mid (Stoikov standard)
- `fv_regression`: infer FV from all 6 book levels via OLS
- `fv_trade_anchor`: trade prices as unbiased FV signal
- `fv_disagreement`: std across methods flags regime changes

### 3. Alpha Comes from *Structure*, Not Magic

**Rule:** Signals must be interpretable: they describe observable market microstructure that predicts price movement.

**Why:** Overfitting to historical data is a trap. Signals grounded in mechanism are portable to new products and rounds.

**How it applies:**
- **OFI** (order flow imbalance): when one side adds depth, it predicts that direction
- **Book Pressure**: asymmetric depth signals directional demand
- **Spread Change**: widening spreads precede volatility shifts
- **Trade Flow**: the sign and pace of market trades predict FV moves
- **Regime** (variance ratio): mean-reversion vs trending determines how aggressively to trade

### 4. Validate Before Deploying

**Rule:** Every signal must show measurable correlation with future returns before it's used.

**Why:** The backtester is imperfect (only matches market trades, no passive fills). Signals that correlate with returns in real data are more likely to work in production.

**How it applies:**
- `run_analysis.py` reports signal correlations with FV returns
- Only add a signal to the trader if correlation is >|0.05| and statistically significant
- Re-validate after each round when new data arrives

---

## What Gets Dropped

**MC Simulator:** Removed entirely. Reasoning:
- It requires a known model, which we don't have for new products
- Simulation from a known model teaches nothing new (tautology)
- Real edge comes from discovering structure in data, not validating a pre-built model
- Time is better spent on alpha signals

**Hardcoded Bot Analysis:** The old `book_analysis.py` conditioned on TOMATOES bot offsets. Now redundant:
- Bot discovery is automated (clustering on book levels)
- Bot offsets are inferred, not assumed
- This frees us from needing to reverse-engineer each product's bots manually

---

## Workflow for New Rounds

When a new round opens with new products:

1. **Collect data** → place day-1 and day-2 price/trade CSVs in `data/prices_round_X_day_Y.csv` format

2. **Run analysis** → `python analysis/run_analysis.py --data-dir data/`

   Output: FV characterization + signal correlations for each product

3. **Read the report:**
   - Is FV static, RW, mean-reverting, or trending?
   - Which signals correlate with returns? (OFI, pressure, trade flow, etc.)
   
4. **Update the strategy** → `model/Model.py`
   - Use the best FV estimator (usually wmid or regression)
   - Wire in the predictive signals
   - Test in backtester
   
5. **Backtest + iterate** → `python visualisation/backtester.py`
   - Load fills in visualiser
   - Check: are signal peaks aligned with buy/sell fills?
   - Refine signal weights or quoting rules

---

## Interpreting the Report

Example output:

```
Fair Value Characterization:
  Type:           RANDOM_WALK
  Return Sigma:   0.496
  Variance Ratio: 1.05
  Hurst Exponent: 0.51

Alpha Signal Correlations with FV Return:
  OFI             : +0.18    ← Strong! Use this
  Pressure        : +0.08    ← Weak, include but low weight
  Trade Flow      : -0.02    ← Noise, skip
  Regime          : -0.01    ← Noise, skip
```

**Read it as:**
- **RW product**: use DynamicTrader pattern (quote 1 tick inside wall, let bot 2 set FV)
- **OFI is predictive**: skew quotes when OFI is positive/negative
- **Trade flow is not predictive**: don't bother routing trades

---

## Key Files

| File | Purpose | When to edit |
|------|---------|-------------|
| `loader.py` | Generic CSV loading | Only if data format changes |
| `stat_utils.py` | Statistical helpers | Only to add new tests |
| `fv_estimator.py` | Fair value methods | Only to add new FV estimator |
| `alpha_signals.py` | Signal generation | When tuning signal windows/normalizations |
| `run_analysis.py` | Reporting harness | Rarely, only for output format changes |

---

## Integration with Model.py

The analysis output feeds into `model/Model.py`:

```python
# In DynamicTrader.get_orders():
FV = self.fv_wmid   # or self.fv_regression, whichever correlates better
ofi = self.ofi_signal

# Use OFI to skew quotes
my_bid = FV - 1 + (ofi * 0.5)  # When OFI positive, bid more aggressively
my_ask = FV + 1 - (ofi * 0.5)  # When OFI positive, ask less aggressively
```

The analysis doesn't directly trade. It **guides strategy design**.

---

## Red Flags

Stop and re-analyze if you see:

1. **All signals uncorrelated with returns** → FV estimator is broken, or no predictable alpha exists
2. **Different signals contradict each other** → regime change or two bots with different behavior
3. **Signal magnitude spikes at day boundaries** → overnight risk or EOD behavior, worth investigating
4. **Fair value estimators severely disagree** → possibly a new bot appeared or market regime shifted

---

## Philosophy Summary

```
Raw Order Book Data
         ↓
   Auto-discover structure
     (no assumptions)
         ↓
   Estimate Fair Value (4 methods)
     & cross-validate
         ↓
   Generate mechanical signals
     (OFI, pressure, regime, ...)
         ↓
   Report correlations &
     product characterization
         ↓
   Manually update Model.py
     based on insights
         ↓
   Backtest & iterate
```

**Goal:** In one week per round, go from unknown products to a traded strategy, discovering the structure as you go.
