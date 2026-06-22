"""
Fit IV smile for VELVETFRUIT_EXTRACT_VOUCHER series across historical days.

Outputs:
  - Per-TTE IV levels per strike
  - Smile polynomial (quadratic in moneyness) fit each snapshot
  - Residuals (voucher richness/cheapness) time series
  - Summary diagnostics for strategy design

Run: python analysis/smile_analysis.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from options import implied_vol, moneyness_m, bs_call, bs_delta

STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]  # active strikes (4000/4500 = deep ITM, 6000/6500 = dead)
DEEP_ITM = [4000, 4500]
DEAD = [6000, 6500]


def load_day(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    return df


def pivot_mids(df: pd.DataFrame) -> pd.DataFrame:
    """Return timestamp-indexed df with one column per product."""
    return df.pivot(index='timestamp', columns='product', values='mid_price')


def compute_ivs_for_day(df: pd.DataFrame, tte_days: float) -> pd.DataFrame:
    """Return DF indexed by timestamp with IV per strike and S (underlying)."""
    mids = pivot_mids(df)
    S = mids['VELVETFRUIT_EXTRACT']
    rows = []
    for ts, s_val in S.items():
        if pd.isna(s_val):
            continue
        row = {'timestamp': ts, 'S': s_val}
        for k in STRIKES:
            col = f'VEV_{k}'
            if col not in mids.columns:
                continue
            price = mids.at[ts, col]
            if pd.isna(price):
                row[f'iv_{k}'] = np.nan
                continue
            iv = implied_vol(price, s_val, float(k), tte_days)
            row[f'iv_{k}'] = iv if iv is not None else np.nan
            row[f'm_{k}'] = moneyness_m(s_val, float(k), tte_days)
        rows.append(row)
    return pd.DataFrame(rows).set_index('timestamp')


def fit_smile(iv_df: pd.DataFrame) -> pd.DataFrame:
    """Fit quadratic IV = a + b*m + c*m^2 per snapshot.
    Returns per-timestamp coefficients + residuals."""
    coeffs = []
    for ts, row in iv_df.iterrows():
        ms = []
        vs = []
        for k in STRIKES:
            iv = row.get(f'iv_{k}')
            m = row.get(f'm_{k}')
            if pd.notna(iv) and pd.notna(m):
                ms.append(m); vs.append(iv)
        if len(ms) < 3:
            continue
        ms = np.array(ms); vs = np.array(vs)
        c = np.polyfit(ms, vs, 2)  # highest degree first: c2, c1, c0
        rec = {'timestamp': ts, 'c2': c[0], 'c1': c[1], 'c0': c[2]}
        # ATM IV = c0 (m=0)
        rec['atm_iv'] = c[2]
        # residuals
        fitted = np.polyval(c, ms)
        for k, m_k, iv_k, fit_k in zip(STRIKES, ms, vs, fitted):
            rec[f'resid_{k}'] = iv_k - fit_k
        coeffs.append(rec)
    return pd.DataFrame(coeffs).set_index('timestamp')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/round3')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    # day 0 -> TTE 8, day 1 -> TTE 7, day 2 -> TTE 6
    day_tte = {0: 8, 1: 7, 2: 6}

    all_iv = {}
    all_smile = {}
    for day, tte in day_tte.items():
        path = data_dir / f'prices_round_3_day_{day}.csv'
        if not path.exists():
            continue
        print(f'\n=== Day {day}  (TTE={tte}d) ===')
        df = load_day(path)
        iv_df = compute_ivs_for_day(df, float(tte))
        smile_df = fit_smile(iv_df)
        all_iv[day] = iv_df
        all_smile[day] = smile_df

        # Summary
        iv_cols = [c for c in iv_df.columns if c.startswith('iv_')]
        print('  Mean IV per strike:')
        for c in iv_cols:
            k = c.split('_')[1]
            mean_iv = iv_df[c].mean()
            std_iv = iv_df[c].std()
            print(f'    K={k}: IV={mean_iv:.4f}  std={std_iv:.4f}')
        print(f'  Smile: ATM IV (c0) mean={smile_df["c0"].mean():.4f}  std={smile_df["c0"].std():.4f}')
        print(f'         skew (c1)     mean={smile_df["c1"].mean():.4f}')
        print(f'         curv (c2)     mean={smile_df["c2"].mean():.4f}')

        # Residual stability (half-life via lag-1 autocorr)
        print('  Residual std by strike (trade signal richness):')
        for k in STRIKES:
            col = f'resid_{k}'
            if col in smile_df.columns:
                r = smile_df[col].dropna()
                ac1 = r.autocorr(lag=1) if len(r) > 10 else np.nan
                print(f'    K={k}: resid_std={r.std():.5f}  lag1_ac={ac1:.3f}')

    # Persistence check: does ATM IV change day-over-day?
    print('\n=== Cross-day ATM IV ===')
    for d, s in all_smile.items():
        print(f'  Day {d} (TTE={day_tte[d]}): ATM IV = {s["c0"].mean():.4f}  first={s["c0"].iloc[0]:.4f}  last={s["c0"].iloc[-1]:.4f}')

    # Save combined smile residuals for downstream use
    out_dir = Path('analysis/cache')
    out_dir.mkdir(exist_ok=True)
    for d, s in all_smile.items():
        s.to_csv(out_dir / f'smile_day_{d}.csv')
    print(f'\nSaved smile fits to {out_dir}/')


if __name__ == '__main__':
    main()
