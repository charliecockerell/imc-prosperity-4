"""
IV / moneyness diagnostic report.

Loads ROUND_3 historical (days 0/1/2 = TTE 8/7/6), computes per-snapshot
implied vols across the live strikes, fits a quadratic smile in
m = ln(K/S)/sqrt(T), and looks for:

  1. Persistent per-strike residuals (structural mispricing).
  2. Residual autocorrelation / half-life (mean-reversion edge).
  3. Realised vol vs implied vol gap (over- or under-priced level).
  4. Liquidity by strike (which strikes are tradeable).

Output: analysis/iv_report.pdf

"""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from options import implied_vol, moneyness_m

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "round3"
OUT  = ROOT / "reports" / "iv_report.pdf"

STRIKES_ALL    = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
STRIKES_LIVE   = [5000, 5100, 5200, 5300, 5400, 5500]   # have non-trivial time value
INITIAL_TTE    = 8.0  # historical day 0 TTE


# ── data ──────────────────────────────────────────────────────────────────

def load_day(day: int) -> pd.DataFrame:
    """
    Loads in velvet fruit extract and the vouchers and turns into voucher table,
    cleaning up the data to be used throughout the analysis.
    
    """
    p = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    ve = (p[p["product"] == "VELVETFRUIT_EXTRACT"]
          [["timestamp", "mid_price", "bid_price_1", "ask_price_1"]]
          .rename(columns={"mid_price": "S"}))  # S = mid
    vou = p[p["product"].str.startswith("VEV_")].copy()
    vou["K"] = vou["product"].str.replace("VEV_", "").astype(int)   # K = strikes
    vou = vou.merge(ve[["timestamp", "S"]], on="timestamp", how="left")
    vou["T"] = INITIAL_TTE - day - vou["timestamp"] / 1_000_000 # T = time to expiry
    vou["day"] = day
    return vou


def compute_ivs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds implied vol and moneyness to the table
    
    """
    rows = []
    for r in df.itertuples():
        iv = implied_vol(r.mid_price, r.S, float(r.K), r.T)
        if iv is None or iv <= 0 or iv > 1.0:
            continue
        m = moneyness_m(r.S, float(r.K), r.T)
        rows.append({"day": r.day, "ts": r.timestamp, "K": r.K, "S": r.S,
                     "T": r.T, "mid": r.mid_price, "iv": iv, "m": m,
                     "spread": r.ask_price_1 - r.bid_price_1})
    return pd.DataFrame(rows)


# ── stats ─────────────────────────────────────────────────────────────────

def half_life(x: np.ndarray) -> float:
    """
    
    Half-life from AR(1): x_{t+1} = c + φ·x_t + ε.
    The constant is absorbed by the mean centering of cov and var
    HL = -ln(2)/ln(|phi|).
    
    """
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 50: return np.nan
    a, b = x[:-1], x[1:]
    phi = np.cov(a, b, ddof=0)[0, 1] / np.var(a)
    if not (0 < phi < 1): return np.nan
    return -math.log(2) / math.log(phi)


def realised_vol_per_day(raw: pd.DataFrame, day: int) -> float:
    """

    Actual underlying volatility: std of log returns of S, scaled to a per-day
    stdev so it's comparable to IV.

    """
    s = (raw[raw["day"] == day]
         .drop_duplicates("timestamp")
         .sort_values("timestamp")["S"]
         .reset_index(drop=True))
    r = np.diff(np.log(s.values))
    # ~10000 timestamps in a day → per-day stdev = std(r) * sqrt(n)
    return float(np.std(r) * math.sqrt(len(r)))


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("loading days 0/1/2 …")
    raw = pd.concat([load_day(d) for d in [0, 1, 2]], ignore_index=True)
    iv = compute_ivs(raw)
    iv = iv[iv["K"].isin(STRIKES_LIVE)].copy()
    print(f"  {len(iv):,} IV observations across {iv['K'].nunique()} strikes × 3 days")

    # quadratic smile fit (pooled all days, all snapshots)
    coeffs_pool = np.polyfit(iv["m"], iv["iv"], 2)
    iv["resid_pool"] = iv["iv"] - np.polyval(coeffs_pool, iv["m"])

    # per-snapshot smile fit (one quad per timestamp) — stricter null
    iv = iv.sort_values(["day", "ts", "K"]).reset_index(drop=True)
    iv["resid_snap"] = np.nan
    for (d, ts), g in iv.groupby(["day", "ts"]):
        if len(g) < 4:  # need ≥4 points for a quadratic
            continue
        c = np.polyfit(g["m"], g["iv"], 2)
        iv.loc[g.index, "resid_snap"] = g["iv"].values - np.polyval(c, g["m"].values)

    # realised vol per day
    rv = {d: realised_vol_per_day(raw, d) for d in [0, 1, 2]}
    atm_iv = iv[iv["K"].isin([5200, 5300])].groupby("day")["iv"].mean().to_dict()

    # ── PDF ────────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:

        # -- page 1: smile shape
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        for d in [0, 1, 2]:
            sub = iv[iv["day"] == d]
            ax[0].scatter(sub["m"], sub["iv"], s=2, alpha=0.15, label=f"day {d} (TTE {8-d})")
        ms = np.linspace(iv["m"].min(), iv["m"].max(), 200)
        ax[0].plot(ms, np.polyval(coeffs_pool, ms), "k-", lw=2, label="pooled fit")
        ax[0].set_xlabel("m = ln(K/S) / √T")
        ax[0].set_ylabel("IV (per √day)")
        ax[0].set_title("Volatility smile (all snapshots)")
        ax[0].legend(loc="best", fontsize=8)
        ax[0].grid(alpha=0.3)

        per_strike = iv.groupby("K")["iv"].agg(["mean", "std", "count"]).round(6)
        per_strike["resid_mean"] = iv.groupby("K")["resid_pool"].mean().round(6)
        per_strike["resid_snap_mean"] = iv.groupby("K")["resid_snap"].mean().round(6)
        ax[1].axis("off")
        ax[1].text(0, 1, "Per-strike IV summary", fontsize=11, fontweight="bold", va="top")
        ax[1].text(0, 0.95, per_strike.to_string(), family="monospace", fontsize=8.5,
                   va="top")
        a, b, c = coeffs_pool
        ax[1].text(0, 0.05, f"Quad fit: IV(m) = {a:.5f}·m² + {b:.5f}·m + {c:.5f}",
                   family="monospace", fontsize=9)
        fig.suptitle("Round 3 — IV smile across live strikes (days 0/1/2)")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # -- page 2: smile per day (overlaid) to show stability
        fig, ax = plt.subplots(figsize=(9, 5))
        per_day_strike = iv.groupby(["day", "K"])["iv"].mean().unstack(0)
        for d in [0, 1, 2]:
            ax.plot(per_day_strike.index, per_day_strike[d], "o-", label=f"day {d} (TTE {8-d})")
        ax.set_xlabel("Strike K"); ax.set_ylabel("Mean IV (per √day)")
        ax.set_title("Smile is essentially flat & stable across days")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # -- page 3: residual time series per strike
        fig, axes = plt.subplots(len(STRIKES_LIVE), 1, figsize=(10, 1.6 * len(STRIKES_LIVE)),
                                 sharex=True)
        for ax_i, k in zip(axes, STRIKES_LIVE):
            sub = iv[iv["K"] == k].sort_values(["day", "ts"]).reset_index(drop=True)
            t = sub["day"] * 1_000_000 + sub["ts"]
            ax_i.plot(t, sub["resid_snap"], lw=0.5)
            ax_i.axhline(0, color="k", lw=0.4)
            mu = sub["resid_snap"].mean(); sd = sub["resid_snap"].std()
            hl = half_life(sub["resid_snap"].dropna().values)
            ax_i.set_ylabel(f"K={k}\nμ={mu:+.5f}\nσ={sd:.5f}\nHL={hl:.0f}t" if not np.isnan(hl)
                            else f"K={k}\nμ={mu:+.5f}\nσ={sd:.5f}", fontsize=8)
            ax_i.grid(alpha=0.2)
        axes[-1].set_xlabel("ticks (concat days 0→2)")
        fig.suptitle("Per-snapshot smile residuals (deviation from cross-strike fit per tick)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # -- page 4: liquidity & realised vs implied
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        # liquidity proxy: count of valid IV obs + median spread (already in Xirecs)
        liq = raw[raw["K"].isin(STRIKES_ALL)].groupby("K").agg(
            n_quotes=("mid_price", "size"),
            spread=("ask_price_1", lambda s: (s - raw.loc[s.index, "bid_price_1"]).median()),
        )
        ax[0].bar(liq.index.astype(str), liq["n_quotes"], color="steelblue")
        ax[0].set_xlabel("Strike"); ax[0].set_ylabel("# snapshots with valid mid")
        ax[0].set_title("Liquidity proxy (snapshot count)")
        for k, v in zip(liq.index.astype(str), liq["spread"]):
            ax[0].text(str(k), 0, f"sp={v:.0f}", ha="center", va="bottom", fontsize=7)
        ax[0].grid(alpha=0.3, axis="y")

        # realised vs implied
        days = [0, 1, 2]
        rv_arr = [rv[d] for d in days]
        iv_arr = [atm_iv[d] for d in days]
        x = np.arange(3); w = 0.35
        ax[1].bar(x - w/2, rv_arr, w, label="Realised σ (per day)", color="firebrick")
        ax[1].bar(x + w/2, iv_arr, w, label="ATM IV (mean K=5200/5300)", color="steelblue")
        ax[1].set_xticks(x); ax[1].set_xticklabels([f"day {d}" for d in days])
        ax[1].set_ylabel("σ (per √day)")
        ax[1].set_title("Realised vs Implied")
        ax[1].grid(alpha=0.3, axis="y"); ax[1].legend(fontsize=8)
        for i, (rvv, ivv) in enumerate(zip(rv_arr, iv_arr)):
            ax[1].text(i - w/2, rvv, f"{rvv:.4f}", ha="center", va="bottom", fontsize=7)
            ax[1].text(i + w/2, ivv, f"{ivv:.4f}", ha="center", va="bottom", fontsize=7)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # -- page 5: signal vs noise summary table
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.axis("off")
        lines = ["SIGNAL vs NOISE — per-strike residual diagnostics",
                 "=" * 80, ""]
        lines.append(f"{'K':>5} {'mean_res':>10} {'sd_res':>10} {'t-stat':>8} "
                     f"{'AR(1)':>7} {'HL(t)':>7} {'edge_PnL':>10}  verdict")
        lines.append("-" * 80)
        # vega ≈ S * pdf(d1) * sqrt(T). Use S=5250, T=7, sigma=0.012 → vega≈5500.
        # But IMC residual edge is realised when mean-reverting. Edge per round trip ≈ 2*sd*vega.
        for k in STRIKES_LIVE:
            sub = iv[iv["K"] == k]
            r = sub["resid_snap"].dropna().values
            if len(r) == 0:
                continue
            mu = r.mean(); sd = r.std()
            tstat = mu / (sd / math.sqrt(len(r))) if sd > 0 else 0
            a_ = r[:-1]; b_ = r[1:]
            phi = np.cov(a_, b_, ddof=0)[0, 1] / np.var(a_) if np.var(a_) > 0 else 0
            hl  = -math.log(2) / math.log(phi) if 0 < phi < 1 else np.nan
            # Approx vega for ATM-ish K, S≈5250, T≈7, σ≈0.012:
            # vega = S*pdf(d1)*sqrt(T). Just approximate per strike.
            S0, T0, sig0 = 5250.0, 7.0, 0.012
            v = sig0 * math.sqrt(T0)
            d1 = (math.log(S0 / k) + 0.5 * sig0 * sig0 * T0) / v
            vega = S0 * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(T0)
            edge_pnl = 2 * sd * vega  # round-trip edge in Xirecs per contract
            verdict = ("MEAN-REVERT" if (not np.isnan(hl) and hl < 200 and abs(tstat) > 3)
                       else "drift-only" if abs(tstat) > 3
                       else "NOISE")
            lines.append(f"{k:>5} {mu:>+10.6f} {sd:>10.6f} {tstat:>+8.1f} "
                         f"{phi:>7.3f} {hl:>7.0f} {edge_pnl:>10.1f}  {verdict}")
        lines += ["", "Interpretation:",
                  "  • t-stat > 3 ⇒ residual mean is statistically non-zero (real drift)",
                  "  • AR(1) > 0 ⇒ residual is autocorrelated (slow-moving, tradeable)",
                  "  • HL = half-life in ticks (lower = faster reversion = more cycles/day)",
                  "  • edge_PnL = 2·σ_resid·vega — ballpark Xirecs per round-trip if you fade",
                  "    the residual at +1σ and exit at the smile.",
                  "",
                  f"Realised σ per day:  {rv}",
                  f"Implied σ per day:   {atm_iv}",
                  f"  → IV {'over' if list(atm_iv.values())[0] > list(rv.values())[0] else 'under'}"
                  f"-prices realised; gap matters for the level (vega-PnL on long/short vol)."]
        ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9, va="top")
        pdf.savefig(fig); plt.close(fig)

    print(f"\n✓ wrote {OUT}")


if __name__ == "__main__":
    main()
