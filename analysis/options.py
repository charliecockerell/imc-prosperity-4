"""
Black-Scholes pricer, implied vol solver, Greeks.

Conventions (IMC Prosperity):
- r = 0 (Xirecs currency, no discounting)
- q = 0 (no dividends)
- TTE measured in "days"; vol expressed as per-day stdev of log-returns
- Vouchers are European calls on VELVETFRUIT_EXTRACT
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call price with r=q=0."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / v
    d2 = d1 - v
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else (0.5 if S == K else 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / v
    return _norm_cdf(d1)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / v
    return S * _norm_pdf(d1) * math.sqrt(T)


def implied_vol(price: float, S: float, K: float, T: float,
                tol: float = 1e-6, max_iter: int = 80) -> float | None:
    """Newton-Raphson IV solver with bisection fallback. Returns None if no solution."""
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-9 or T <= 0:
        return None
    # Upper bound: call cannot exceed S.
    if price >= S:
        return None

    # Newton with bisection fallback for robustness
    lo, hi = 1e-6, 5.0
    sigma = 0.02  # 2% per day initial guess
    for _ in range(max_iter):
        p = bs_call(S, K, T, sigma)
        diff = p - price
        if abs(diff) < tol:
            return sigma
        v = bs_vega(S, K, T, sigma)
        if v < 1e-10:
            break
        new_sigma = sigma - diff / v
        if new_sigma <= lo or new_sigma >= hi:
            # fall back to bisection
            break
        sigma = new_sigma
    # Bisection
    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = bs_call(S, K, T, mid)
        if p > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return 0.5 * (lo + hi)
    return 0.5 * (lo + hi)


def moneyness_m(S: float, K: float, T: float) -> float:
    """Standard log-moneyness normalised by sqrt(T): m = ln(K/S)/sqrt(T).
    Useful for smile fitting because IV(m) is approximately time-invariant."""
    return math.log(K / S) / math.sqrt(max(T, 1e-9))


@dataclass
class VoucherQuote:
    strike: int
    mid: float
    bid: float
    ask: float


def iv_from_quotes(S: float, vouchers: list[VoucherQuote], T: float) -> list[tuple[int, float | None]]:
    """Batch IV computation for a snapshot."""
    out = []
    for v in vouchers:
        iv = implied_vol(v.mid, S, float(v.strike), T)
        out.append((v.strike, iv))
    return out
