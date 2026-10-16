"""Lagged VaR backtesting and likelihood-ratio diagnostics."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_pof(violations: pd.Series, expected_probability: float) -> dict[str, float]:
    """Kupiec proportion-of-failures test for correct exception frequency."""
    v = np.asarray(violations, dtype=int)
    if len(v) == 0 or not 0 < expected_probability < 1:
        raise ValueError("Supply observations and an expected probability in (0, 1).")
    n, x = len(v), int(v.sum())
    phat = np.clip(x / n, 1e-12, 1 - 1e-12)
    p = expected_probability
    ll_null = (n - x) * math.log(1 - p) + x * math.log(p)
    ll_alt = (n - x) * math.log(1 - phat) + x * math.log(phat)
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return {"exceptions": x, "observations": n, "exception_rate": x / n,
            "expected_exception_rate": p, "kupiec_lr": lr, "kupiec_pvalue": float(chi2.sf(lr, 1))}


def christoffersen_independence(violations: pd.Series) -> dict[str, float]:
    """Test whether VaR exceptions cluster, using the first-order Markov model."""
    v = np.asarray(violations, dtype=int)
    if len(v) < 2:
        raise ValueError("At least two violation observations are required.")
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))
    def log_bernoulli(success, total, fallback):
        if total == 0:
            return 0.0
        p = np.clip(success / total, 1e-12, 1 - 1e-12)
        return success * math.log(p) + (total - success) * math.log(1 - p)
    total_transitions = n00 + n01 + n10 + n11
    p = (n01 + n11) / max(total_transitions, 1)
    ll_iid = log_bernoulli(n01 + n11, total_transitions, p)
    ll_markov = log_bernoulli(n01, n00 + n01, p) + log_bernoulli(n11, n10 + n11, p)
    lr = max(0.0, -2.0 * (ll_iid - ll_markov))
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "independence_lr": lr, "independence_pvalue": float(chi2.sf(lr, 1))}


def historical_var_backtest(returns: pd.Series, confidence: float = 0.99, window: int = 252) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare prior-window historical VaR forecasts against next-day losses.

    The rolling quantile is shifted one day so a loss is never used to forecast
    its own VaR. Early rows are omitted until a full estimation window exists.
    """
    r = pd.Series(returns).dropna().astype(float)
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1.0.")
    if window < 30 or len(r) <= window + 20:
        raise ValueError("Need at least 20 out-of-sample forecasts beyond the VaR window, with window >= 30.")
    losses = -r
    forecast = losses.rolling(window=window, min_periods=window).quantile(confidence).shift(1)
    frame = pd.DataFrame({"return": r, "loss": losses, "var_forecast": forecast}).dropna()
    frame["violation"] = (frame["loss"] > frame["var_forecast"]).astype(int)
    summary = kupiec_pof(frame["violation"], 1.0 - confidence)
    summary.update(christoffersen_independence(frame["violation"]))
    return frame, summary
