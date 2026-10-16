"""Portfolio return, tail-risk, and risk-statistic calculations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm


def validate_weights(weights: pd.Series, tolerance: float = 1e-6) -> pd.Series:
    """Require finite weights that sum to one; retain any intentional shorts."""
    w = pd.Series(weights, dtype=float)
    if w.empty or not np.isfinite(w).all():
        raise ValueError("Weights must be non-empty and finite.")
    if abs(float(w.sum()) - 1.0) > tolerance:
        raise ValueError(f"Portfolio weights must sum to 1.0; received {w.sum():.8f}.")
    return w


def asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns and remove the leading undefined row."""
    if (prices <= 0).any().any():
        raise ValueError("All prices must be positive.")
    returns = prices.sort_index().pct_change(fill_method=None).dropna(how="any")
    if returns.empty:
        raise ValueError("Not enough observations to calculate returns.")
    return returns


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Aggregate asset returns using a fixed-weight, daily-rebalanced proxy."""
    w = validate_weights(weights)
    missing = sorted(set(w.index) - set(returns.columns))
    if missing:
        raise ValueError(f"Weights reference symbols absent from returns: {missing}")
    r = returns[w.index]
    result = r.mul(w, axis=1).sum(axis=1)
    result.name = "portfolio_return"
    return result


def tail_risk(returns: pd.Series, confidence: float = 0.99) -> dict[str, float]:
    """Calculate historical and normal-parametric one-day loss VaR/CVaR.

    VaR and CVaR are reported as positive loss fractions. CVaR is the average
    loss conditional on loss being at least the estimated VaR.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0.")
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 30:
        raise ValueError("At least 30 returns are required for tail-risk estimates.")
    losses = -r.to_numpy()
    hist_var = float(np.quantile(losses, confidence, method="linear"))
    hist_cvar = float(losses[losses >= hist_var].mean())
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    if sigma <= 0:
        param_var = param_cvar = max(0.0, -mu)
    else:
        z = float(norm.ppf(confidence))
        param_var = -mu + sigma * z
        param_cvar = -mu + sigma * float(norm.pdf(z)) / (1.0 - confidence)
    return {
        "confidence": confidence,
        "historical_var_1d": hist_var,
        "historical_cvar_1d": hist_cvar,
        "normal_var_1d": float(param_var),
        "normal_cvar_1d": float(param_cvar),
    }


def ewma_volatility(returns: pd.Series, decay: float = 0.94) -> pd.Series:
    """Estimate daily volatility using RiskMetrics-style EWMA variance."""
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be between zero and one.")
    r = pd.Series(returns).astype(float)
    # Shift by one observation: the volatility at t uses returns through t-1,
    # which is the correct convention for a next-day forecast.
    variance = r.shift(1).pow(2).ewm(alpha=1.0 - decay, adjust=False, min_periods=20).mean()
    return np.sqrt(variance)


def block_bootstrap_tail_risk(asset_returns: pd.DataFrame, weights: pd.Series, confidence: float = 0.99,
                               horizon_days: int = 10, simulations: int = 25000,
                               block_size: int = 5, random_seed: int = 42) -> dict[str, float]:
    """Estimate multi-day portfolio tail loss using a moving-block bootstrap.

    Contiguous blocks are sampled with replacement to retain some short-term
    serial dependence; the same sampled rows are used for every asset so
    contemporaneous cross-asset correlation is preserved. The daily portfolio
    returns are compounded across the horizon. This is a historical simulation,
    not a forecast guarantee; the empirical sample still limits tail coverage.
    """
    w = validate_weights(weights)
    missing = sorted(set(w.index) - set(asset_returns.columns))
    if missing:
        raise ValueError(f"Weights reference symbols absent from returns: {missing}")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0.")
    if horizon_days < 1 or simulations < 500 or block_size < 1:
        raise ValueError("horizon_days/block_size must be positive and simulations must be at least 500.")
    r = asset_returns[w.index].dropna().to_numpy(dtype=float)
    if len(r) < max(30, block_size + 1):
        raise ValueError("At least 30 complete asset-return rows are required for block bootstrap risk.")
    rng = np.random.default_rng(random_seed)
    blocks_needed = int(np.ceil(horizon_days / block_size))
    # Each row of `starts` selects one contiguous historical block. Sampling
    # from valid starts avoids wraparound blocks that jump from the last date to the first.
    starts = rng.integers(0, len(r) - block_size + 1, size=(simulations, blocks_needed))
    offsets = np.arange(block_size)
    row_indices = (starts[:, :, None] + offsets[None, None, :]).reshape(simulations, -1)[:, :horizon_days]
    sampled_assets = r[row_indices]  # (simulations, horizon_days, assets)
    simulated_daily_portfolio = np.einsum("shk,k->sh", sampled_assets, w.to_numpy(dtype=float))
    cumulative_returns = np.prod(1.0 + simulated_daily_portfolio, axis=1) - 1.0
    losses = -cumulative_returns
    var = float(np.quantile(losses, confidence, method="linear"))
    cvar = float(losses[losses >= var].mean())
    return {"confidence": float(confidence), "horizon_days": int(horizon_days),
            "simulations": int(simulations), "block_size_days": int(block_size),
            "random_seed": int(random_seed), "block_bootstrap_var": var,
            "block_bootstrap_cvar": cvar, "simulated_mean_loss": float(losses.mean()),
            "simulated_loss_std": float(losses.std(ddof=1))}


def performance_metrics(returns: pd.Series, periods_per_year: int = 252, risk_free_annual: float = 0.0) -> dict[str, float]:
    """Calculate annualized summary metrics for a periodic simple-return series."""
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2:
        raise ValueError("At least two return observations are needed.")
    wealth = (1.0 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    annual_return = float((1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0) if (1.0 + r).prod() > 0 else -1.0
    rf_period = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    sharpe = float(((r - rf_period).mean() / r.std(ddof=1)) * np.sqrt(periods_per_year)) if r.std(ddof=1) > 0 else float("nan")
    return {
        "observations": int(len(r)),
        "annualized_return": annual_return,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "best_period_return": float(r.max()),
        "worst_period_return": float(r.min()),
        "hit_rate": float((r > 0).mean()),
    }
