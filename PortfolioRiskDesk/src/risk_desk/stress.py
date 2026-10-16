"""Transparent scenario analysis at asset and portfolio level."""
from __future__ import annotations
import pandas as pd

DEFAULT_SCENARIOS = {
    "equity_selloff": {"SPY": -0.15, "QQQ": -0.20, "IWM": -0.22, "TLT": 0.04, "GLD": 0.03},
    "rates_selloff": {"SPY": -0.06, "QQQ": -0.08, "IWM": -0.07, "TLT": -0.15, "GLD": -0.02},
    "flight_to_quality": {"SPY": -0.08, "QQQ": -0.10, "IWM": -0.12, "TLT": 0.10, "GLD": 0.06},
    "inflation_shock": {"SPY": -0.10, "QQQ": -0.13, "IWM": -0.12, "TLT": -0.12, "GLD": 0.08},
}


def scenario_table(weights: pd.Series, scenarios: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Apply percentage shocks and attribute approximate P&L by holding."""
    scenarios = scenarios or DEFAULT_SCENARIOS
    rows = []
    for name, shocks in scenarios.items():
        contributions = {ticker: float(weights[ticker]) * float(shocks.get(ticker, 0.0)) for ticker in weights.index}
        rows.append({"scenario": name, **{f"shock_{k}": float(shocks.get(k, 0.0)) for k in weights.index},
                     **{f"contribution_{k}": v for k, v in contributions.items()},
                     "portfolio_return_shock": sum(contributions.values())})
    return pd.DataFrame(rows).set_index("scenario")
