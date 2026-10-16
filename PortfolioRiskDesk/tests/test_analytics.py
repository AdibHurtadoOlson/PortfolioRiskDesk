import numpy as np
import pandas as pd
import pytest
from riskdesk.analytics import validate_weights, asset_returns, portfolio_returns, tail_risk, ewma_volatility, performance_metrics


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError): validate_weights(pd.Series([0.2, 0.7]))
    assert validate_weights(pd.Series([1.2, -0.2])).sum() == pytest.approx(1.0)


def test_returns_and_portfolio_weights():
    prices = pd.DataFrame({"A": [100, 110, 99], "B": [50, 50, 55]}, index=pd.date_range("2020-01-01", periods=3))
    rets = asset_returns(prices)
    assert rets.iloc[0]["A"] == pytest.approx(0.10)
    got = portfolio_returns(rets, pd.Series({"A": 0.5, "B": 0.5}))
    assert got.iloc[0] == pytest.approx(0.05)


def test_tail_risk_cvar_exceeds_var_in_skewed_loss_sample():
    r = pd.Series(np.r_[np.linspace(-0.03, 0.02, 99), -0.15])
    metrics = tail_risk(r, 0.95)
    assert metrics["historical_cvar_1d"] >= metrics["historical_var_1d"]
    assert metrics["historical_var_1d"] > 0


def test_ewma_forecast_is_lagged():
    r = pd.Series(np.r_[np.zeros(30), 0.5])
    vol = ewma_volatility(r)
    assert pd.isna(vol.iloc[0])
    # Volatility immediately before the final shock must not incorporate it.
    assert vol.iloc[-1] == pytest.approx(vol.iloc[-2])
    assert vol.iloc[-1] >= 0


def test_block_bootstrap_risk_is_reproducible_and_validated():
    from riskdesk.analytics import block_bootstrap_tail_risk
    rng = np.random.default_rng(23)
    r = pd.DataFrame(rng.normal(0.0002, 0.01, size=(250, 2)), columns=["A", "B"])
    weights = pd.Series({"A": 0.6, "B": 0.4})
    one = block_bootstrap_tail_risk(r, weights, confidence=0.95, simulations=2000, random_seed=9)
    two = block_bootstrap_tail_risk(r, weights, confidence=0.95, simulations=2000, random_seed=9)
    assert one["block_bootstrap_var"] == pytest.approx(two["block_bootstrap_var"])
    assert one["block_bootstrap_cvar"] >= one["block_bootstrap_var"]
    with pytest.raises(ValueError): block_bootstrap_tail_risk(r, weights, simulations=100)


def test_performance_metrics_bounds():
    r = pd.Series([0.01, -0.02, 0.03, -0.01])
    m = performance_metrics(r, periods_per_year=12)
    assert m["observations"] == 4
    assert m["max_drawdown"] <= 0
    assert 0 <= m["hit_rate"] <= 1
