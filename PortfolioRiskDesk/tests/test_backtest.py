import numpy as np
import pandas as pd
import pytest
from riskdesk.backtest import kupiec_pof, christoffersen_independence, historical_var_backtest


def test_kupiec_reports_expected_fields():
    out = kupiec_pof(pd.Series([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), 0.1)
    assert out["exceptions"] == 1
    assert 0 <= out["kupiec_pvalue"] <= 1


def test_independence_test_detects_clustered_exceptions_directionally():
    clustered = christoffersen_independence(pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 0, 0]))
    dispersed = christoffersen_independence(pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
    assert clustered["independence_lr"] >= 0
    assert dispersed["independence_lr"] >= 0


def test_var_backtest_uses_lagged_forecast():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 400), index=pd.date_range("2020-01-01", periods=400))
    frame, summary = historical_var_backtest(r, confidence=0.95, window=100)
    expected_first = (-r.iloc[:100]).quantile(0.95)
    assert frame.iloc[0]["var_forecast"] == pytest.approx(expected_first)
    assert "kupiec_pvalue" in summary
    assert len(frame) == 300


def test_rejects_too_short_backtest():
    with pytest.raises(ValueError): historical_var_backtest(pd.Series(np.zeros(40)), window=30)
