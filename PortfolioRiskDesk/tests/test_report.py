import numpy as np
import pandas as pd
from riskdesk.report import create_report

def test_report_writes_risk_pack(tmp_path):
    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2021-01-01", periods=420)
    returns = rng.normal(0.0003, 0.012, size=(420, 2))
    prices = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=["SPY", "TLT"])
    summary = create_report(prices, pd.Series({"SPY": 0.6, "TLT": 0.4}), 0.95, tmp_path, "test fixture")
    assert summary["tail_risk"]["historical_var_1d"] > 0
    for name in ["risk_summary.json", "asset_risk.csv", "stress_scenarios.csv", "var_backtest.csv", "risk_dashboard.png"]:
        assert (tmp_path / name).exists()
