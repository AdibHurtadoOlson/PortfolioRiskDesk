import numpy as np
import pandas as pd
import pytest
from riskdesk.data import normalize_yfinance_frame, validate_prices

def test_yfinance_field_first_multiindex_is_normalized():
    dates = pd.date_range("2022-01-01", periods=120, freq="B")
    columns = pd.MultiIndex.from_product([["Close", "Open"], ["AAA", "BBB"]])
    values = np.column_stack([np.linspace(10, 20, 120), np.linspace(20, 25, 120),
                              np.linspace(9, 19, 120), np.linspace(19, 24, 120)])
    raw = pd.DataFrame(values, index=dates, columns=columns)
    normalized = normalize_yfinance_frame(raw, ["AAA", "BBB"])
    assert list(normalized.columns) == ["AAA", "BBB"]
    assert len(normalized) == 120

def test_yfinance_ticker_first_multiindex_is_normalized():
    dates = pd.date_range("2022-01-01", periods=120, freq="B")
    columns = pd.MultiIndex.from_product([["AAA", "BBB"], ["Close", "Open"]])
    raw = pd.DataFrame(np.tile(np.array([10, 9, 20, 19]), (120, 1)) + np.arange(120)[:, None] * 0.01,
                       index=dates, columns=columns)
    normalized = normalize_yfinance_frame(raw, ["AAA", "BBB"])
    assert list(normalized.columns) == ["AAA", "BBB"]

def test_bad_price_data_fails_loudly():
    prices = pd.DataFrame({"A": [10, 11, -1]}, index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError): validate_prices(prices)
