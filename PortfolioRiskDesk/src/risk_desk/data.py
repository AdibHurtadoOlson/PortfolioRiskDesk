"""Market-data loading with explicit cache behavior and input checks."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def normalize_yfinance_frame(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Return a flat date-by-ticker adjusted-close frame from yfinance output.

    yfinance has returned both single-level and MultiIndex columns over time,
    so this routine identifies the adjusted/auto-adjusted close robustly.
    """
    if frame is None or frame.empty:
        raise ValueError("Yahoo Finance returned no price rows. Check symbols/date range and retry.")
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # With auto_adjust=True, Close already represents adjusted prices. Probe
        # both levels because yfinance column order can differ by call/version.
        field_level = None
        field_name = None
        for level in range(out.columns.nlevels):
            candidates = [c for c in out.columns.get_level_values(level).unique()
                          if str(c).lower() in {"close", "adj close"}]
            if candidates:
                field_level = level
                field_name = "Close" if "Close" in candidates else candidates[0]
                break
        if field_level is None:
            raise ValueError("MultiIndex price response contains no Close or Adj Close field.")
        out = out.xs(field_name, axis=1, level=field_level)
    else:
        if "Close" in out.columns:
            out = out[["Close"]].rename(columns={"Close": tickers[0]})
        elif "Adj Close" in out.columns:
            out = out[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        else:
            raise ValueError("Downloaded data does not contain Close or Adj Close.")
    if isinstance(out, pd.Series):
        out = out.to_frame(name=tickers[0])
    # Some yfinance versions return a ticker column even for a single ticker.
    out.columns = [str(c) for c in out.columns]
    missing = sorted(set(tickers) - set(out.columns))
    if missing:
        raise ValueError(f"Price download is missing requested ticker columns: {missing}")
    out = out[tickers].apply(pd.to_numeric, errors="coerce")
    out.index = pd.to_datetime(out.index).tz_localize(None) if getattr(out.index, "tz", None) else pd.to_datetime(out.index)
    out.index.name = "Date"
    return validate_prices(out)


def validate_prices(prices: pd.DataFrame, max_missing_fraction: float = 0.15) -> pd.DataFrame:
    """Validate and clean a price panel without silently hiding bad input."""
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("Price panel is empty.")
    p = prices.copy()
    p.index = pd.to_datetime(p.index)
    if p.index.has_duplicates:
        raise ValueError("Price dates contain duplicates; inspect source data before continuing.")
    p = p.sort_index()
    if not p.index.is_monotonic_increasing:
        raise ValueError("Price dates must be increasing.")
    if (p.dropna(how="all") <= 0).any().any():
        raise ValueError("Prices must be positive wherever observed.")
    missing = p.isna().mean()
    bad = missing[missing > max_missing_fraction]
    if len(bad):
        raise ValueError(f"Too many missing prices for symbols: {bad.to_dict()}")
    # Do not forward-fill prices across long closures or missing data. Only rows
    # with all assets present are used in a common-weight portfolio backtest.
    p = p.dropna(how="any")
    if len(p) < 100:
        raise ValueError(f"Only {len(p)} complete price rows remain; at least 100 are required for risk analytics.")
    return p.astype(float)


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download auto-adjusted daily close data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Missing yfinance. Install project dependencies with `pip install -r requirements.txt`.") from exc
    raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True,
                       actions=False, progress=False, threads=True)
    return normalize_yfinance_frame(raw, tickers)


def get_prices(tickers: list[str], start: str, cache_path: Path, use_cache: bool = False,
               end: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load live prices by default; cached prices are used only when requested."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if use_cache:
        if not cache_path.exists():
            raise FileNotFoundError(f"Requested cache does not exist: {cache_path}. Run once online first.")
        p = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        p = validate_prices(p)
        missing = sorted(set(tickers) - set(p.columns))
        if missing:
            raise ValueError(f"Cache is missing symbols {missing}; refresh it online or use a matching universe.")
        p = p[tickers]
        return p.loc[p.index >= pd.Timestamp(start)], "cache"
    prices = download_prices(tickers, start, end)
    prices.to_csv(cache_path, index_label="Date")
    return prices, "Yahoo Finance via yfinance"
