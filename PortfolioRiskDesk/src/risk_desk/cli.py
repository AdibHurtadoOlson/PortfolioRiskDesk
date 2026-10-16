"""Command-line entry point for refreshing prices and producing the risk pack."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .data import get_prices
from .report import create_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a daily market-risk analysis with Yahoo Finance prices.")
    parser.add_argument("--tickers", default="SPY,QQQ,IWM,TLT,GLD", help="Comma-separated Yahoo Finance symbols")
    parser.add_argument("--weights", default="0.35,0.20,0.15,0.20,0.10", help="Comma-separated weights in ticker order; must sum to 1")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None, help="Exclusive end date (YYYY-MM-DD); default is latest available")
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--simulations", type=int, default=25000, help="Number of 10-day moving-block bootstrap paths")
    parser.add_argument("--use-cache", action="store_true", help="Use an existing cached file rather than refreshing the live feed")
    parser.add_argument("--cache", default="data/cache/prices.csv")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    values = [float(x.strip()) for x in args.weights.split(",")]
    if len(tickers) != len(values):
        parser.error("--tickers and --weights must contain the same number of values")
    if len(set(tickers)) != len(tickers):
        parser.error("Duplicate ticker symbols are not allowed")
    weights = pd.Series(values, index=tickers, dtype=float)
    prices, source = get_prices(tickers, args.start, Path(args.cache), args.use_cache, args.end)
    summary = create_report(prices, weights, args.confidence, Path(args.outdir), source, args.simulations)
    print(f"Risk pack written to {Path(args.outdir).resolve()}")
    print(f"Data source: {source}; latest observation: {prices.index.max().date()}")
    print(f"Historical VaR ({args.confidence:.1%}): {summary['tail_risk']['historical_var_1d']:.3%}; "
          f"CVaR: {summary['tail_risk']['historical_cvar_1d']:.3%}")
    print(f"10-day block-bootstrap VaR: {summary['block_bootstrap_10d_tail_risk']['block_bootstrap_var']:.2%}")

if __name__ == "__main__":
    main()
