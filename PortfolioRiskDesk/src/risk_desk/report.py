"""Write a compact daily risk pack for review and handoff."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .analytics import asset_returns, portfolio_returns, tail_risk, performance_metrics, ewma_volatility, block_bootstrap_tail_risk
from .backtest import historical_var_backtest
from .stress import scenario_table


def create_report(prices: pd.DataFrame, weights: pd.Series, confidence: float, outdir: Path, source: str, simulations: int = 25000) -> dict:
    """Run the complete risk workflow and save machine-readable deliverables."""
    outdir.mkdir(parents=True, exist_ok=True)
    rets = asset_returns(prices)
    p_ret = portfolio_returns(rets, weights)
    risk = tail_risk(p_ret, confidence)
    bootstrap_risk = block_bootstrap_tail_risk(rets, weights, confidence=confidence,
                                               horizon_days=10, simulations=simulations, block_size=5)
    perf = performance_metrics(p_ret)
    summary = {"data_source": source, "start_date": str(prices.index.min().date()),
               "end_date": str(prices.index.max().date()), "assets": list(weights.index),
               "weights": weights.to_dict(), "portfolio_performance": perf, "tail_risk": risk,
               "block_bootstrap_10d_tail_risk": bootstrap_risk,
               "ewma_volatility_latest": float(ewma_volatility(p_ret).dropna().iloc[-1])}
    (outdir / "risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    asset_rows = []
    for col in rets.columns:
        a = tail_risk(rets[col], confidence)
        a.update({"ticker": col, "weight": float(weights[col]),
                  "annualized_volatility": float(rets[col].std(ddof=1) * (252 ** 0.5)),
                  "weighted_mean_daily_return": float(weights[col] * rets[col].mean())})
        asset_rows.append(a)
    pd.DataFrame(asset_rows).set_index("ticker").to_csv(outdir / "asset_risk.csv")
    rets.corr().to_csv(outdir / "correlation.csv")
    scenario_table(weights).to_csv(outdir / "stress_scenarios.csv")
    # VaR backtest uses a full rolling window if enough data is available.
    if len(p_ret) > 252:
        bt, bt_summary = historical_var_backtest(p_ret, confidence, 252)
        bt.to_csv(outdir / "var_backtest.csv", index_label="Date")
        summary["var_backtest"] = bt_summary
        (outdir / "risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        pd.DataFrame(columns=["return", "loss", "var_forecast", "violation"]).to_csv(outdir / "var_backtest.csv")
        summary["var_backtest_note"] = "Skipped: fewer than 253 portfolio-return observations."
        (outdir / "risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wealth = (1 + p_ret).cumprod()
    roll_vol = p_ret.rolling(63).std() * (252 ** 0.5)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    wealth.plot(ax=axes[0], title="Portfolio growth of $1 (daily rebalanced proxy)")
    axes[0].set_ylabel("Growth")
    (roll_vol).plot(ax=axes[1], title="63-day rolling annualized volatility")
    axes[1].set_ylabel("Volatility")
    (-p_ret).hist(bins=50, ax=axes[2])
    axes[2].axvline(risk["historical_var_1d"], linestyle="--", label="Historical VaR")
    axes[2].set_title("Daily loss distribution"); axes[2].legend()
    fig.savefig(outdir / "risk_dashboard.png", dpi=160); plt.close(fig)
    return summary
