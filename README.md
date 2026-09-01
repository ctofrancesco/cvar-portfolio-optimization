# CVaR Portfolio Optimization

A Spyder-friendly portfolio research project that minimizes empirical
Conditional Value-at-Risk using the Rockafellar-Uryasev formulation in CVXPY.

## What it demonstrates

- Historical price collection and return cleaning
- Explicit chronological train/test split
- Long-only Min-CVaR and global minimum-variance optimization
- Equal-weight benchmark
- Empirical VaR, CVaR, volatility, Sharpe ratio, and maximum drawdown
- Out-of-sample comparison using weights estimated only on the training set
- CVaR efficient-frontier construction

## Run in Spyder

1. Open `cvar_portfolio_optimization.py`.
2. Install the packages in `requirements.txt` in Spyder's Python environment.
3. Review the configuration block, then run the file or execute its `# %%`
   cells individually.

The default universe contains eight US equities. The script uses 2018-2022
as training data and 2023-2024 as an out-of-sample test. It also writes two
PNG charts to the current working directory.

## Methodological notes

This is a research demonstration, not a full production backtest. It omits
transaction costs, turnover constraints, taxes, and dynamic re-estimation.
The results are not investment advice.

Data source: Yahoo Finance through `yfinance`.
