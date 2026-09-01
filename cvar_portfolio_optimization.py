"""
CVaR Portfolio Optimization (Tail Risk)
========================================
Portfolio optimizer that minimizes Conditional Value-at-Risk (CVaR) using
the linear Rockafellar-Uryasev formulation and CVXPY.

The program:
    1. downloads adjusted prices;
    2. calculates daily returns;
    3. splits the data into training and test periods;
    4. optimizes Min-CVaR and Min-Variance on the training set;
    5. compares portfolios in sample and out of sample;
    6. builds a CVaR efficient frontier;
    7. produces charts.

Designed for Spyder and executable cell by cell through # %% markers.

Dependencies:
    pip install numpy pandas cvxpy yfinance matplotlib
"""

# %% 1. IMPORT
import numpy as np
import pandas as pd
import cvxpy as cp
import yfinance as yf
import matplotlib.pyplot as plt


# %% 2. CONFIGURATION
TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "JPM",
    "XOM",
    "JNJ",
    "V",
]

START = "2018-01-01"

# yfinance treats END as an exclusive date.
# Using 2025-01-01 therefore includes all of 2024.
END = "2025-01-01"

# Time split for the out-of-sample test:
# train = 2018-2022
# test  = 2023-2024
SPLIT_DATE = "2023-01-01"

ALPHA = 0.95
WEIGHT_CAP = 0.40
ALLOW_SHORT = False

TRADING_DAYS = 252


# %% 3. SUPPORT FUNCTIONS
def validate_returns(returns):
    """Validate that the return matrix is usable."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("The returns DataFrame is empty.")

    if returns.shape[0] < 2:
        raise ValueError("At least two observations are required.")

    if returns.shape[1] < 1:
        raise ValueError("No assets are present.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain NaN or infinite values.")


def validate_alpha(alpha):
    """Validate the CVaR confidence level."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def build_weight_constraints(w, n_assets, allow_short, weight_cap):
    """
    Build portfolio-weight constraints.

    Long-only:
        0 <= w_i <= weight_cap

    Short selling:
        -weight_cap <= w_i <= weight_cap
    """
    if weight_cap <= 0:
        raise ValueError("weight_cap must be positive.")

    # Because sum(w)=1, a total portfolio weight of 1 must be feasible.
    if n_assets * weight_cap < 1 - 1e-10:
        raise ValueError(
            "Infeasible constraints: N * WEIGHT_CAP must be at least 1."
        )

    constraints = [cp.sum(w) == 1]

    if allow_short:
        constraints += [
            w >= -weight_cap,
            w <= weight_cap,
        ]
    else:
        constraints += [
            w >= 0,
            w <= weight_cap,
        ]

    return constraints


def solve_problem(problem):
    """
    Solve a CVXPY problem using available reliable solvers.

    CLARABEL is the first choice in modern CVXPY versions.
    ECOS and SCS are fallbacks.
    """
    installed = set(cp.installed_solvers())
    solver_order = ["CLARABEL", "ECOS", "SCS"]

    errors = []

    for solver in solver_order:
        if solver not in installed:
            continue

        try:
            problem.solve(solver=solver)
        except cp.error.SolverError as exc:
            errors.append(f"{solver}: {exc}")
            continue

        if problem.status == cp.OPTIMAL:
            return solver

        if problem.status == cp.OPTIMAL_INACCURATE:
            print(
                f"WARNING: {solver} returned "
                "'optimal_inaccurate'."
            )
            return solver

        errors.append(f"{solver}: status={problem.status}")

    raise RuntimeError(
        "No solver produced a valid solution. "
        + " | ".join(errors)
    )


# %% 4. DATA DOWNLOAD
def download_prices(tickers, start, end):
    """Download adjusted closing prices from Yahoo Finance."""

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")

    # With multiple tickers, yfinance normally returns a MultiIndex.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("Close column not found in downloaded data.")

        prices = raw["Close"].copy()

    else:
        if "Close" not in raw.columns:
            raise RuntimeError("Close column not found in downloaded data.")

        prices = raw[["Close"]].copy()

        if len(tickers) == 1:
            prices.columns = [tickers[0]]

    # Check for missing tickers.
    missing_tickers = [
        ticker
        for ticker in tickers
        if ticker not in prices.columns
        or prices[ticker].isna().all()
    ]

    if missing_tickers:
        raise RuntimeError(
            "Missing data for tickers: "
            + ", ".join(missing_tickers)
        )

    # Preserve the order specified in TICKERS.
    prices = prices[list(tickers)]

    # Do not forward-fill: this avoids creating artificial zero returns.
    rows_before = len(prices)
    prices = prices.dropna(how="any")
    rows_removed = rows_before - len(prices)

    if rows_removed > 0:
        print(
            f"Removed {rows_removed} dates with at least one missing price."
        )

    if prices.empty:
        raise RuntimeError(
            "No complete observation is available after cleaning."
        )

    return prices


def compute_returns(prices):
    """Calculate simple daily returns."""
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    validate_returns(returns)
    return returns


prices = download_prices(TICKERS, START, END)
returns = compute_returns(prices)

print("\nData period:")
print(f"From: {returns.index.min().date()}")
print(f"To:   {returns.index.max().date()}")
print(f"Number of observations: {len(returns)}")

print("\nDescriptive statistics:")
print(returns.describe())


# %% 5. TRAIN / TEST SPLIT OUT-OF-SAMPLE
split_date = pd.Timestamp(SPLIT_DATE)

train_returns = returns.loc[returns.index < split_date].copy()
test_returns = returns.loc[returns.index >= split_date].copy()

validate_returns(train_returns)
validate_returns(test_returns)

print("\n--- TRAIN / TEST SPLIT ---")
print(
    f"Train: {train_returns.index.min().date()} -> "
    f"{train_returns.index.max().date()} "
    f"({len(train_returns)} observations)"
)

print(
    f"Test:  {test_returns.index.min().date()} -> "
    f"{test_returns.index.max().date()} "
    f"({len(test_returns)} observations)"
)


# %% 6. CVaR OPTIMIZATION - ROCKAFELLAR & URYASEV
def optimize_cvar(
    returns,
    alpha=0.95,
    target_return=None,
    allow_short=False,
    weight_cap=1.0,
):
    """
    Minimize empirical CVaR:

        min_{w,zeta,u}
            zeta + 1/((1-alpha)*T) * sum(u_t)

        s.t.
            u_t >= 0
            u_t >= -R_t'w - zeta
            sum(w) = 1

    where:
        w     = portfolio weights
        zeta  = VaR threshold
        u_t   = loss above VaR in scenario t

    target_return, when specified, is an annualized arithmetic mean return.
    """

    validate_returns(returns)
    validate_alpha(alpha)

    T, N = returns.shape
    R = returns.to_numpy(dtype=float)

    w = cp.Variable(N)
    zeta = cp.Variable()

    # u_t >= 0 is embedded directly in the variable definition.
    u = cp.Variable(T, nonneg=True)

    portfolio_returns = R @ w
    losses = -portfolio_returns

    cvar_expr = (
        zeta
        + cp.sum(u) / ((1 - alpha) * T)
    )

    constraints = build_weight_constraints(
        w,
        N,
        allow_short,
        weight_cap,
    )

    constraints.append(
        u >= losses - zeta
    )

    if target_return is not None:
        mean_daily = returns.mean().to_numpy()

        constraints.append(
            mean_daily @ w
            >= target_return / TRADING_DAYS
        )

    problem = cp.Problem(
        cp.Minimize(cvar_expr),
        constraints,
    )

    solver_used = solve_problem(problem)

    if w.value is None or zeta.value is None:
        raise RuntimeError(
            "The solver did not return a numerical solution."
        )

    weights = pd.Series(
        np.asarray(w.value).ravel(),
        index=returns.columns,
        dtype=float,
    )

    # Remove tiny numerical residuals.
    weights[np.abs(weights) < 1e-10] = 0.0

    if abs(weights.sum() - 1) > 1e-5:
        raise RuntimeError(
            "The resulting weights do not numerically sum to 1."
        )

    return (
        weights,
        float(cvar_expr.value),
        float(zeta.value),
        problem.status,
        solver_used,
    )


w_cvar, cvar_val, var_val, status, solver_cvar = optimize_cvar(
    train_returns,
    alpha=ALPHA,
    allow_short=ALLOW_SHORT,
    weight_cap=WEIGHT_CAP,
)

print("\n--- MIN-CVaR PORTFOLIO ---")
print(f"Solver: {solver_cvar}")
print(f"Status: {status}")

print("\nWeights:")
print(w_cvar.round(4))

print(f"\nWeight sum: {w_cvar.sum():.6f}")
print(
    f"Daily optimizer CVaR ({int(ALPHA * 100)}%): "
    f"{cvar_val:.4%}"
)
print(
    f"Daily VaR threshold zeta ({int(ALPHA * 100)}%): "
    f"{var_val:.4%}"
)


# %% 7. MINIMUM-VARIANCE PORTFOLIO
def optimize_min_variance(
    returns,
    allow_short=False,
    weight_cap=1.0,
):
    """Calculate the global minimum-variance portfolio."""

    validate_returns(returns)

    N = returns.shape[1]

    sigma = returns.cov().to_numpy(dtype=float)

    # Reduce possible asymmetries caused by numerical precision.
    sigma = (sigma + sigma.T) / 2

    w = cp.Variable(N)

    constraints = build_weight_constraints(
        w,
        N,
        allow_short,
        weight_cap,
    )

    objective = cp.Minimize(
        cp.quad_form(w, sigma)
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    solver_used = solve_problem(problem)

    if w.value is None:
        raise RuntimeError(
            "Min-Variance: no numerical solution is available."
        )

    weights = pd.Series(
        np.asarray(w.value).ravel(),
        index=returns.columns,
        dtype=float,
    )

    weights[np.abs(weights) < 1e-10] = 0.0

    return weights, problem.status, solver_used


w_mv, status_mv, solver_mv = optimize_min_variance(
    train_returns,
    allow_short=ALLOW_SHORT,
    weight_cap=WEIGHT_CAP,
)

print("\n--- MIN-VARIANCE PORTFOLIO ---")
print(f"Solver: {solver_mv}")
print(f"Status: {status_mv}")

print("\nWeights:")
print(w_mv.round(4))

print(f"\nWeight sum: {w_mv.sum():.6f}")


# Equal-weight portfolio as an additional benchmark.
w_equal = pd.Series(
    1 / len(TICKERS),
    index=train_returns.columns,
    dtype=float,
)

print("\n--- EQUAL-WEIGHT PORTFOLIO ---")
print(w_equal.round(4))


# %% 8. EMPIRICAL VaR AND CVaR CONSISTENT WITH ROCKAFELLAR-URYASEV
def empirical_var_cvar(portfolio_returns, alpha=0.95):
    """
    Calculate VaR and CVaR on the empirical loss distribution.

    CVaR is calculated with the same Rockafellar-Uryasev function used
    during optimization.
    """

    validate_alpha(alpha)

    losses = -np.asarray(
        portfolio_returns,
        dtype=float,
    )

    if losses.size == 0:
        raise ValueError("The return series is empty.")

    if not np.isfinite(losses).all():
        raise ValueError(
            "The series contains NaN or infinite values."
        )

    n = len(losses)

    # Empirical VaR: first value in the empirical distribution for which
    # F(x) >= alpha.
    sorted_losses = np.sort(losses)

    var_index = int(np.ceil(alpha * n)) - 1
    var_value = sorted_losses[var_index]

    # Rockafellar-Uryasev formula.
    cvar_value = (
        var_value
        + np.maximum(losses - var_value, 0).sum()
        / ((1 - alpha) * n)
    )

    return float(var_value), float(cvar_value)


# %% 9. RISK / RETURN METRICS
def portfolio_metrics(
    returns,
    weights,
    alpha=0.95,
):
    """
    Calculate portfolio metrics.

    Note:
    returns @ weights corresponds to a constant-weight portfolio. In a
    time-series evaluation, this means keeping weights constant through
    rebalancing.
    """

    validate_returns(returns)
    validate_alpha(alpha)

    portfolio_returns = returns @ weights

    mean_daily = portfolio_returns.mean()
    vol_daily = portfolio_returns.std()

    ann_return = mean_daily * TRADING_DAYS
    ann_vol = vol_daily * np.sqrt(TRADING_DAYS)

    # Sharpe ratio assuming a zero risk-free rate.
    if ann_vol > 0:
        sharpe = ann_return / ann_vol
    else:
        sharpe = np.nan

    var_hist, cvar_hist = empirical_var_cvar(
        portfolio_returns,
        alpha,
    )

    # Wealth index for simple returns.
    if (1 + portfolio_returns <= 0).any():
        # The portfolio lost at least 100% in one day.
        max_drawdown = -1.0
    else:
        wealth = (1 + portfolio_returns).cumprod()

        # Implicitly include initial capital W_0 = 1.
        running_peak = wealth.cummax().clip(lower=1.0)

        drawdowns = (
            wealth / running_peak - 1
        )

        max_drawdown = drawdowns.min()

    return {
        "Mean annual return": float(ann_return),
        "Annual volatility": float(ann_vol),
        "Sharpe (rf=0)": float(sharpe),
        f"Daily VaR {int(alpha * 100)}%": var_hist,
        f"Daily CVaR {int(alpha * 100)}%": cvar_hist,
        "Max Drawdown": float(max_drawdown),
    }


def print_metrics(title, metrics):
    """Print metrics using the appropriate format."""

    print(f"\n--- {title} ---")

    for name, value in metrics.items():

        if name.startswith("Sharpe"):
            print(f"{name}: {value:.4f}")

        else:
            print(f"{name}: {value:.4%}")


# %% 10. IN-SAMPLE RESULTS
metrics_cvar_train = portfolio_metrics(
    train_returns,
    w_cvar,
    ALPHA,
)

metrics_mv_train = portfolio_metrics(
    train_returns,
    w_mv,
    ALPHA,
)

metrics_equal_train = portfolio_metrics(
    train_returns,
    w_equal,
    ALPHA,
)

print_metrics(
    "TRAIN - Min-CVaR",
    metrics_cvar_train,
)

print_metrics(
    "TRAIN - Min-Variance",
    metrics_mv_train,
)

print_metrics(
    "TRAIN - Equal Weight",
    metrics_equal_train,
)


# Consistency check: this CVaR should be nearly identical to the solver's
# objective value.
cvar_check = metrics_cvar_train[
    f"Daily CVaR {int(ALPHA * 100)}%"
]

print("\nCVaR consistency check:")
print(f"Optimizer:       {cvar_val:.8%}")
print(f"Recalculated:    {cvar_check:.8%}")
print(
    f"Difference:      "
    f"{abs(cvar_val - cvar_check):.10f}"
)


# %% 11. TRUE OUT-OF-SAMPLE TEST
# Weights are estimated exclusively on the training set. The test set is not
# used during optimization.

metrics_cvar_test = portfolio_metrics(
    test_returns,
    w_cvar,
    ALPHA,
)

metrics_mv_test = portfolio_metrics(
    test_returns,
    w_mv,
    ALPHA,
)

metrics_equal_test = portfolio_metrics(
    test_returns,
    w_equal,
    ALPHA,
)

print("\n========================================")
print("        OUT-OF-SAMPLE RESULTS")
print("========================================")

print_metrics(
    "TEST - Min-CVaR",
    metrics_cvar_test,
)

print_metrics(
    "TEST - Min-Variance",
    metrics_mv_test,
)

print_metrics(
    "TEST - Equal Weight",
    metrics_equal_test,
)


# %% 12. MAXIMUM FEASIBLE RETURN
def maximum_feasible_return(
    returns,
    allow_short=False,
    weight_cap=1.0,
):
    """
    Calculate the maximum annualized mean return achievable under the same
    constraints used during optimization.
    """

    validate_returns(returns)

    N = returns.shape[1]
    mean_daily = returns.mean().to_numpy()

    w = cp.Variable(N)

    constraints = build_weight_constraints(
        w,
        N,
        allow_short,
        weight_cap,
    )

    problem = cp.Problem(
        cp.Maximize(mean_daily @ w),
        constraints,
    )

    solve_problem(problem)

    if w.value is None:
        raise RuntimeError(
            "Unable to determine the maximum feasible return."
        )

    max_return = (
        float(mean_daily @ w.value)
        * TRADING_DAYS
    )

    return max_return


# %% 13. CVaR EFFICIENT FRONTIER
def efficient_frontier_cvar(
    returns,
    alpha=0.95,
    n_points=20,
    allow_short=False,
    weight_cap=1.0,
):
    """
    Build the return-CVaR efficient frontier.

    The frontier starts at the global minimum-CVaR portfolio and extends to
    the maximum feasible return under the same constraints.
    """

    validate_returns(returns)
    validate_alpha(alpha)

    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    mean_daily = returns.mean().to_numpy()

    # Minimum-CVaR point without a return constraint.
    w_min_cvar, _, _, _, _ = optimize_cvar(
        returns,
        alpha=alpha,
        target_return=None,
        allow_short=allow_short,
        weight_cap=weight_cap,
    )

    min_cvar_return = (
        float(
            mean_daily
            @ w_min_cvar.to_numpy()
        )
        * TRADING_DAYS
    )

    max_return = maximum_feasible_return(
        returns,
        allow_short=allow_short,
        weight_cap=weight_cap,
    )

    targets = np.linspace(
        min_cvar_return,
        max_return,
        n_points,
    )

    rows = []

    for target in targets:

        weights, cvar, _, status, _ = optimize_cvar(
            returns,
            alpha=alpha,
            target_return=target,
            allow_short=allow_short,
            weight_cap=weight_cap,
        )

        achieved_return = (
            float(
                mean_daily
                @ weights.to_numpy()
            )
            * TRADING_DAYS
        )

        rows.append(
            {
                "target_return": target,
                "return": achieved_return,
                "cvar": cvar,
                "status": status,
            }
        )

    frontier = pd.DataFrame(rows)

    if frontier.empty:
        raise RuntimeError(
            "Unable to build the CVaR frontier."
        )

    return frontier


frontier = efficient_frontier_cvar(
    train_returns,
    alpha=ALPHA,
    n_points=20,
    allow_short=ALLOW_SHORT,
    weight_cap=WEIGHT_CAP,
)

print("\n--- CVaR FRONTIER ---")
print(frontier.round(6))


# %% 14. PORTFOLIO-WEIGHT CHART
fig, ax = plt.subplots(
    figsize=(10, 5)
)

weights_comparison = pd.DataFrame(
    {
        "Min-CVaR": w_cvar,
        "Min-Variance": w_mv,
        "Equal Weight": w_equal,
    }
)

weights_comparison.plot(
    kind="bar",
    ax=ax,
)

ax.set_title(
    "Portfolio weights estimated on the training set"
)

ax.set_xlabel("Asset")
ax.set_ylabel("Weight")

ax.axhline(
    0,
    color="black",
    linewidth=0.8,
)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "weights_comparison.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# %% 15. CVaR EFFICIENT-FRONTIER CHART
plt.figure(
    figsize=(8, 6)
)

plt.plot(
    frontier["cvar"],
    frontier["return"],
    "o-",
    linewidth=1.5,
)

plt.scatter(
    cvar_val,
    float(
        train_returns.mean().to_numpy()
        @ w_cvar.to_numpy()
    ) * TRADING_DAYS,
    color="red",
    s=70,
    label="Global Min-CVaR",
    zorder=5,
)

plt.xlabel(
    f"Daily CVaR ({int(ALPHA * 100)}%)"
)

plt.ylabel(
    "Annualized mean return"
)

plt.title(
    "CVaR Efficient Frontier - Training Set"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "cvar_efficient_frontier.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# %% 16. OUT-OF-SAMPLE SUMMARY COMPARISON
comparison_test = pd.DataFrame(
    {
        "Min-CVaR": metrics_cvar_test,
        "Min-Variance": metrics_mv_test,
        "Equal Weight": metrics_equal_test,
    }
)

print("\n========================================")
print("     FINAL OUT-OF-SAMPLE COMPARISON")
print("========================================")

print(comparison_test)
