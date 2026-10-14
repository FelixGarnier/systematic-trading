"""
src/systrade/backtest/report.py
────────────────────────────────
Visualisations du bloc 9.

    reports/tearsheet.html        QuantStats, tearsheet complet
    reports/dashboard.html        Plotly : equity, drawdown, Sharpe glissant, classes
    reports/equity_drawdown.png   Matplotlib, figure statique
    reports/walk_forward.png      Matplotlib, Sharpe des 4 sous-périodes

Le tearsheet utilise 256 jours par an, comme le reste du framework.
Le benchmark affiché est notre série SPY, pas un téléchargement externe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from systrade.backtest.metrics import ANN_FACTOR, drawdown_series, equity_curve, rolling_sharpe
from systrade.data.store import PROJECT_ROOT

CLASS_LABELS: dict[str, str] = {
    "equities_us": "Actions US",
    "equities_intl": "Actions internationales",
    "bonds": "Obligations",
    "commodities": "Matières premières",
    "fx": "Devises",
}

CURVE_LABELS: dict[str, str] = {
    "strategy": "Stratégie",
    "SPY": "SPY",
    "60/40": "60/40",
    "long-only vol target": "Long-only, même vol target",
}


def write_report(result: dict, output_dir: Path | None = None) -> dict[str, Path]:
    """Écrit les trois familles de graphiques et retourne leurs chemins."""
    out = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)

    curves: dict[str, pd.Series] = result["curves"]
    strategy = curves["strategy"].dropna()
    paths = {
        "equity_drawdown": _matplotlib_equity(strategy, curves, out / "equity_drawdown.png"),
        "dashboard": _plotly_dashboard(result, out / "dashboard.html"),
        "tearsheet": _quantstats_tearsheet(strategy, curves.get("SPY"), out / "tearsheet.html"),
    }
    if "walk_forward" in result and len(result["walk_forward"]):
        paths["walk_forward"] = _matplotlib_walk_forward(
            result["walk_forward"], out / "walk_forward.png"
        )
    return paths


def _matplotlib_equity(
    strategy: pd.Series,
    curves: dict[str, pd.Series],
    path: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[2, 1])
    for key, series in curves.items():
        equity = equity_curve(series.dropna())
        axes[0].plot(equity.index, equity.values, label=CURVE_LABELS.get(key, key), linewidth=1.4)
    axes[0].set_ylabel("Capital (base 1)")
    axes[0].set_title("Equity et drawdown depuis 2009")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.25)

    dd = drawdown_series(strategy)
    axes[1].fill_between(dd.index, dd.values, 0, color="#9a3412", alpha=0.8)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _matplotlib_walk_forward(table: pd.DataFrame, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(table.index)
    values = table["sharpe"].to_numpy()
    colors = ["#1d4ed8" if v >= 0 else "#9a3412" for v in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Sharpe")
    ax.set_title("Walk-forward, sans réestimation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plotly_dashboard(result: dict, path: Path) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    curves: dict[str, pd.Series] = result["curves"]
    strategy = curves["strategy"].dropna()
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Equity (base 1)",
            "Drawdown de la stratégie",
            "Sharpe glissant, 3 ans",
            "Contribution cumulée par classe",
        ),
    )
    for key, series in curves.items():
        equity = equity_curve(series.dropna())
        fig.add_trace(
            go.Scatter(x=equity.index, y=equity.values, name=CURVE_LABELS.get(key, key), mode="lines"),
            row=1, col=1,
        )

    dd = drawdown_series(strategy)
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd.values, name="Drawdown", fill="tozeroy", line_color="#9a3412"),
        row=2, col=1,
    )

    roll = rolling_sharpe(strategy)
    fig.add_trace(
        go.Scatter(x=roll.index, y=roll.values, name="Sharpe 3 ans", line_color="#1d4ed8"),
        row=3, col=1,
    )

    classes = result.get("class_returns")
    if classes is not None and len(classes.columns):
        cumulative = classes.fillna(0.0).cumsum()
        for col in cumulative.columns:
            fig.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative[col].values,
                    name=CLASS_LABELS.get(col, col),
                    stackgroup="classes",
                    mode="lines",
                ),
                row=4, col=1,
            )

    fig.update_layout(
        height=1100,
        title="Systematic Trading — backtest depuis 2009",
        legend_tracegroupgap=8,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(path, include_plotlyjs=True)
    return path


def _quantstats_tearsheet(
    strategy: pd.Series,
    benchmark: pd.Series | None,
    path: Path,
) -> Path:
    import quantstats as qs

    series = strategy.dropna().copy()
    series.name = "Strategie"
    bench = None
    if benchmark is not None:
        bench = benchmark.reindex(series.index).dropna()
        common = series.index.intersection(bench.index)
        series = series.loc[common]
        bench = bench.loc[common]
        bench.name = "SPY"

    qs.reports.html(
        series,
        benchmark=bench,
        rf=0.0,
        title="Systematic Trading",
        output=str(path),
        compounded=True,
        periods_per_year=ANN_FACTOR,
        strategy_title="Strategie",
        benchmark_title="SPY",
    )
    return path
