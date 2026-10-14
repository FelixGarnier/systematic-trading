"""
tests/test_report.py — les trois sorties visuelles du bloc 9.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from systrade.backtest.report import write_report


def test_write_report_creates_the_three_families(tmp_path: Path):
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2016-01-04", periods=400)
    shock = rng.normal(0.0003, 0.01, len(idx))
    strategy = pd.Series(shock, index=idx, name="strategy")
    spy = pd.Series(shock * 0.8 + 0.0001, index=idx, name="SPY")
    mix = pd.Series(shock * 0.5, index=idx, name="60/40")
    long_only = pd.Series(shock * 0.6, index=idx, name="long-only")
    classes = pd.DataFrame(
        {
            "equities_us": shock * 0.3,
            "bonds": shock * 0.2,
        },
        index=idx,
    )
    walk = pd.DataFrame(
        {"sharpe": [0.4, 0.2, 0.6, -0.1], "cagr": [0.04, 0.02, 0.06, -0.01],
         "max_drawdown": [-0.1, -0.12, -0.08, -0.2], "n_days": [100, 100, 100, 100]},
        index=["P1", "P2", "P3", "P4"],
    )
    result = {
        "curves": {
            "strategy": strategy,
            "SPY": spy,
            "60/40": mix,
            "long-only vol target": long_only,
        },
        "class_returns": classes,
        "walk_forward": walk,
    }

    paths = write_report(result, tmp_path)

    assert paths["tearsheet"].exists() and paths["tearsheet"].stat().st_size > 1000
    assert paths["dashboard"].exists() and "plotly" in paths["dashboard"].read_text(encoding="utf-8").lower()
    assert paths["equity_drawdown"].exists() and paths["equity_drawdown"].stat().st_size > 1000
    assert paths["walk_forward"].exists()
