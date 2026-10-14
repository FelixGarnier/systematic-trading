"""
tests/test_metrics.py — métriques du bloc 9.
"""

import numpy as np
import pandas as pd
import pytest

from systrade.backtest.metrics import (
    ANN_FACTOR,
    WALK_FORWARD,
    annual_volatility,
    cagr,
    calmar_ratio,
    equity_curve,
    hit_rate,
    max_drawdown,
    rolling_sharpe,
    sharpe_ratio,
    skewness,
    summarize,
    walk_forward,
)


def test_sharpe_matches_definition():
    """√256 × μ / σ, avec l'écart-type échantillon."""
    returns = pd.Series([0.01, 0.03])
    sigma = float(np.std([0.01, 0.03], ddof=1))
    expected = np.sqrt(256) * 0.02 / sigma
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_zero_mean_is_zero():
    assert sharpe_ratio(pd.Series([0.01, -0.01, 0.02, -0.02])) == pytest.approx(0.0)


def test_sharpe_undefined_when_flat():
    assert np.isnan(sharpe_ratio(pd.Series([0.01, 0.01, 0.01])))


def test_max_drawdown_from_peak():
    """+10 % puis retour à 0,90 : drawdown = 0,90/1,10 − 1."""
    returns = pd.Series([0.10, 0.90 / 1.10 - 1.0])
    assert max_drawdown(returns) == pytest.approx(0.90 / 1.10 - 1.0)
    assert equity_curve(returns).iloc[-1] == pytest.approx(0.90)


def test_max_drawdown_is_zero_without_loss():
    assert max_drawdown(pd.Series([0.01, 0.02, 0.01])) == pytest.approx(0.0)


def test_cagr_over_one_year_of_256_days():
    daily = 1.10 ** (1 / 256) - 1
    returns = pd.Series([daily] * 256)
    assert cagr(returns) == pytest.approx(0.10, rel=1e-9)


def test_hit_rate_ignores_the_sign_of_zeros_as_misses():
    """Un zéro n'est pas un jour gagnant."""
    assert hit_rate(pd.Series([0.01, -0.02, 0.0, 0.03])) == pytest.approx(0.5)


def test_calmar_is_sharpe_over_abs_drawdown():
    returns = pd.Series([0.02, -0.01, 0.03, -0.04, 0.01])
    expected = sharpe_ratio(returns) / abs(max_drawdown(returns))
    assert calmar_ratio(returns) == pytest.approx(expected)


def test_skewness_of_a_right_tail():
    """Une grosse hausse isolée donne une skewness positive."""
    symmetric = pd.Series([-0.01, 0.01] * 20)
    tailed = pd.concat([symmetric, pd.Series([0.20])])
    assert skewness(tailed) > skewness(symmetric)


def test_summarize_contains_primary_metrics():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0004, 0.01, 300))
    stats = summarize(returns, leverage=pd.Series([1.2, 1.4]), turnover_annual=8.0)
    for key in ("sharpe", "cagr", "volatility", "max_drawdown", "calmar", "skewness", "hit_rate"):
        assert key in stats
        assert np.isfinite(stats[key])
    assert stats["leverage_mean"] == pytest.approx(1.3)
    assert stats["turnover_annual"] == pytest.approx(8.0)
    assert annual_volatility(returns) == pytest.approx(stats["volatility"])


def test_rolling_sharpe_needs_a_full_window():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0004, 0.01, 40))
    rolled = rolling_sharpe(returns, window=30)
    assert rolled.iloc[:29].isna().all()
    assert np.isfinite(rolled.iloc[-1])
    full = sharpe_ratio(returns.iloc[-30:])
    manual = np.sqrt(ANN_FACTOR) * returns.iloc[-30:].mean() / returns.iloc[-30:].std(ddof=1)
    assert full == pytest.approx(manual)
    assert rolled.iloc[-1] == pytest.approx(full)


def test_walk_forward_slices_four_periods_without_overlap():
    dates = pd.bdate_range("2009-01-01", "2024-06-28")
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0003, 0.01, len(dates)), index=dates)
    table = walk_forward(returns)
    assert list(table.index) == [name for name, _, _ in WALK_FORWARD]
    covered = sum(table["n_days"])
    # Les quatre fenêtres se suivent sans se recouvrir.
    assert covered == sum(
        returns.loc[start:end].dropna().shape[0]
        for _, start, end in WALK_FORWARD
    )
    assert table["sharpe"].notna().all()
