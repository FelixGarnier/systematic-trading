"""
tests/test_costs.py — tests unitaires pour src/systrade/portfolio/costs.py
"""

import pandas as pd
import pytest

from systrade.portfolio.costs import (
    ANN_FACTOR,
    DEFAULT_SPREAD,
    SLIPPAGE_MULTIPLIER,
    annual_turnover,
    borrow_cost,
    borrow_rates,
    daily_turnover,
    sharpe_cost,
    trade_cost,
    within_speed_limit,
)

_DATES = pd.bdate_range("2024-01-02", periods=3)


def test_borrow_rates_match_architecture():
    """Les taux lus dans la config sont ceux validés au bloc 8."""
    rates = borrow_rates()
    assert rates["SPY"] == pytest.approx(0.003)
    assert rates["IWM"] == pytest.approx(0.003)
    assert rates["VGK"] == pytest.approx(0.005)
    assert rates["EEM"] == pytest.approx(0.015)
    assert rates["EPP"] == pytest.approx(0.015)
    assert rates["TIP"] == pytest.approx(0.003)
    assert rates["GLD"] == pytest.approx(0.005)
    assert rates["DBC"] == pytest.approx(0.008)
    assert rates["UUP"] == pytest.approx(0.005)
    assert len(rates) == 14


def test_borrow_cost_only_on_shorts():
    """Un long ne paie pas de borrow. Un short paie taux × notionnel / 256."""
    positions = pd.DataFrame({"SPY": [10.0, -10.0]}, index=_DATES[:2])
    prices = pd.DataFrame({"SPY": [100.0, 100.0]}, index=_DATES[:2])
    cost = borrow_cost(positions, prices, rates={"SPY": 0.003})
    assert cost["SPY"].iloc[0] == pytest.approx(0.0)
    assert cost["SPY"].iloc[1] == pytest.approx(10.0 * 100.0 * 0.003 / ANN_FACTOR)


def test_borrow_cost_uses_each_instrument_rate():
    positions = pd.DataFrame({"SPY": [-5.0], "EEM": [-5.0]}, index=_DATES[:1])
    prices = pd.DataFrame({"SPY": [100.0], "EEM": [40.0]}, index=_DATES[:1])
    rates = {"SPY": 0.003, "EEM": 0.015}
    cost = borrow_cost(positions, prices, rates=rates)
    assert cost["SPY"].iloc[0] == pytest.approx(5 * 100 * 0.003 / ANN_FACTOR)
    assert cost["EEM"].iloc[0] == pytest.approx(5 * 40 * 0.015 / ANN_FACTOR)


def test_trade_cost_spread_and_slippage():
    """coût = |ΔN| × prix × spread × slippage."""
    trades = pd.DataFrame({"SPY": [10.0, -4.0]}, index=_DATES[:2])
    prices = pd.DataFrame({"SPY": [100.0, 100.0]}, index=_DATES[:2])
    cost = trade_cost(trades, prices, spread=0.0002, slippage=1.5, commission=0.0)
    assert cost["SPY"].iloc[0] == pytest.approx(10 * 100 * 0.0002 * 1.5)
    assert cost["SPY"].iloc[1] == pytest.approx(4 * 100 * 0.0002 * 1.5)


def test_trade_cost_adds_commission_only_when_trading():
    trades = pd.DataFrame({"SPY": [10.0, 0.0]}, index=_DATES[:2])
    prices = pd.DataFrame({"SPY": [100.0, 100.0]}, index=_DATES[:2])
    cost = trade_cost(trades, prices, spread=0.0, slippage=1.0, commission=1.25)
    assert cost["SPY"].iloc[0] == pytest.approx(1.25)
    assert cost["SPY"].iloc[1] == pytest.approx(0.0)


def test_daily_turnover_is_traded_notional_over_capital():
    trades = pd.DataFrame({"SPY": [10.0], "IEF": [-5.0]}, index=_DATES[:1])
    prices = pd.DataFrame({"SPY": [100.0], "IEF": [80.0]}, index=_DATES[:1])
    # |10|×100 + |-5|×80 = 1 400, capital 10 000 → 0.14
    turnover = daily_turnover(trades, prices, capital=10_000.0)
    assert turnover.iloc[0] == pytest.approx(0.14)


def test_annual_turnover_scales_by_256():
    daily = pd.Series([0.01, 0.03], index=_DATES[:2])
    assert annual_turnover(daily) == pytest.approx(0.02 * ANN_FACTOR)


def test_sharpe_cost_formula():
    """SR_coût = spread × turnover annuel / σ."""
    got = sharpe_cost(spread=0.0002, turnover_annual=10.0, vol_annual=0.13)
    assert got == pytest.approx(0.0002 * 10.0 / 0.13)


def test_sharpe_cost_rejects_non_positive_vol():
    with pytest.raises(ValueError):
        sharpe_cost(0.0002, 10.0, vol_annual=0.0)


def test_speed_limit_accepts_cheap_rule():
    cost = sharpe_cost(DEFAULT_SPREAD, turnover_annual=5.0, vol_annual=0.13)
    assert within_speed_limit(cost, expected_sharpe=0.5)


def test_speed_limit_rejects_expensive_rule():
    # Turnover absurde : le coût dépasse 1/3 d'un Sharpe de 0,3.
    cost = sharpe_cost(DEFAULT_SPREAD, turnover_annual=20_000.0, vol_annual=0.13)
    assert not within_speed_limit(cost, expected_sharpe=0.3)


def test_default_slippage_is_backtest_multiplier():
    assert SLIPPAGE_MULTIPLIER == pytest.approx(1.5)
