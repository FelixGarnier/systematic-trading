"""
tests/test_engine.py — moteur de backtest du bloc 9.
"""

import numpy as np
import pandas as pd
import pytest

from systrade.backtest.engine import (
    asset_classes,
    backtest_from_forecast,
    buy_and_hold,
    class_contributions,
    long_only_forecast,
    shares_per_capital,
    simulate,
    sixty_forty,
)
from systrade.portfolio.sizing import MAX_LEVERAGE

_DATES = pd.bdate_range("2024-01-02", periods=4)


def _market(prices: list[float], returns: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = _DATES[: len(prices)]
    px = pd.DataFrame({"SPY": prices}, index=idx)
    rets = pd.DataFrame({"SPY": returns}, index=idx)
    return px, rets


def test_pnl_uses_yesterday_position_and_unadjusted_price():
    """PnL_t = N_{t-1} × P_{t-1} × r_t. Le trade du jour t ne gagne pas r_t."""
    prices, returns = _market([100.0, 110.0], [0.0, 0.10])
    k = pd.DataFrame({"SPY": [0.0001, 0.0001]}, index=prices.index)
    result = simulate(k, prices, returns, capital0=100_000.0, delta=0.0, spread=0.0, rates={})

    assert result["positions"]["SPY"].iloc[0] == pytest.approx(10.0)
    assert result["returns"].iloc[0] == pytest.approx(0.0)
    # 10 actions × 100 $ × 10 % = 100 $ de P&L, gagnés le lendemain.
    assert result["capital"].iloc[1] == pytest.approx(100_100.0)
    assert result["returns"].iloc[1] == pytest.approx(0.001)


def test_trade_cost_is_paid_when_the_position_is_opened():
    prices, returns = _market([100.0, 100.0], [0.0, 0.0])
    k = pd.DataFrame({"SPY": [0.0001, 0.0001]}, index=prices.index)
    result = simulate(
        k, prices, returns, capital0=100_000.0, delta=0.0, spread=0.01, slippage=1.0, rates={}
    )
    # 10 actions × 100 $ × 1 % = 10 $
    assert result["trade_costs"].iloc[0] == pytest.approx(10.0)
    assert result["capital"].iloc[0] == pytest.approx(99_990.0)


def test_levered_long_pays_the_tbill_on_the_debit():
    """Le cash résiduel négatif paie le T-bill. Le jour 0 est plat, donc sans débit."""
    prices, returns = _market([100.0, 100.0], [0.0, 0.0])
    k = pd.DataFrame({"SPY": [0.02, 0.02]}, index=prices.index)
    rf = pd.Series(0.05, index=prices.index)
    result = simulate(
        k, prices, returns, capital0=100_000.0, delta=0.0, spread=0.0, rates={}, funding=rf
    )
    held = result["positions"]["SPY"].iloc[0]
    cash = result["capital_start"].iloc[1] - held * 100.0
    assert cash < 0.0
    assert result["funding"].iloc[1] == pytest.approx(cash * 0.05 / 256)


def test_short_cash_earns_the_tbill():
    prices, returns = _market([100.0, 100.0], [0.0, 0.0])
    k = pd.DataFrame({"SPY": [-0.01, -0.01]}, index=prices.index)
    rf = pd.Series(0.05, index=prices.index)
    result = simulate(
        k, prices, returns, capital0=100_000.0, delta=0.0, spread=0.0, rates={}, funding=rf
    )
    held = result["positions"]["SPY"].iloc[0]
    cash = result["capital_start"].iloc[1] - held * 100.0
    assert cash > result["capital_start"].iloc[1]
    assert result["funding"].iloc[1] == pytest.approx(cash * 0.05 / 256)
    assert result["funding"].iloc[1] > 0.0


def test_borrow_is_charged_on_the_short_held_overnight():
    prices, returns = _market([100.0, 100.0], [0.0, 0.0])
    k = pd.DataFrame({"SPY": [-0.0001, -0.0001]}, index=prices.index)
    result = simulate(
        k, prices, returns, capital0=100_000.0, delta=0.0, spread=0.0,
        rates={"SPY": 0.256},
    )
    # Jour 0 : on ouvre le short, pas encore d'emprunt.
    # Jour 1 : 10 actions × 100 $ × 0,256 / 256 = 1 $.
    assert result["borrow_costs"].iloc[0] == pytest.approx(0.0)
    assert result["borrow_costs"].iloc[1] == pytest.approx(1.0)
    assert result["capital"].iloc[1] == pytest.approx(99_999.0)


def test_leverage_cap_binds_after_the_buffer():
    """Depuis zéro, le buffer va à 90 % de la cible, puis le levier ramène à 2×."""
    idx = _DATES[:1]
    prices = pd.DataFrame({"SPY": [100.0]}, index=idx)
    returns = pd.DataFrame({"SPY": [0.0]}, index=idx)
    k = pd.DataFrame({"SPY": [0.05]}, index=idx)  # cible = 5 000 actions
    result = simulate(k, prices, returns, capital0=100_000.0, delta=0.10, spread=0.0, rates={})
    assert result["positions"]["SPY"].iloc[0] == pytest.approx(2_000.0)
    assert result["leverage"].iloc[0] == pytest.approx(MAX_LEVERAGE)


def test_future_return_does_not_change_past_positions():
    rng = np.random.default_rng(0)
    n = 15
    idx = pd.bdate_range("2024-01-02", periods=n)
    prices = pd.DataFrame({"SPY": np.full(n, 100.0)}, index=idx)
    returns = pd.DataFrame({"SPY": rng.normal(0, 0.01, n)}, index=idx)
    k = pd.DataFrame({"SPY": rng.uniform(0.001, 0.003, n)}, index=idx)

    base = simulate(k, prices, returns, capital0=100_000.0, spread=0.0, rates={})
    shocked = returns.copy()
    shocked.iloc[10, 0] = 0.25
    alt = simulate(k, prices, shocked, capital0=100_000.0, spread=0.0, rates={})
    assert base["positions"]["SPY"].iloc[9] == pytest.approx(alt["positions"]["SPY"].iloc[9])


def test_shares_per_capital_formula():
    idx = _DATES[:1]
    forecast = pd.DataFrame({"SPY": [10.0]}, index=idx)
    vol = pd.DataFrame({"SPY": [0.16]}, index=idx)
    price = pd.DataFrame({"SPY": [100.0]}, index=idx)
    weights = pd.DataFrame({"SPY": [0.10]}, index=idx)
    idm = pd.Series([2.0], index=idx)
    k = shares_per_capital(forecast, vol, price, weights, idm, vol_target=0.13, forecast_scale=10.0)
    # k = 0,13 × 10 × 0,10 × 2 / (10 × 0,16 × 100)
    assert k["SPY"].iloc[0] == pytest.approx(0.13 * 10 * 0.10 * 2 / (10 * 0.16 * 100))


def test_sixty_forty_and_buy_and_hold():
    idx = _DATES[:3]
    prices = pd.DataFrame(
        {"SPY": [100.0, 110.0, 110.0], "IEF": [100.0, 100.0, 102.0]},
        index=idx,
    )
    spy = buy_and_hold(prices, "SPY")
    mix = sixty_forty(prices)
    assert spy.iloc[1] == pytest.approx(0.10)
    assert mix.iloc[1] == pytest.approx(0.60 * 0.10 + 0.40 * 0.0)
    assert mix.iloc[2] == pytest.approx(0.40 * 0.02)


def test_long_only_forecast_is_ten_when_price_and_vol_exist():
    idx = _DATES[:2]
    prices = pd.DataFrame({"SPY": [100.0, np.nan]}, index=idx)
    vol = pd.DataFrame({"SPY": [0.16, 0.16]}, index=idx)
    forecast = long_only_forecast(prices, vol)
    assert forecast["SPY"].iloc[0] == pytest.approx(10.0)
    assert pd.isna(forecast["SPY"].iloc[1])


def test_class_contributions_sum_members():
    idx = _DATES[:2]
    pnl = pd.DataFrame({"SPY": [10.0, 0.0], "IWM": [5.0, 0.0], "GLD": [-2.0, 0.0]}, index=idx)
    capital_start = pd.Series([100.0, 100.0], index=idx)
    classes = {"SPY": "equities_us", "IWM": "equities_us", "GLD": "commodities"}
    contrib = class_contributions(pnl, capital_start, classes)
    assert contrib["equities_us"].iloc[0] == pytest.approx(0.15)
    assert contrib["commodities"].iloc[0] == pytest.approx(-0.02)


def test_asset_classes_cover_the_universe():
    classes = asset_classes()
    assert classes["SPY"] == "equities_us"
    assert classes["GLD"] == "commodities"
    assert classes["UUP"] == "fx"
    assert len(classes) == 14


def test_backtest_from_forecast_respects_leverage_and_stays_finite():
    rng = np.random.default_rng(2)
    n = 80
    idx = pd.bdate_range("2020-01-02", periods=n)
    shock = rng.normal(0.0003, 0.01, (n, 2))
    prices = pd.DataFrame(
        100 * np.cumprod(1 + shock, axis=0),
        index=idx,
        columns=["SPY", "IEF"],
    )
    forecast = pd.DataFrame(rng.normal(0, 8, (n, 2)), index=idx, columns=["SPY", "IEF"]).clip(-20, 20)
    vol_daily = pd.DataFrame(0.01, index=idx, columns=["SPY", "IEF"])
    vol_annual = pd.DataFrame(0.16, index=idx, columns=["SPY", "IEF"])
    result = backtest_from_forecast(
        forecast,
        prices,
        prices,
        vol_daily,
        vol_annual,
        weights={"SPY": 0.5, "IEF": 0.5},
        rates={"SPY": 0.003, "IEF": 0.003},
        min_obs=20,
        ewma_span=1,
        spread=0.0,
    )
    assert np.isfinite(result["returns"]).all()
    assert result["capital"].iloc[-1] > 0
    assert result["leverage"].max() <= MAX_LEVERAGE + 1e-8
