"""
tests/test_checks.py — tests unitaires pour src/systrade/data/checks.py
"""

import pandas as pd
import numpy as np
import pytest

from systrade.data.checks import (
    DataWarning,
    check_missing_days,
    check_stale_prices,
    check_extreme_returns,
    check_non_positive_prices,
    run_all_checks,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_prices(data: dict, dates=None) -> pd.DataFrame:
    """Construit un DataFrame de prix à partir d'un dict {ticker: [values]}."""
    if dates is None:
        dates = pd.bdate_range("2024-01-02", periods=len(next(iter(data.values()))))
    return pd.DataFrame(data, index=dates)


# ── Tests DataWarning ─────────────────────────────────────────────────────────

def test_datawarning_str():
    """__str__ doit afficher le check, le ticker et le nombre d'occurrences."""
    w = DataWarning(
        check="test_check",
        ticker="SPY",
        dates=[pd.Timestamp("2024-01-02")],
        message="test message",
    )
    s = str(w)
    assert "test_check" in s
    assert "SPY" in s
    assert "1 occurrence" in s


# ── Tests check_missing_days ──────────────────────────────────────────────────

def test_missing_days_clean_data():
    """Aucun avertissement sur des données sans NaN."""
    prices = _make_prices({"SPY": [470.0, 471.0, 469.0], "IWM": [200.0, 201.0, 199.0]})
    assert check_missing_days(prices) == []


def test_missing_days_pre_launch_nan():
    """Les NaN avant le premier prix valide sont signalés comme 'ETF non lancé'."""
    prices = _make_prices({"SPY": [470.0, 471.0, 469.0], "NEW": [float("nan"), 50.0, 51.0]})
    warnings = check_missing_days(prices)
    assert len(warnings) == 1
    assert warnings[0].check == "missing_days"
    assert "non lancé" in warnings[0].message


def test_missing_days_mid_series_nan():
    """Les NaN au milieu de la série sont signalés comme 'données manquantes'."""
    prices = _make_prices({"SPY": [470.0, 471.0, 469.0], "IWM": [200.0, float("nan"), 199.0]})
    warnings = check_missing_days(prices)
    assert len(warnings) == 1
    assert "manquantes en cours" in warnings[0].message


# ── Tests check_stale_prices ──────────────────────────────────────────────────

def test_stale_prices_clean_data():
    """Aucun avertissement si les prix changent chaque jour."""
    prices = _make_prices({"SPY": [470.0, 471.0, 472.0, 473.0, 474.0, 475.0]})
    assert check_stale_prices(prices, max_consecutive=5) == []


def test_stale_prices_detected():
    """5 jours consécutifs au même prix doivent déclencher un avertissement."""
    prices = _make_prices({"SPY": [470.0, 470.0, 470.0, 470.0, 470.0, 471.0]})
    warnings = check_stale_prices(prices, max_consecutive=5)
    assert len(warnings) == 1
    assert warnings[0].check == "stale_prices"
    assert warnings[0].ticker == "SPY"


def test_stale_prices_below_threshold():
    """4 jours consécutifs (< seuil de 5) ne doivent pas déclencher d'avertissement."""
    prices = _make_prices({"SPY": [470.0, 470.0, 470.0, 470.0, 471.0, 472.0]})
    assert check_stale_prices(prices, max_consecutive=5) == []


# ── Tests check_extreme_returns ───────────────────────────────────────────────

def test_extreme_returns_clean_data():
    """Aucun avertissement sur des rendements normaux."""
    rng = np.random.default_rng(42)
    # Rendements normaux (~0.5% par jour)
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.005, 100))
    df = _make_prices({"SPY": prices.tolist()})
    assert check_extreme_returns(df, threshold_sigma=8, min_periods=30) == []


def test_extreme_returns_detected():
    """Un rendement de +50% en un jour doit être détecté."""
    rng = np.random.default_rng(0)
    prices = list(100 * np.cumprod(1 + rng.normal(0, 0.005, 60)))
    # Injecter un choc de +50% au jour 50
    prices[50] = prices[49] * 1.50
    df = _make_prices({"SPY": prices})
    warnings = check_extreme_returns(df, threshold_sigma=8, min_periods=30)
    assert len(warnings) == 1
    assert warnings[0].check == "extreme_returns"


# ── Tests check_non_positive_prices ──────────────────────────────────────────

def test_non_positive_prices_clean():
    """Aucun avertissement sur des prix positifs."""
    prices = _make_prices({"SPY": [470.0, 471.0, 469.0]})
    assert check_non_positive_prices(prices) == []


def test_non_positive_price_zero():
    """Un prix de 0 doit être détecté."""
    prices = _make_prices({"SPY": [470.0, 0.0, 469.0]})
    warnings = check_non_positive_prices(prices)
    assert len(warnings) == 1
    assert warnings[0].check == "non_positive_price"


def test_non_positive_price_negative():
    """Un prix négatif doit être détecté."""
    prices = _make_prices({"SPY": [470.0, -1.0, 469.0]})
    warnings = check_non_positive_prices(prices)
    assert len(warnings) == 1


# ── Tests run_all_checks ──────────────────────────────────────────────────────

def test_run_all_checks_returns_list():
    """run_all_checks() doit toujours retourner une liste."""
    prices = _make_prices({"SPY": [470.0, 471.0, 469.0]})
    result = run_all_checks(prices, prices)
    assert isinstance(result, list)


def test_run_all_checks_clean_data():
    """run_all_checks() ne doit retourner aucun avertissement sur des données propres."""
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {"SPY": 100 * np.cumprod(1 + rng.normal(0.0005, 0.005, 60)),
         "IWM": 100 * np.cumprod(1 + rng.normal(0.0005, 0.005, 60))},
        index=pd.bdate_range("2024-01-02", periods=60),
    )
    assert run_all_checks(prices, prices) == []
