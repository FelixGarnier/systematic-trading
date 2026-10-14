"""
tests/test_carry.py — tests unitaires pour src/systrade/rules/carry.py
"""

import numpy as np
import pandas as pd
import pytest

from systrade.rules.carry import (
    BOND_CARRY_MAP,
    CARRY_SPANS,
    EQUITY_CARRY_TICKERS,
    all_carry_signals,
    bond_carry_raw,
    carry_raw,
    equity_carry_raw,
    normalize_carry,
    smooth_carry,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_EQUITY_TICKERS = ["SPY", "IWM", "VGK"]
_BOND_MAP = {"IEF": "DGS7", "TLT": "DGS20"}
_ALL_TICKERS = _EQUITY_TICKERS + list(_BOND_MAP.keys())

N = 300
_DATES = pd.bdate_range("2020-01-02", periods=N)


def _prices_unadj(tickers=_EQUITY_TICKERS) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {t: 100 + rng.normal(0, 1, N).cumsum() for t in tickers}
    return pd.DataFrame(data, index=_DATES).clip(lower=1)


def _dividends(tickers=_EQUITY_TICKERS, annual_yield: float = 0.02) -> pd.DataFrame:
    """Dividende trimestriel simulé (~0.5% du prix par trimestre)."""
    price = _prices_unadj(tickers)
    div = pd.DataFrame(0.0, index=_DATES, columns=tickers)
    # Un versement tous les ~63 jours
    for i in range(0, N, 63):
        div.iloc[i] = price.iloc[i] * (annual_yield / 4)
    return div


def _rates() -> pd.DataFrame:
    """Taux FRED simulés (en %)."""
    rng = np.random.default_rng(2)
    dtb3 = 2.0 + rng.normal(0, 0.05, N).cumsum().clip(-2, 8)
    dgs7  = dtb3 + 1.5 + rng.normal(0, 0.1, N)
    dgs20 = dtb3 + 2.0 + rng.normal(0, 0.1, N)
    baml  = dtb3 + 3.0 + rng.normal(0, 0.2, N)
    return pd.DataFrame(
        {"DTB3": dtb3, "DGS7": dgs7, "DGS20": dgs20, "BAMLC0A0CMEY": baml},
        index=_DATES,
    )


def _vol_annual(tickers=_ALL_TICKERS, val: float = 0.15) -> pd.DataFrame:
    return pd.DataFrame(val, index=_DATES, columns=tickers)


# ── Tests equity_carry_raw ────────────────────────────────────────────────────

def test_equity_carry_shape():
    """La forme doit correspondre aux tickers equity présents dans dividends."""
    div = _dividends()
    pu  = _prices_unadj()
    cash = _rates()["DTB3"] / 100
    carry = equity_carry_raw(div, pu, cash)
    assert carry.shape == (N, len(_EQUITY_TICKERS))


def test_equity_carry_columns():
    """Les colonnes doivent être les tickers equity, pas plus."""
    div = _dividends()
    pu  = _prices_unadj()
    cash = _rates()["DTB3"] / 100
    carry = equity_carry_raw(div, pu, cash)
    assert set(carry.columns) == set(_EQUITY_TICKERS)


def test_equity_carry_positive_yield():
    """Avec un rendement de dividendes supérieur au taux cash, carry > 0."""
    # Prix stable = 100, dividende TTM = 4 (4%), cash rate = 2% → carry ≈ +2%
    div = pd.DataFrame({"SPY": [1.0] + [0.0] * (N - 1)}, index=_DATES)  # 1 div en t=0
    # sum 252j = 1.0 → yield = 1/100 = 1%
    pu  = pd.DataFrame({"SPY": [100.0] * N}, index=_DATES)
    cash = pd.Series(0.005, index=_DATES)  # 0.5%
    carry = equity_carry_raw(div, pu, cash)
    # Après le 1er jour, TTM = 1$ → yield = 1% → carry = 1% - 0.5% = 0.5%
    assert carry["SPY"].iloc[1] == pytest.approx(0.005, abs=1e-6)


def test_equity_carry_no_dividends_negative():
    """Sans dividendes, le carry doit être négatif (yield = 0 < cash rate)."""
    div = pd.DataFrame({"SPY": [0.0] * N}, index=_DATES)
    pu  = pd.DataFrame({"SPY": [100.0] * N}, index=_DATES)
    cash = pd.Series(0.03, index=_DATES)  # cash rate = 3%
    carry = equity_carry_raw(div, pu, cash)
    assert (carry["SPY"] < 0).all()


def test_equity_carry_empty_if_no_matching_tickers():
    """Retourne un DataFrame vide si aucun ticker equity n'est présent."""
    div = pd.DataFrame({"IEF": [0.0] * N}, index=_DATES)
    pu  = pd.DataFrame({"IEF": [100.0] * N}, index=_DATES)
    cash = pd.Series(0.02, index=_DATES)
    carry = equity_carry_raw(div, pu, cash)
    assert carry.empty


# ── Tests bond_carry_raw ──────────────────────────────────────────────────────

def test_bond_carry_shape():
    """La forme doit correspondre au nombre d'obligations dans le mapping."""
    carry = bond_carry_raw(_rates(), _BOND_MAP)
    assert carry.shape == (N, len(_BOND_MAP))


def test_bond_carry_columns():
    """Les colonnes doivent être les tickers obligataires."""
    carry = bond_carry_raw(_rates(), _BOND_MAP)
    assert set(carry.columns) == set(_BOND_MAP.keys())


def test_bond_carry_positive_term_premium():
    """DGS7 > DTB3 → carry IEF > 0."""
    rates = _rates()  # DGS7 = DTB3 + 1.5% par construction
    carry = bond_carry_raw(rates, {"IEF": "DGS7"})
    assert (carry["IEF"].dropna() > 0).all()


def test_bond_carry_in_decimal():
    """Le carry doit être exprimé en décimal, pas en pourcentage."""
    rates = pd.DataFrame(
        {"DTB3": [2.0], "DGS7": [4.0]},
        index=pd.bdate_range("2024-01-02", periods=1),
    )
    carry = bond_carry_raw(rates, {"IEF": "DGS7"})
    assert carry["IEF"].iloc[0] == pytest.approx(0.02)  # (4-2)/100


def test_bond_carry_missing_fred_col_skipped():
    """Un instrument dont la colonne FRED est absente doit être ignoré."""
    rates = _rates().drop(columns=["DGS7"])
    carry = bond_carry_raw(rates, _BOND_MAP)
    assert "IEF" not in carry.columns
    assert "TLT" in carry.columns


# ── Tests carry_raw ───────────────────────────────────────────────────────────

def test_carry_raw_combines_equity_and_bond():
    """carry_raw doit retourner les colonnes equity ET bond."""
    div = _dividends()
    pu  = _prices_unadj()
    rates = _rates()
    combined = carry_raw(div, pu, rates, bond_carry_map=_BOND_MAP)
    assert set(_EQUITY_TICKERS).issubset(combined.columns)
    assert set(_BOND_MAP.keys()).issubset(combined.columns)


def test_carry_raw_index_aligned_to_dividends():
    """L'index de sortie doit correspondre à l'index des dividendes."""
    div = _dividends()
    pu  = _prices_unadj()
    combined = carry_raw(div, pu, _rates(), bond_carry_map=_BOND_MAP)
    pd.testing.assert_index_equal(combined.index, div.index)


# ── Tests normalize_carry ─────────────────────────────────────────────────────

def test_normalize_carry_correct_value():
    """carry_normalisé = carry_brut / vol_annual."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    brut = pd.DataFrame({"SPY": [0.02] * 5}, index=dates)
    vol  = pd.DataFrame({"SPY": [0.15] * 5}, index=dates)
    norm = normalize_carry(brut, vol)
    assert norm["SPY"].iloc[0] == pytest.approx(0.02 / 0.15)


def test_normalize_carry_only_common_columns():
    """Seules les colonnes présentes dans les deux DataFrames sont retournées."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    brut = pd.DataFrame({"SPY": [0.02] * 5, "IEF": [0.01] * 5}, index=dates)
    vol  = pd.DataFrame({"SPY": [0.15] * 5}, index=dates)  # IEF absent
    norm = normalize_carry(brut, vol)
    assert "IEF" not in norm.columns
    assert "SPY" in norm.columns


def test_normalize_carry_zero_vol_gives_nan():
    """Une vol nulle doit donner NaN, pas inf."""
    dates = pd.bdate_range("2024-01-02", periods=3)
    brut = pd.DataFrame({"SPY": [0.02] * 3}, index=dates)
    vol  = pd.DataFrame({"SPY": [0.0] * 3}, index=dates)
    norm = normalize_carry(brut, vol)
    assert norm["SPY"].isna().all()


# ── Tests smooth_carry ────────────────────────────────────────────────────────

def test_smooth_carry_returns_two_dataframes():
    """smooth_carry doit retourner un tuple de deux DataFrames."""
    norm = pd.DataFrame({"SPY": np.random.default_rng(3).normal(0, 1, N)}, index=_DATES)
    medium, slow = smooth_carry(norm)
    assert isinstance(medium, pd.DataFrame)
    assert isinstance(slow, pd.DataFrame)


def test_smooth_carry_slow_smoother_than_medium():
    """Le signal slow (EMA 20) doit avoir une std inférieure au signal medium (EMA 5)."""
    rng = np.random.default_rng(4)
    norm = pd.DataFrame({"SPY": rng.normal(0, 1, N)}, index=_DATES)
    medium, slow = smooth_carry(norm)
    assert slow["SPY"].std() < medium["SPY"].std()


def test_smooth_carry_same_columns():
    """Les colonnes doivent être identiques entre normalized, medium et slow."""
    norm = pd.DataFrame(
        {"SPY": [0.1] * N, "IEF": [0.05] * N}, index=_DATES
    )
    medium, slow = smooth_carry(norm)
    assert list(medium.columns) == list(norm.columns)
    assert list(slow.columns)   == list(norm.columns)


def test_smooth_carry_nan_before_min_periods():
    """Avant span_medium observations, carry_medium doit être NaN."""
    norm = pd.DataFrame({"SPY": [0.1] * N}, index=_DATES)
    medium, slow = smooth_carry(norm, span_medium=5, span_slow=20)
    assert medium["SPY"].iloc[:4].isna().all()
    assert slow["SPY"].iloc[:19].isna().all()


# ── Tests all_carry_signals ───────────────────────────────────────────────────

def test_all_carry_signals_keys():
    """Le dict doit avoir exactement les clés 'carry_medium' et 'carry_slow'."""
    div  = _dividends()
    pu   = _prices_unadj()
    rates = _rates()
    vol  = _vol_annual()
    signals = all_carry_signals(div, pu, rates, vol, bond_carry_map=_BOND_MAP)
    assert set(signals.keys()) == {"carry_medium", "carry_slow"}


def test_all_carry_signals_columns_are_eligible_instruments():
    """Les colonnes doivent être les instruments éligibles au carry."""
    div  = _dividends()
    pu   = _prices_unadj()
    rates = _rates()
    vol  = _vol_annual()
    signals = all_carry_signals(div, pu, rates, vol, bond_carry_map=_BOND_MAP)
    expected = set(_EQUITY_TICKERS) | set(_BOND_MAP.keys())
    assert set(signals["carry_medium"].columns) == expected


def test_all_carry_signals_slow_smoother_than_medium():
    """carry_slow doit toujours avoir une std inférieure à carry_medium."""
    div  = _dividends()
    pu   = _prices_unadj()
    rates = _rates()
    vol  = _vol_annual()
    signals = all_carry_signals(div, pu, rates, vol, bond_carry_map=_BOND_MAP)
    for ticker in signals["carry_medium"].columns:
        std_med  = signals["carry_medium"][ticker].std()
        std_slow = signals["carry_slow"][ticker].std()
        assert std_slow < std_med, f"carry_slow not smoother for {ticker}"
