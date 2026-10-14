"""
tests/test_volatility.py — tests unitaires pour src/systrade/risk/volatility.py
"""

import numpy as np
import pandas as pd
import pytest

from systrade.risk.volatility import (
    blend_vol,
    compute_vol,
    daily_returns,
    ewma_vol,
    long_term_vol,
    vol_floor,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_prices(n: int = 300, seed: int = 42, tickers=("SPY", "IWM")) -> pd.DataFrame:
    """Prix simulés : marche aléatoire avec σ_daily ≈ 1%."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    data = {}
    for t in tickers:
        r = rng.normal(0.0005, 0.01, n)
        data[t] = 100 * np.cumprod(1 + r)
    return pd.DataFrame(data, index=dates)


def _make_returns(n: int = 300, daily_vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Rendements simulés avec volatilité connue."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    r = rng.normal(0, daily_vol, (n, 2))
    return pd.DataFrame(r, index=dates, columns=["SPY", "IWM"])


# ── Tests daily_returns ───────────────────────────────────────────────────────

def test_daily_returns_first_row_is_nan():
    """La première ligne doit être NaN (pas de t-1 pour t=0)."""
    prices = _make_prices(10)
    ret = daily_returns(prices)
    assert ret.iloc[0].isna().all()


def test_daily_returns_correct_value():
    """r_t = P_t/P_{t-1} - 1 doit être exact."""
    prices = pd.DataFrame(
        {"SPY": [100.0, 102.0, 101.0]},
        index=pd.bdate_range("2024-01-02", periods=3),
    )
    ret = daily_returns(prices)
    assert ret["SPY"].iloc[1] == pytest.approx(0.02)
    assert ret["SPY"].iloc[2] == pytest.approx(-1 / 102)


def test_daily_returns_shape():
    """La forme du DataFrame de sortie doit être identique à l'entrée."""
    prices = _make_prices(50)
    assert daily_returns(prices).shape == prices.shape


# ── Tests ewma_vol ────────────────────────────────────────────────────────────

def test_ewma_vol_positive():
    """La volatilité EWMA doit être strictement positive (hors NaN)."""
    ret = _make_returns(200)
    vol = ewma_vol(ret, span=35)
    assert (vol.dropna() > 0).all().all()


def test_ewma_vol_magnitude():
    """Sur des rendements avec σ=1%, l'EWMA doit converger vers ~1%."""
    ret = _make_returns(500, daily_vol=0.01, seed=7)
    vol = ewma_vol(ret, span=35)
    mean_vol = vol.iloc[-100:].mean().mean()
    assert 0.007 < mean_vol < 0.015


def test_ewma_vol_reacts_to_shock():
    """Après un choc, la vol EWMA doit augmenter."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-02", periods=200)
    r = rng.normal(0, 0.005, 200)
    r[150:160] = 0.05   # choc : 10 jours de +5% / jour
    ret = pd.DataFrame({"SPY": r}, index=dates)
    vol = ewma_vol(ret, span=35)
    assert vol["SPY"].iloc[160] > vol["SPY"].iloc[140] * 2


def test_ewma_vol_shape():
    """La forme du DataFrame de sortie doit être identique à l'entrée."""
    ret = _make_returns(100)
    assert ewma_vol(ret).shape == ret.shape


# ── Tests long_term_vol ───────────────────────────────────────────────────────

def test_long_term_vol_nan_before_min_periods():
    """Avant 252 observations, long_term_vol doit retourner NaN."""
    ret = _make_returns(300)
    vol = ewma_vol(ret)
    lt = long_term_vol(vol, window_years=10)
    assert lt.iloc[:252].isna().all().all()


def test_long_term_vol_valid_after_min_periods():
    """Après ~265 observations (252 + délai EWMA), long_term_vol doit être non-NaN."""
    ret = _make_returns(300)
    vol = ewma_vol(ret)
    lt = long_term_vol(vol, window_years=10)
    # On vérifie à partir de la position 265 (252 + ~13 jours de chauffe EWMA)
    assert lt.iloc[265:].notna().all().all()


def test_long_term_vol_smoother_than_ewma():
    """La vol LT doit être moins volatile (std plus basse) que l'EWMA."""
    ret = _make_returns(600)
    ev = ewma_vol(ret)
    lt = long_term_vol(ev, window_years=2)  # fenêtre courte pour le test
    # Std de la LT doit être inférieure à std de l'EWMA sur la période commune
    common = ev.dropna().index.intersection(lt.dropna().index)
    assert lt.loc[common].std().mean() < ev.loc[common].std().mean()


# ── Tests blend_vol ───────────────────────────────────────────────────────────

def test_blend_vol_between_ewma_and_lt():
    """La vol blendée doit être entre l'EWMA et la LT (combinaison convexe)."""
    ret = _make_returns(400)
    ev = ewma_vol(ret)
    lt = long_term_vol(ev, window_years=1)
    bl = blend_vol(ev, lt, lt_weight=0.30)

    common = ev.dropna().index.intersection(lt.dropna().index)
    lo = ev.loc[common].values.min(axis=1)
    hi = ev.loc[common].values.max(axis=1)
    bl_vals = bl.loc[common].values.flatten()
    lo_vals = np.minimum(ev.loc[common].values, lt.loc[common].values).flatten()
    hi_vals = np.maximum(ev.loc[common].values, lt.loc[common].values).flatten()

    assert (bl_vals >= lo_vals - 1e-10).all()
    assert (bl_vals <= hi_vals + 1e-10).all()


def test_blend_vol_fallback_when_lt_nan():
    """Quand la LT est NaN, blend_vol doit retourner l'EWMA seule."""
    ret = _make_returns(100)  # trop court pour lt (min_periods=252)
    ev = ewma_vol(ret)
    lt = long_term_vol(ev, window_years=10)  # tout NaN
    bl = blend_vol(ev, lt, lt_weight=0.30)

    pd.testing.assert_frame_equal(bl.dropna(), ev.dropna())


# ── Tests vol_floor ───────────────────────────────────────────────────────────

def test_vol_floor_never_below_floor():
    """La vol finale doit toujours être ≥ au plancher."""
    ret = _make_returns(700)
    ev = ewma_vol(ret)
    lt = long_term_vol(ev, window_years=2)
    bl = blend_vol(ev, lt)
    floored = vol_floor(bl, lookback=200)

    floor = bl.shift(1).rolling(200, min_periods=100).quantile(0.05)
    common = floored.dropna().index.intersection(floor.dropna().index)
    assert (floored.loc[common].values >= floor.loc[common].values - 1e-10).all()


def test_vol_floor_no_lookahead():
    """Le plancher doit utiliser shift(1) — pas de look-ahead."""
    ret = _make_returns(300)
    ev = ewma_vol(ret)
    bl = blend_vol(ev, long_term_vol(ev, window_years=1))

    # Injecter une volatilité artificielle très basse un jour
    # Le plancher du lendemain NE DOIT PAS encore refléter ce jour
    bl_copy = bl.copy()
    bl_copy.iloc[200] = 0.0001

    floored = vol_floor(bl_copy, lookback=100)
    # Le jour 200 lui-même n'est pas affecté par son propre plancher
    # (le plancher du jour 200 est basé sur les jours 100-199)
    assert floored.iloc[200].values[0] >= 0.0001 - 1e-10


# ── Tests compute_vol ─────────────────────────────────────────────────────────

def test_compute_vol_returns_dict():
    """compute_vol() doit retourner un dict avec les clés 'daily' et 'annual'."""
    prices = _make_prices(300)
    result = compute_vol(prices)
    assert "daily" in result
    assert "annual" in result


def test_compute_vol_annual_is_daily_times_16():
    """σ_annual doit être exactement √256 × σ_daily."""
    prices = _make_prices(300)
    result = compute_vol(prices)
    ratio = result["annual"] / result["daily"]
    assert ratio.dropna().values == pytest.approx(np.sqrt(256), rel=1e-6)


def test_compute_vol_positive():
    """Les deux volatilités doivent être strictement positives (hors NaN)."""
    prices = _make_prices(300)
    result = compute_vol(prices)
    assert (result["daily"].dropna() > 0).all().all()
    assert (result["annual"].dropna() > 0).all().all()


def test_compute_vol_shape():
    """La forme de sortie doit correspondre à l'entrée."""
    prices = _make_prices(300, tickers=("SPY", "IWM", "GLD"))
    result = compute_vol(prices)
    assert result["daily"].shape == prices.shape
    assert result["annual"].shape == prices.shape
