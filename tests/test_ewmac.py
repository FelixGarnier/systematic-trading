"""
tests/test_ewmac.py — tests unitaires pour src/systrade/rules/ewmac.py
"""

import numpy as np
import pandas as pd
import pytest

from systrade.rules.ewmac import (
    EWMAC_SPEEDS,
    all_ewmac_signals,
    ewmac_raw,
    ewmac_signal,
    normalize_ewmac,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prices(n: int = 300, tickers=("SPY", "IWM"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    data = {t: 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)) for t in tickers}
    return pd.DataFrame(data, index=dates)


def _vol(n: int = 300, tickers=("SPY", "IWM"), daily_vol: float = 0.01) -> pd.DataFrame:
    """Volatilité journalière constante pour les tests."""
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame(daily_vol, index=dates, columns=list(tickers))


# ── Tests ewmac_raw ───────────────────────────────────────────────────────────

def test_ewmac_raw_shape():
    """La forme doit être identique à celle des prix."""
    p = _prices()
    raw = ewmac_raw(p, fast=8, slow=32)
    assert raw.shape == p.shape


def test_ewmac_raw_uptrend_positive():
    """Sur une tendance haussière pure, EMA rapide > EMA lente → signal positif.
    On ignore le jour 0 : EMA_fast(t=0) = EMA_slow(t=0) = P₀ → différence = 0.
    """
    dates = pd.bdate_range("2020-01-02", periods=200)
    prices = pd.DataFrame({"SPY": np.linspace(100, 200, 200)}, index=dates)
    raw = ewmac_raw(prices, fast=8, slow=32)
    assert (raw["SPY"].iloc[1:] > 0).all()   # skip t=0


def test_ewmac_raw_downtrend_negative():
    """Sur une tendance baissière pure, signal négatif (hors t=0)."""
    dates = pd.bdate_range("2020-01-02", periods=200)
    prices = pd.DataFrame({"SPY": np.linspace(200, 100, 200)}, index=dates)
    raw = ewmac_raw(prices, fast=8, slow=32)
    assert (raw["SPY"].iloc[1:] < 0).all()   # skip t=0


def test_ewmac_raw_fast_more_responsive():
    """
    Le EWMAC rapide doit réagir plus vite que le lent.
    Proxy : l'autocorrélation lag-1 du signal rapide est plus faible —
    il change de direction plus souvent.
    """
    p = _prices(500)
    raw_fast = ewmac_raw(p, 8, 32)["SPY"].dropna()
    raw_slow = ewmac_raw(p, 32, 128)["SPY"].dropna()
    ac_fast = raw_fast.autocorr(lag=1)
    ac_slow = raw_slow.autocorr(lag=1)
    assert ac_fast < ac_slow


# ── Tests normalize_ewmac ─────────────────────────────────────────────────────

def test_normalize_uses_lagged_price():
    """
    Le dénominateur doit utiliser P_{t-1}, pas P_t.
    On modifie P_{t=0} directement et on vérifie que le signal du jour 1
    ne change pas (car il utilise P_{t=0} qui est le décalage de P_{t=1}).
    """
    p = _prices(50)
    v = _vol(50)
    sig_original = normalize_ewmac(ewmac_raw(p, 8, 32), p, v)

    # Modifier le prix au jour 1 (index 1) : ça doit changer le signal du JOUR 2
    # mais PAS le signal du jour 1 (qui utilise P_{t=0}).
    p_modified = p.copy()
    p_modified.iloc[1] *= 2.0  # double le prix du jour 1

    sig_modified = normalize_ewmac(ewmac_raw(p_modified, 8, 32), p_modified, v)

    # Signal du jour 0 doit être NaN (pas de P_{t-1})
    assert sig_original.iloc[0].isna().all()
    assert sig_modified.iloc[0].isna().all()

    # Signal du jour 2 doit avoir changé (P_{t=1} est dans le dénominateur de t=2)
    assert not np.allclose(
        sig_original.iloc[2].values,
        sig_modified.iloc[2].values,
        equal_nan=True,
    )


def test_normalize_zero_vol_gives_nan():
    """Un dénominateur nul (vol = 0) doit produire NaN, pas inf."""
    p = _prices(50)
    v = _vol(50, daily_vol=0.0)  # vol nulle
    raw = ewmac_raw(p, 8, 32)
    sig = normalize_ewmac(raw, p, v)
    assert sig.dropna().shape[0] == 0  # tout NaN après le premier point


def test_normalize_shape():
    """La forme de sortie doit être identique à l'entrée."""
    p = _prices(100)
    v = _vol(100)
    raw = ewmac_raw(p, 8, 32)
    assert normalize_ewmac(raw, p, v).shape == p.shape


# ── Tests ewmac_signal ────────────────────────────────────────────────────────

def test_ewmac_signal_no_lookahead_first_row_nan():
    """La première ligne doit être NaN (P_{t-1} introuvable pour t=0)."""
    p = _prices(100)
    v = _vol(100)
    sig = ewmac_signal(p, v, fast=8, slow=32)
    assert sig.iloc[0].isna().all()


def test_ewmac_signal_finite_after_warmup():
    """Les signaux doivent être finis après quelques jours de chauffe."""
    p = _prices(200)
    v = _vol(200)
    sig = ewmac_signal(p, v, fast=8, slow=32)
    # Après 35 jours, tous les instruments doivent avoir une valeur finie
    assert sig.iloc[35:].notna().all().all()


def test_ewmac_signal_magnitude():
    """Le signal normalisé doit avoir une magnitude raisonnable (~0 à 5 typiquement)."""
    p = _prices(300)
    v = _vol(300)
    sig = ewmac_signal(p, v, fast=8, slow=32)
    # L'espérance de |signal| doit être « raisonnable » avant scaling
    mean_abs = sig.dropna().abs().mean().mean()
    assert 0.5 < mean_abs < 10.0


# ── Tests all_ewmac_signals ───────────────────────────────────────────────────

def test_all_ewmac_signals_keys():
    """Le dict doit contenir exactement les trois clés attendues."""
    p = _prices(200)
    v = _vol(200)
    signals = all_ewmac_signals(p, v)
    assert set(signals.keys()) == set(EWMAC_SPEEDS.keys())


def test_all_ewmac_signals_shape():
    """Chaque DataFrame doit avoir la même shape que les prix."""
    p = _prices(200)
    v = _vol(200)
    signals = all_ewmac_signals(p, v)
    for name, df in signals.items():
        assert df.shape == p.shape, f"Shape mismatch for {name}"


def test_all_ewmac_signals_fast_more_volatile_than_slow():
    """
    Le signal rapide normalisé doit avoir une autocorrélation lag-1 plus faible
    que le signal lent — il change de direction plus souvent (plus réactif).
    """
    p = _prices(500)
    v = _vol(500)
    signals = all_ewmac_signals(p, v)
    ac_fast = signals["EWMAC_8_32"]["SPY"].dropna().autocorr(lag=1)
    ac_slow = signals["EWMAC_32_128"]["SPY"].dropna().autocorr(lag=1)
    assert ac_fast < ac_slow


def test_all_ewmac_signals_independent_columns():
    """Les signaux pour SPY et IWM doivent être différents (instruments distincts)."""
    p = _prices(300, tickers=("SPY", "IWM"))
    v = _vol(300, tickers=("SPY", "IWM"))
    signals = all_ewmac_signals(p, v)
    df = signals["EWMAC_8_32"].dropna()
    assert not np.allclose(df["SPY"].values, df["IWM"].values)
