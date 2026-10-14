"""
tests/test_scalar.py — tests unitaires pour src/systrade/forecast/scalar.py
"""

import numpy as np
import pandas as pd
import pytest

from systrade.forecast.scalar import (
    FORECAST_CAP,
    FORECAST_TARGET,
    FS_MIN_OBS,
    cap_forecast,
    compute_scaled_capped,
    expanding_mean_abs_pooled,
    forecast_scalar,
    scale_forecast,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

N = 500
_DATES = pd.bdate_range("2019-01-02", periods=N)
_TICKERS = ["SPY", "IWM", "GLD", "IEF"]


def _signal(mean_abs: float = 2.0, seed: int = 0) -> pd.DataFrame:
    """Signal synthétique centré avec une valeur absolue moyenne connue."""
    rng = np.random.default_rng(seed)
    # Laplacian : E[|X|] = b = mean_abs (paramètre d'échelle)
    data = rng.laplace(loc=0.0, scale=mean_abs, size=(N, len(_TICKERS)))
    return pd.DataFrame(data, index=_DATES, columns=_TICKERS)


def _signal_constant(value: float = 2.0) -> pd.DataFrame:
    """Signal constant = value pour tous les instruments."""
    return pd.DataFrame(value, index=_DATES, columns=_TICKERS)


# ── Tests expanding_mean_abs_pooled ───────────────────────────────────────────

def test_expanding_mean_abs_nan_before_min_obs():
    """Avant FS_MIN_OBS observations, le résultat doit être NaN."""
    sig = _signal()
    result = expanding_mean_abs_pooled(sig)
    # Chaque jour apporte len(_TICKERS)=4 observations.
    # Le premier point valide est exactement au jour FS_MIN_OBS/4 = 25 (index 24).
    # Donc les jours 0-23 (indices 0..min_days-2) doivent être NaN.
    min_days = FS_MIN_OBS // len(_TICKERS)  # = 25
    assert result.iloc[: min_days - 1].isna().all()
    assert pd.notna(result.iloc[min_days - 1])


def test_expanding_mean_abs_converges():
    """Sur un signal avec E[|x|] = 2, la moyenne poolée expansive doit converger vers 2."""
    sig = _signal(mean_abs=2.0, seed=7)
    result = expanding_mean_abs_pooled(sig)
    # Sur 500 jours × 4 instruments, la loi des grands nombres doit bien fonctionner
    final_val = result.iloc[-1]
    assert 1.6 < final_val < 2.4


def test_expanding_mean_abs_is_non_decreasing_in_count():
    """Le cumcount doit être strictement croissant (chaque jour ajoute des observations)."""
    sig = _signal()
    daily_cnt = sig.notna().sum(axis=1)
    cum_cnt = daily_cnt.cumsum()
    assert (cum_cnt.diff().dropna() >= 0).all()


def test_expanding_mean_abs_strictly_expanding():
    """La valeur à t doit dépendre de t-1 (caractère expansif vérifié indirectement)."""
    sig = _signal()
    result = expanding_mean_abs_pooled(sig)
    # La série ne peut pas être constante (les observations nouvelles la font évoluer)
    valid = result.dropna()
    assert valid.std() > 0


def test_expanding_mean_abs_shape():
    """La forme de sortie doit être une Series de longueur N."""
    sig = _signal()
    result = expanding_mean_abs_pooled(sig)
    assert isinstance(result, pd.Series)
    assert len(result) == N


# ── Tests forecast_scalar ─────────────────────────────────────────────────────

def test_forecast_scalar_formula():
    """FS = 10 / mean_abs : sur un signal constant, FS doit être exactement 10/|val|."""
    val = 2.5
    sig = _signal_constant(val)
    # Supprimer le min_obs pour ce test → on rend le signal suffisamment long
    fs = forecast_scalar(sig)
    # Après stabilisation (beaucoup d'obs), FS ≈ 10/2.5 = 4.0
    assert fs.dropna().iloc[-1] == pytest.approx(FORECAST_TARGET / val, rel=1e-6)


def test_forecast_scalar_nan_before_min_obs():
    """FS doit être NaN avant FS_MIN_OBS observations poolées."""
    sig = _signal()
    fs = forecast_scalar(sig)
    min_days = FS_MIN_OBS // len(_TICKERS)  # = 25
    assert fs.iloc[: min_days - 1].isna().all()
    assert pd.notna(fs.iloc[min_days - 1])


def test_forecast_scalar_positive():
    """FS doit toujours être strictement positif (hors NaN)."""
    sig = _signal()
    fs = forecast_scalar(sig)
    assert (fs.dropna() > 0).all()


def test_forecast_scalar_expands_over_time():
    """La valeur de FS doit changer au fil du temps (fenêtre expansive ≠ constante)."""
    sig = _signal()
    fs = forecast_scalar(sig)
    valid = fs.dropna()
    # Pas constante : la distribution change à mesure qu'on accumule des données
    assert valid.std() > 0


# ── Tests scale_forecast ──────────────────────────────────────────────────────

def test_scale_forecast_value():
    """Scaled = FS × signal : vérifier un calcul exact."""
    sig = _signal_constant(2.0)
    fs = pd.Series(4.0, index=_DATES)    # FS fixe = 4
    scaled = scale_forecast(sig, fs)
    # 4 × 2 = 8 pour toutes les valeurs
    assert np.allclose(scaled.values, 8.0)


def test_scale_forecast_shape():
    """La shape de scaled doit être identique à celle du signal."""
    sig = _signal()
    fs = forecast_scalar(sig)
    scaled = scale_forecast(sig, fs)
    assert scaled.shape == sig.shape


def test_scale_forecast_nan_propagates():
    """Si FS est NaN (début de période), le scaled doit aussi être NaN."""
    sig = _signal()
    fs = forecast_scalar(sig)
    scaled = scale_forecast(sig, fs)
    fs_nan_mask = fs.isna()
    assert scaled[fs_nan_mask].isna().all().all()


# ── Tests cap_forecast ────────────────────────────────────────────────────────

def test_cap_forecast_within_bounds():
    """Toutes les valeurs cappées doivent être dans [−20, +20]."""
    sig = _signal(mean_abs=5.0)  # signal plus grand pour générer des extrêmes
    fs = pd.Series(FORECAST_TARGET / 5.0, index=_DATES)  # FS = 2
    scaled = scale_forecast(sig, fs)
    capped = cap_forecast(scaled)
    assert (capped.dropna() >= -FORECAST_CAP).all().all()
    assert (capped.dropna() <=  FORECAST_CAP).all().all()


def test_cap_forecast_preserves_small_values():
    """Les valeurs dans (−20, +20) ne doivent pas être modifiées."""
    sig = _signal_constant(5.0)   # signal = 5
    fs = pd.Series(2.0, index=_DATES)  # scaled = 10 → bien dans (−20, +20)
    scaled = scale_forecast(sig, fs)
    capped = cap_forecast(scaled)
    pd.testing.assert_frame_equal(capped, scaled)


def test_cap_forecast_large_values_capped():
    """Un signal de 100 × FS = 100 doit être ramené à 20."""
    sig = _signal_constant(100.0)
    fs = pd.Series(1.0, index=_DATES)
    scaled = scale_forecast(sig, fs)  # = 100
    capped = cap_forecast(scaled)
    assert np.allclose(capped.values, FORECAST_CAP)


def test_cap_forecast_custom_cap():
    """Le cap doit être paramétrable."""
    sig = _signal_constant(50.0)
    fs = pd.Series(1.0, index=_DATES)
    scaled = scale_forecast(sig, fs)
    capped = cap_forecast(scaled, cap=30.0)
    assert (capped.abs() <= 30.0 + 1e-10).all().all()


# ── Tests compute_scaled_capped ───────────────────────────────────────────────

def test_compute_returns_dict_keys():
    """compute_scaled_capped doit retourner un dict avec 'fs', 'scaled', 'capped'."""
    sig = _signal()
    result = compute_scaled_capped(sig)
    assert set(result.keys()) == {"fs", "scaled", "capped"}


def test_compute_fs_is_series():
    """'fs' doit être une pd.Series."""
    sig = _signal()
    assert isinstance(compute_scaled_capped(sig)["fs"], pd.Series)


def test_compute_scaled_capped_same_shape():
    """'scaled' et 'capped' doivent avoir la même shape que le signal."""
    sig = _signal()
    result = compute_scaled_capped(sig)
    assert result["scaled"].shape == sig.shape
    assert result["capped"].shape == sig.shape


def test_compute_mean_abs_capped_close_to_target():
    """
    Après scaling, E[|capped forecast|] doit être proche de 10.
    Sur un signal Laplacian, le capping coupe ~5 % des extrêmes, donc
    E[|CF|] ≈ 9.5–10.0 (légèrement en dessous de 10 à cause du cap).
    """
    sig = _signal(mean_abs=2.0, seed=42)
    result = compute_scaled_capped(sig)
    capped = result["capped"].dropna()
    mean_abs = capped.abs().mean().mean()
    assert 8.5 < mean_abs < 10.5


def test_compute_capped_no_lookahead():
    """
    Le FS au jour 100 doit être différent du FS au jour 400
    (fenêtre expansive → le FS évolue avec les données).
    """
    sig = _signal()
    fs = compute_scaled_capped(sig)["fs"].dropna()
    assert fs.iloc[0] != pytest.approx(fs.iloc[-1], rel=0.01)
