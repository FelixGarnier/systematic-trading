"""
tests/test_sizing.py — tests unitaires pour src/systrade/portfolio/sizing.py
"""

import numpy as np
import pandas as pd
import pytest

from systrade.portfolio.sizing import (
    FORECAST_SCALE,
    MAX_LEVERAGE,
    VOL_TARGET,
    apply_leverage_constraint,
    compute_positions,
    gross_leverage,
    target_positions,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

N = 300
_DATES   = pd.bdate_range("2020-01-02", periods=N)
_TICKERS = ["SPY", "IWM", "GLD"]

CAPITAL  = 100_000.0
PRICE    = 400.0        # prix brut unique pour tous les instruments
VOL_ANN  = 0.16         # σ_annual = 16 %


def _const(val: float, tickers=_TICKERS) -> pd.DataFrame:
    return pd.DataFrame(val, index=_DATES, columns=tickers)


def _forecast(val: float = 10.0) -> pd.DataFrame:
    return _const(val)


def _vol() -> pd.DataFrame:
    return _const(VOL_ANN)


def _prices() -> pd.DataFrame:
    return _const(PRICE)


# ── Tests target_positions ────────────────────────────────────────────────────

def test_target_positions_formula():
    """
    N* = Capital × τ × F / (10 × σ × P).
    Avec F=10, τ=0.13, σ=0.16, P=400 :
    N* = 100 000 × 0.13 × 10 / (10 × 0.16 × 400) = 13 000 / 640 ≈ 20.3125
    """
    N = target_positions(CAPITAL, _forecast(10.0), _vol(), _prices())
    expected = CAPITAL * VOL_TARGET * 10.0 / (FORECAST_SCALE * VOL_ANN * PRICE)
    assert N["SPY"].iloc[0] == pytest.approx(expected, rel=1e-6)


def test_target_positions_positive_forecast_positive_position():
    """Un forecast positif doit donner une position longue (N* > 0)."""
    N = target_positions(CAPITAL, _forecast(+10.0), _vol(), _prices())
    assert (N.dropna() > 0).all().all()


def test_target_positions_negative_forecast_negative_position():
    """Un forecast négatif doit donner une position courte (N* < 0)."""
    N = target_positions(CAPITAL, _forecast(-10.0), _vol(), _prices())
    assert (N.dropna() < 0).all().all()


def test_target_positions_zero_forecast_zero_position():
    """Un forecast nul doit donner une position nulle."""
    pos = target_positions(CAPITAL, _forecast(0.0), _vol(), _prices())
    assert np.allclose(pos.dropna().values, 0.0)


def test_target_positions_vol_targeting():
    """
    À F = 10 (forecast cible), la position doit contribuer exactement τ = 13 %
    de volatilité annuelle sur le capital.

    Vol_contribution = |N* × P × σ_annual| / Capital = τ
    """
    N_df = target_positions(CAPITAL, _forecast(10.0), _vol(), _prices())
    N = N_df["SPY"].iloc[0]
    vol_contribution = abs(N * PRICE * VOL_ANN) / CAPITAL
    assert vol_contribution == pytest.approx(VOL_TARGET, rel=1e-6)


def test_target_positions_proportional_to_forecast():
    """N* doit être proportionnel à F (relation linéaire)."""
    N10 = target_positions(CAPITAL, _forecast(10.0), _vol(), _prices())
    N20 = target_positions(CAPITAL, _forecast(20.0), _vol(), _prices())
    assert N20["SPY"].iloc[0] == pytest.approx(2 * N10["SPY"].iloc[0], rel=1e-6)


def test_target_positions_proportional_to_capital():
    """N* doit être proportionnel au capital."""
    N100k = target_positions(100_000, _forecast(), _vol(), _prices())
    N200k = target_positions(200_000, _forecast(), _vol(), _prices())
    assert N200k["SPY"].iloc[0] == pytest.approx(2 * N100k["SPY"].iloc[0], rel=1e-6)


def test_target_positions_inverse_to_vol():
    """N* doit être inversement proportionnel à la volatilité."""
    N_low  = target_positions(CAPITAL, _forecast(), _const(0.08),  _prices())
    N_high = target_positions(CAPITAL, _forecast(), _const(0.16), _prices())
    assert N_low["SPY"].iloc[0] == pytest.approx(2 * N_high["SPY"].iloc[0], rel=1e-6)


def test_target_positions_inverse_to_price():
    """N* doit être inversement proportionnel au prix brut."""
    N_cheap = target_positions(CAPITAL, _forecast(), _vol(), _const(200.0))
    N_dear  = target_positions(CAPITAL, _forecast(), _vol(), _const(400.0))
    assert N_cheap["SPY"].iloc[0] == pytest.approx(2 * N_dear["SPY"].iloc[0], rel=1e-6)


def test_target_positions_capital_series():
    """Capital peut être une Series (valeur courante variable)."""
    cap = pd.Series(CAPITAL * (1 + np.linspace(0, 0.2, N)), index=_DATES)
    pos = target_positions(cap, _forecast(), _vol(), _prices())
    # La position du premier jour doit correspondre au capital initial
    expected_0 = CAPITAL * VOL_TARGET * 10.0 / (FORECAST_SCALE * VOL_ANN * PRICE)
    assert pos["SPY"].iloc[0] == pytest.approx(expected_0, rel=1e-4)
    # La position doit croître avec le capital
    assert pos["SPY"].iloc[-1] > pos["SPY"].iloc[0]


def test_target_positions_shape():
    """La shape de sortie doit être identique à celle de forecast."""
    N = target_positions(CAPITAL, _forecast(), _vol(), _prices())
    assert N.shape == _forecast().shape


# ── Tests apply_leverage_constraint ──────────────────────────────────────────

def test_leverage_constraint_applied():
    """Des positions trop grosses doivent être rescalées vers max_leverage."""
    # F = +20 (max forecast) avec vol 16% et prix 400$ → levier >> 2
    N_big = target_positions(CAPITAL, _forecast(20.0), _vol(), _prices())
    N_adj = apply_leverage_constraint(N_big, _prices(), CAPITAL)
    lev = gross_leverage(N_adj, _prices(), CAPITAL)
    assert (lev.dropna() <= MAX_LEVERAGE + 1e-9).all()


def test_leverage_constraint_not_applied_when_below():
    """Des positions sous-levées ne doivent pas être modifiées."""
    # F = 1 → très faible levier
    N_small = target_positions(CAPITAL, _forecast(1.0), _vol(), _prices())
    N_adj   = apply_leverage_constraint(N_small, _prices(), CAPITAL)
    pd.testing.assert_frame_equal(N_small.dropna(), N_adj.dropna())


def test_leverage_constraint_never_amplifies():
    """La contrainte ne doit jamais augmenter les positions (scale ≤ 1)."""
    N = target_positions(CAPITAL, _forecast(5.0), _vol(), _prices())
    N_adj = apply_leverage_constraint(N, _prices(), CAPITAL)
    # Chaque position ajustée ≤ position originale en valeur absolue
    assert (N_adj.abs().dropna() <= N.abs().dropna() + 1e-9).all().all()


def test_leverage_constraint_uniform_rescaling():
    """Le rescaling doit être uniforme : les ratios entre instruments sont conservés."""
    N = target_positions(CAPITAL, _forecast(20.0), _vol(), _prices())
    N_adj = apply_leverage_constraint(N, _prices(), CAPITAL)
    # Ratio SPY/IWM doit être conservé après rescaling
    ratio_before = (N["SPY"] / N["IWM"]).dropna()
    ratio_after  = (N_adj["SPY"] / N_adj["IWM"]).dropna()
    pd.testing.assert_series_equal(ratio_before, ratio_after, check_names=False)


# ── Tests gross_leverage ──────────────────────────────────────────────────────

def test_gross_leverage_formula():
    """
    Levier = Σ |N_i × P_i| / Capital.
    Avec N=20 actions × 400$ × 3 instruments = 24 000$ / 100 000$ = 0.24.
    """
    N = pd.DataFrame(20.0, index=_DATES, columns=_TICKERS)
    lev = gross_leverage(N, _prices(), CAPITAL)
    expected = 20.0 * PRICE * len(_TICKERS) / CAPITAL
    assert lev.iloc[0] == pytest.approx(expected, rel=1e-6)


def test_gross_leverage_long_short_symmetry():
    """Le levier brut doit traiter les positions longues et courtes de façon identique."""
    N_long  = pd.DataFrame( 10.0, index=_DATES, columns=_TICKERS)
    N_short = pd.DataFrame(-10.0, index=_DATES, columns=_TICKERS)
    lev_l = gross_leverage(N_long,  _prices(), CAPITAL)
    lev_s = gross_leverage(N_short, _prices(), CAPITAL)
    pd.testing.assert_series_equal(lev_l, lev_s)


# ── Tests compute_positions ───────────────────────────────────────────────────

def test_compute_positions_keys():
    """compute_positions doit retourner les clés attendues."""
    result = compute_positions(CAPITAL, _forecast(), _vol(), _prices())
    assert set(result.keys()) == {"positions_target", "positions", "leverage", "vol_scalar"}


def test_compute_positions_leverage_within_bounds():
    """Le levier final doit toujours être ≤ max_leverage."""
    result = compute_positions(CAPITAL, _forecast(20.0), _vol(), _prices())
    assert (result["leverage"].dropna() <= MAX_LEVERAGE + 1e-9).all()


def test_compute_positions_target_vs_final():
    """Les positions finales doivent être ≤ aux positions cibles en valeur absolue."""
    result = compute_positions(CAPITAL, _forecast(20.0), _vol(), _prices())
    N_t = result["positions_target"].dropna()
    N_f = result["positions"].dropna()
    assert (N_f.abs() <= N_t.abs() + 1e-9).all().all()


def test_compute_positions_vol_scalar_at_f10():
    """
    Le Volatility Scalar = Capital × τ / (σ × P) doit correspondre à N*(F=10).
    """
    result = compute_positions(CAPITAL, _forecast(10.0), _vol(), _prices())
    expected_vs = CAPITAL * VOL_TARGET / (VOL_ANN * PRICE)
    assert result["vol_scalar"]["SPY"].iloc[0] == pytest.approx(expected_vs, rel=1e-4)
