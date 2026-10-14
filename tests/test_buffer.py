"""
tests/test_buffer.py — tests unitaires pour src/systrade/portfolio/buffer.py
"""

import numpy as np
import pandas as pd
import pytest

from systrade.portfolio.buffer import (
    BUFFER_DELTA,
    apply_buffer,
    buffered_trades,
    compute_buffer,
)
from systrade.portfolio.sizing import MAX_LEVERAGE

_DATES = pd.bdate_range("2024-01-02", periods=6)
_TICKERS = ["SPY", "IEF"]


def _targets(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=_DATES[: len(rows)], columns=_TICKERS)


def _initial(spy: float, ief: float = 0.0) -> pd.Series:
    return pd.Series({"SPY": spy, "IEF": ief})


# ── Buffer ────────────────────────────────────────────────────────────────────

def test_buffer_from_flat_trades_to_near_edge():
    """Depuis 0, on va au bord proche (0,90 × N*), pas à la cible."""
    out = apply_buffer(_targets([[100.0, -100.0]]))
    assert out["SPY"].iloc[0] == pytest.approx(90.0)
    assert out["IEF"].iloc[0] == pytest.approx(-90.0)


def test_buffer_from_above_trades_to_far_edge():
    """Une position trop grosse est ramenée à 1,10 × N*, pas à N*."""
    out = apply_buffer(_targets([[100.0, 0.0]]), initial=_initial(150.0))
    assert out["SPY"].iloc[0] == pytest.approx(110.0)


def test_buffer_inside_band_does_not_trade():
    """Un mouvement de cible qui laisse la position dans la zone ne trade pas."""
    targets = _targets([[100.0, 50.0], [104.0, 52.0], [96.0, 48.0]])
    out = apply_buffer(targets, initial=_initial(100.0, 50.0))
    assert out["SPY"].tolist() == pytest.approx([100.0, 100.0, 100.0])
    assert out["IEF"].tolist() == pytest.approx([50.0, 50.0, 50.0])


def test_buffer_large_move_goes_to_new_edge():
    """Si la cible s'éloigne, on rejoint le bord de la nouvelle zone."""
    targets = _targets([[100.0, 0.0], [200.0, 0.0]])
    out = apply_buffer(targets, initial=_initial(100.0))
    assert out["SPY"].iloc[0] == pytest.approx(100.0)
    assert out["SPY"].iloc[1] == pytest.approx(180.0)


def test_buffer_holds_through_missing_target():
    """Une cible NaN ne force pas un trade : on garde la position."""
    targets = _targets([[100.0, 40.0], [np.nan, np.nan], [100.0, 40.0]])
    out = apply_buffer(targets)
    assert out["SPY"].iloc[0] == pytest.approx(90.0)
    assert out["SPY"].iloc[1] == pytest.approx(90.0)
    assert out["SPY"].iloc[2] == pytest.approx(90.0)


def test_buffer_zero_target_flattens():
    """Cible nulle : la zone a une largeur nulle, on solde."""
    out = apply_buffer(_targets([[0.0, 0.0]]), initial=_initial(25.0, -10.0))
    assert out.iloc[0].tolist() == pytest.approx([0.0, 0.0])


def test_buffer_no_lookahead():
    """La position bufferisée à t ne dépend pas des cibles futures."""
    rng = np.random.default_rng(0)
    targets = pd.DataFrame(
        rng.normal(0, 100, (30, 2)),
        index=pd.bdate_range("2024-01-02", periods=30),
        columns=_TICKERS,
    )
    full = apply_buffer(targets)
    trunc = apply_buffer(targets.iloc[:12])
    pd.testing.assert_series_equal(full.iloc[11], trunc.iloc[11])


def test_buffer_rejects_negative_delta():
    with pytest.raises(ValueError):
        apply_buffer(_targets([[100.0, 0.0]]), delta=-0.1)


def test_buffered_trades_start_from_flat():
    """Le premier ordre est la position elle-même, puisqu'on part de zéro."""
    buffered = apply_buffer(_targets([[100.0, -50.0], [100.0, -50.0]]))
    trades = buffered_trades(buffered)
    assert trades["SPY"].iloc[0] == pytest.approx(buffered["SPY"].iloc[0])
    assert trades["SPY"].iloc[1] == pytest.approx(0.0)


# ── Levier après le buffer ────────────────────────────────────────────────────

def test_compute_buffer_keeps_unscaled_state():
    """L'état du buffer ignore le rescaling de levier."""
    targets = _targets([[100.0, 80.0]] * 3)
    prices = pd.DataFrame(50.0, index=targets.index, columns=_TICKERS)
    result = compute_buffer(targets, prices, capital=10_000.0)
    pd.testing.assert_frame_equal(result["buffered"], apply_buffer(targets))


def test_compute_buffer_caps_leverage():
    """Des cibles énormes sont bufferisées d'abord, puis ramenées à 2×."""
    targets = pd.DataFrame(
        {"SPY": [5_000.0, 5_000.0, 5_000.0]},
        index=_DATES[:3],
    )
    prices = pd.DataFrame({"SPY": 100.0}, index=_DATES[:3])
    result = compute_buffer(targets, prices, capital=100_000.0)

    # Depuis 0, le buffer va à 4 500 actions. Notionnel 450 k$ → levier 4,5×
    # avant contrainte, donc la position détenue est réduite pour tomber à 2×.
    assert result["buffered"]["SPY"].iloc[0] == pytest.approx(4_500.0)
    assert result["leverage"].iloc[0] == pytest.approx(MAX_LEVERAGE)
    assert (result["leverage"] <= MAX_LEVERAGE + 1e-9).all()
    assert result["positions"]["SPY"].iloc[0] < result["buffered"]["SPY"].iloc[0]


def test_compute_buffer_does_not_trade_when_only_leverage_binds():
    """
    Si la cible ne bouge pas, le buffer ne bouge pas, même quand le levier
    est contraignant. Le rescaling est un facteur, pas un nouvel ordre de signal.
    """
    targets = pd.DataFrame({"SPY": [5_000.0, 5_000.0]}, index=_DATES[:2])
    prices = pd.DataFrame({"SPY": [100.0, 100.0]}, index=_DATES[:2])
    result = compute_buffer(targets, prices, capital=100_000.0)
    assert result["buffered"]["SPY"].iloc[1] == pytest.approx(result["buffered"]["SPY"].iloc[0])
    assert result["trades"]["SPY"].iloc[1] == pytest.approx(0.0)


def test_compute_buffer_keys():
    targets = _targets([[10.0, -10.0]])
    prices = pd.DataFrame(100.0, index=targets.index, columns=_TICKERS)
    result = compute_buffer(targets, prices, capital=100_000.0, delta=BUFFER_DELTA)
    assert set(result) == {"buffered", "positions", "leverage", "trades"}
