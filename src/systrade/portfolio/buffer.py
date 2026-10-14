"""
src/systrade/portfolio/buffer.py
─────────────────────────────────
Buffer de trading et contrainte de levier — Bloc 8.

Le buffer est un filtre de décision. On ne trade que si la position détenue
sort de la zone autour de la cible, et on ramène alors au bord de la zone,
pas à la cible. Ça coupe le turnover des petits mouvements de signal.

    zone_t = [N*_t − δ|N*_t|,  N*_t + δ|N*_t|]     δ = 0,10

Pour N* > 0, c'est [0,90 N*, 1,10 N*].

Ordre, non négociable :

    N* (après poids et IDM) → buffer → position à trader → levier → ordre

Le buffer mémorise la position *avant* le rescaling de levier. Sinon un
simple ajustement de levier, sans changement de signal, redéclencherait un
trade le lendemain.

La position réellement détenue est la position bufferisée, puis réduite
uniformément si le levier brut dépasse 2×.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from systrade.portfolio.sizing import (
    MAX_LEVERAGE,
    apply_leverage_constraint,
    gross_leverage,
)

BUFFER_DELTA: float = 0.10


def apply_buffer(
    targets: pd.DataFrame,
    delta: float = BUFFER_DELTA,
    initial: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Position après buffer, sans contrainte de levier.

    Parameters
    ----------
    targets : position cible N* (après IDM), une colonne par instrument.
    delta   : demi-largeur de la zone, en fraction de |N*|.
    initial : position déjà détenue avant la première date. Défaut : flat.

    Une cible NaN ne change pas la position (instrument pas encore listé,
    ou forecast indisponible).
    """
    if delta < 0:
        raise ValueError(f"delta doit être ≥ 0, reçu {delta}")

    values = targets.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    out = np.zeros((n_rows, n_cols), dtype=float)

    if initial is None:
        current = np.zeros(n_cols, dtype=float)
    else:
        current = initial.reindex(targets.columns).fillna(0.0).to_numpy(dtype=float)

    for t in range(n_rows):
        target = values[t]
        valid = np.isfinite(target)
        if valid.any():
            width = delta * np.abs(target[valid])
            lower = target[valid] - width
            upper = target[valid] + width
            current = current.copy()
            current[valid] = np.clip(current[valid], lower, upper)
        out[t] = current

    return pd.DataFrame(out, index=targets.index, columns=targets.columns)


def buffered_trades(buffered: pd.DataFrame) -> pd.DataFrame:
    """Variation de la position bufferisée. Le premier jour, on part de zéro."""
    trades = buffered.diff()
    trades.iloc[0] = buffered.iloc[0]
    return trades


def compute_buffer(
    targets: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    capital: float | pd.Series,
    delta: float = BUFFER_DELTA,
    max_leverage: float = MAX_LEVERAGE,
    initial: pd.Series | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Pipeline bloc 8 : buffer, puis levier.

    Returns
    -------
    buffered : position que le buffer retient (avant levier). C'est son état.
    positions: position détenue après levier. C'est ce qu'on envoie au marché.
    leverage : levier brut de `positions`.
    trades   : variation de `positions` (ordres), en nombre d'actions.
    """
    buffered = apply_buffer(targets, delta=delta, initial=initial)
    positions = apply_leverage_constraint(
        buffered, prices_unadj, capital, max_leverage=max_leverage
    )
    trades = positions.diff()
    trades.iloc[0] = positions.iloc[0]

    return {
        "buffered": buffered,
        "positions": positions,
        "leverage": gross_leverage(positions, prices_unadj, capital),
        "trades": trades,
    }
