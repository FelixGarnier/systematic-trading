"""
src/systrade/rules/ewmac.py
────────────────────────────
Règles de trend-following EWMAC (Exponentially Weighted Moving Average Crossover).

Trois vitesses validées — Bloc 3 :
    Fast   : (f=8,  s=32)
    Medium : (f=16, s=64)
    Slow   : (f=32, s=128)

Formule :
    EWMAC_{f,s,t}  = EMA_f(P_adj_t) - EMA_s(P_adj_t)
    Signal_t       = EWMAC_{f,s,t} / (P_adj_{t-1} × σ_daily_t)

Anti look-ahead :
    • Les EMA sont des filtres récursifs sur le passé : pas de look-ahead par construction.
    • Le dénominateur utilise P_adj_{t-1}  (shift 1) — jamais le prix du jour courant.
    • σ_daily_t est calculé sur les rendements jusqu'à la clôture de t (bloc 2) : OK.
"""

from __future__ import annotations

import pandas as pd

# (fast_span, slow_span) — Carver, *Systematic Trading* chap. 12
EWMAC_SPEEDS: dict[str, tuple[int, int]] = {
    "EWMAC_8_32":   (8,  32),
    "EWMAC_16_64":  (16, 64),
    "EWMAC_32_128": (32, 128),
}


# ── Fonctions de base ─────────────────────────────────────────────────────────

def ewmac_raw(
    prices_adj: pd.DataFrame,
    fast: int,
    slow: int,
) -> pd.DataFrame:
    """
    EMA_fast(P) - EMA_slow(P) pour tous les instruments.

    min_periods=1 : l'EMA démarre dès la première observation disponible.
    La période de chauffe (2 ans) du backtest filtre les premières valeurs instables.
    """
    ema_fast = prices_adj.ewm(span=fast, min_periods=1).mean()
    ema_slow = prices_adj.ewm(span=slow, min_periods=1).mean()
    return ema_fast - ema_slow


def normalize_ewmac(
    raw: pd.DataFrame,
    prices_adj: pd.DataFrame,
    vol_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Signal normalisé = raw / (P_adj_{t-1} × σ_daily_t).

    P_adj_{t-1} via shift(1) → aucun look-ahead.
    Les 0 / NaN au dénominateur sont remplacés par NaN (propagé proprement).
    """
    denom = prices_adj.shift(1) * vol_daily
    denom = denom.replace(0.0, float("nan"))
    return raw / denom


def ewmac_signal(
    prices_adj: pd.DataFrame,
    vol_daily: pd.DataFrame,
    fast: int,
    slow: int,
) -> pd.DataFrame:
    """Signal EWMAC normalisé pour une paire (fast, slow) donnée."""
    raw = ewmac_raw(prices_adj, fast, slow)
    return normalize_ewmac(raw, prices_adj, vol_daily)


# ── Point d'entrée principal ──────────────────────────────────────────────────

def all_ewmac_signals(
    prices_adj: pd.DataFrame,
    vol_daily: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Calcule les trois signaux EWMAC normalisés.

    Parameters
    ----------
    prices_adj : prix ajustés (total return), colonnes = tickers.
    vol_daily  : volatilité journalière (bloc 2), même shape.

    Returns
    -------
    dict[str, pd.DataFrame] avec clés "EWMAC_8_32", "EWMAC_16_64", "EWMAC_32_128".
    """
    return {
        name: ewmac_signal(prices_adj, vol_daily, fast, slow)
        for name, (fast, slow) in EWMAC_SPEEDS.items()
    }
