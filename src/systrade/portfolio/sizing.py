"""
src/systrade/portfolio/sizing.py
──────────────────────────────────
Position sizing — Bloc 6.

Formule centrale (Carver, *Systematic Trading*) :

    N*_t = Capital_t × τ × F_t / (10 × σ_annual_t × P_brut_t)

    • F_t        : Combined Forecast ∈ [−20, +20], E[|F|] = 10
    • τ          : volatility target (13 %)
    • σ_annual_t : volatilité annualisée du bloc 2 (plancher déjà appliqué)
    • P_brut_t   : prix brut (non ajusté)
    • Capital_t  : valeur courante du portefeuille (pas le capital initial)
    • 10         : E[|F|], facteur de normalisation

Réécriture pour intuition :

    N*_t = Volatility_Scalar_t × F_t / 10
    Volatility_Scalar_t = Capital_t × τ / (σ_annual_t × P_brut_t)

    À F = 10 (forecast max attendu) → position contribue exactement τ de vol annuelle.

Sizing sur capital courant — anti-ruin :
    Après un drawdown de 20 %, le capital tombe à 80 k$ et les positions
    se réduisent automatiquement de 20 %.  Avec capital initial fixe,
    le levier effectif monterait à 1,25× après −20 %, ce qui amplifie les pertes.

Contrainte de levier :

    Levier brut_t = Σ |N_i × P_brut_i| / Capital_t ≤ max_leverage (2×)

    Si dépassé : rescaling uniforme de TOUTES les positions.
    Scale = min(max_leverage / levier_t, 1)  → on ne peut qu'ABAISSER le levier.

Ordre des opérations (Bloc 8) :
    signal → N* → buffer → vérification levier → ordre
    Le buffer est appliqué entre N* et le levier ici on ne fait que N* + levier.
"""

from __future__ import annotations

import pandas as pd

# ── Constantes (config globale) ───────────────────────────────────────────────

VOL_TARGET:     float = 0.13    # τ = 13 % annualisé
MAX_LEVERAGE:   float = 2.0     # levier brut maximal
FORECAST_SCALE: float = 10.0   # E[|F|]


# ── Fonctions de base ─────────────────────────────────────────────────────────

def target_positions(
    capital:       float | pd.Series,
    forecast:      pd.DataFrame,
    vol_annual:    pd.DataFrame,
    prices_unadj:  pd.DataFrame,
    vol_target:    float = VOL_TARGET,
    forecast_scale: float = FORECAST_SCALE,
) -> pd.DataFrame:
    """
    N*_t = Capital_t × τ × F_t / (forecast_scale × σ_annual_t × P_brut_t)

    Parameters
    ----------
    capital      : valeur courante du portefeuille (scalaire ou Series indexée par date).
    forecast     : Combined Forecast ∈ [−20, +20].
    vol_annual   : σ annualisée (bloc 2, plancher inclus).
    prices_unadj : prix bruts (non ajustés).
    vol_target   : τ = 13 % par défaut.
    forecast_scale: E[|F|] = 10.

    Returns
    -------
    DataFrame de positions cibles en nombre d'actions (float, peut être négatif).
    """
    cap = _to_series(capital, forecast.index)

    # Dénominateur : forecast_scale × σ_annual × P_brut
    tickers = forecast.columns
    denom = (
        forecast_scale
        * vol_annual.reindex(columns=tickers)
        * prices_unadj.reindex(columns=tickers)
    )
    denom = denom.replace(0.0, float("nan"))

    # Numérateur : Capital × τ × F
    numerator = forecast.mul(cap * vol_target, axis=0)
    return numerator / denom


def apply_leverage_constraint(
    positions:    pd.DataFrame,
    prices_unadj: pd.DataFrame,
    capital:      float | pd.Series,
    max_leverage: float = MAX_LEVERAGE,
) -> pd.DataFrame:
    """
    Rescaling uniforme si levier brut > max_leverage.

    Scale = min(max_leverage / levier_brut, 1.0)
    → on ne fait jamais grossir les positions, seulement les réduire.

    Returns
    -------
    DataFrame de positions ajustées (même shape que positions).
    """
    cap = _to_series(capital, positions.index)

    tickers = positions.columns
    notional = (positions * prices_unadj.reindex(columns=tickers)).abs()
    gross    = notional.sum(axis=1)

    leverage = gross / cap.replace(0.0, float("nan"))
    scale    = (max_leverage / leverage).clip(upper=1.0)

    return positions.mul(scale, axis=0)


def gross_leverage(
    positions:    pd.DataFrame,
    prices_unadj: pd.DataFrame,
    capital:      float | pd.Series,
) -> pd.Series:
    """
    Levier brut_t = Σ |N_i × P_brut_i| / Capital_t.
    """
    cap     = _to_series(capital, positions.index)
    tickers = positions.columns
    notional = (positions * prices_unadj.reindex(columns=tickers)).abs()
    return notional.sum(axis=1) / cap.replace(0.0, float("nan"))


# ── Point d'entrée principal ──────────────────────────────────────────────────

def compute_positions(
    capital:          float | pd.Series,
    combined_forecast: pd.DataFrame,
    vol_annual:        pd.DataFrame,
    prices_unadj:      pd.DataFrame,
    vol_target:    float = VOL_TARGET,
    max_leverage:  float = MAX_LEVERAGE,
    forecast_scale: float = FORECAST_SCALE,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Pipeline complet : Combined Forecast → N* → contrainte levier → N_final.

    Returns
    -------
    dict :
        "positions_target" → N* avant contrainte de levier
        "positions"        → N après contrainte de levier
        "leverage"         → levier brut après contrainte (Series)
        "vol_scalar"       → Volatility Scalar = Capital × τ / (σ × P_brut)
    """
    N_target  = target_positions(
        capital, combined_forecast, vol_annual, prices_unadj, vol_target, forecast_scale
    )
    N_final   = apply_leverage_constraint(N_target, prices_unadj, capital, max_leverage)
    lev       = gross_leverage(N_final, prices_unadj, capital)
    vol_scal  = target_positions(
        capital,
        pd.DataFrame(forecast_scale, index=combined_forecast.index, columns=combined_forecast.columns),
        vol_annual, prices_unadj, vol_target, forecast_scale,
    )

    return {
        "positions_target": N_target,
        "positions":        N_final,
        "leverage":         lev,
        "vol_scalar":       vol_scal,
    }


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _to_series(value: float | pd.Series, index: pd.Index) -> pd.Series:
    """Convertit un scalaire ou une Series en Series alignée sur index."""
    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series(float(value), index=index)
