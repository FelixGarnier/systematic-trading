"""
src/systrade/forecast/scalar.py
────────────────────────────────
Scaling et capping des signaux bruts — Bloc 4.

Objectif : amener tous les signaux à E[|F|] = 10 avant combinaison.

Formule :
    FS_t  = 10 / E_expansif_poolé[|signal_t|]
    SF_t  = FS_t × signal_t          (Scaled Forecast)
    CF_t  = clip(SF_t, −20, +20)     (Capped Forecast)

Méthode de calibrage du FS :
    • Fenêtre expansive : FS_t n'utilise que les observations ≤ t → pas de look-ahead.
    • Poolée sur tous les instruments : à chaque date t, on calcule
      mean(|signal|) sur toutes les paires (instrument, date') avec date' ≤ t.
      Implémentation efficace via cumsum pondéré (chaque observation a le même poids).
    • Un seul FS par règle : la normalisation correcte (÷ P×σ pour EWMAC, ÷ σ
      pour carry) place tous les signaux dans le même espace statistique.

Anti look-ahead :
    • Le cumsum progressif garantit que FS_t = f(données ≤ t).
    • Une fenêtre fixe sur l'ensemble de l'historique serait un look-ahead implicite.

Références scalaires Carver (*Systematic Trading*, chap. 12) :
    EWMAC  8/ 32 : ~5.3
    EWMAC 16/ 64 : ~3.75
    EWMAC 32/128 : ~2.65
    Carry        : ~30
"""

from __future__ import annotations

import pandas as pd

# ── Paramètres globaux ────────────────────────────────────────────────────────

FORECAST_TARGET: float = 10.0   # E[|F|] cible
FORECAST_CAP:    float = 20.0   # clip ±20 (coupe ~5 % des extrêmes)
FS_MIN_OBS:      int   = 100    # observations poolées minimales avant de produire un FS


# ── Fonctions de base ─────────────────────────────────────────────────────────

def expanding_mean_abs_pooled(signal: pd.DataFrame) -> pd.Series:
    """
    E_expansif_poolé[|signal|] à chaque date t.

    À la date t, on calcule la moyenne de |signal| sur toutes les paires
    (instrument, date') valides avec date' ≤ t.  Chaque observation compte
    pour 1 (pondération uniforme), quelle que soit l'heure ou l'instrument.

    Implémentation efficace :
        cumsum(|signal|)    / cumcount(non-NaN)
        au lieu de boucler sur les dates.

    Returns
    -------
    pd.Series indexée sur les dates, NaN tant que < FS_MIN_OBS observations.
    """
    daily_abs_sum   = signal.abs().sum(axis=1, skipna=True)
    daily_valid_cnt = signal.notna().sum(axis=1)

    cum_sum   = daily_abs_sum.cumsum()
    cum_count = daily_valid_cnt.cumsum()

    mean_abs = cum_sum / cum_count.replace(0, float("nan"))
    # Masquer les premières dates où on n'a pas encore assez d'observations
    mean_abs = mean_abs.where(cum_count >= FS_MIN_OBS)
    return mean_abs


def forecast_scalar(
    signal: pd.DataFrame,
    target: float = FORECAST_TARGET,
) -> pd.Series:
    """
    FS_t = target / E_expansif_poolé[|signal_t|].

    Returns
    -------
    pd.Series, NaN avant FS_MIN_OBS observations poolées.
    """
    mean_abs = expanding_mean_abs_pooled(signal)
    return target / mean_abs


def scale_forecast(
    signal: pd.DataFrame,
    fs: pd.Series,
) -> pd.DataFrame:
    """
    Scaled Forecast = FS_t × signal_t, broadcast sur tous les instruments.

    Parameters
    ----------
    signal : signal normalisé (même index que fs).
    fs     : Series du Forecast Scalar (une valeur par date).
    """
    return signal.mul(fs, axis=0)


def cap_forecast(
    scaled: pd.DataFrame,
    cap: float = FORECAST_CAP,
) -> pd.DataFrame:
    """
    Capped Forecast = clip(scaled, −cap, +cap).

    Les ~5 % d'observations dépassant ±20 sont ramenées au bord.
    """
    return scaled.clip(-cap, cap)


# ── Point d'entrée principal ──────────────────────────────────────────────────

def compute_scaled_capped(
    signal: pd.DataFrame,
    target: float = FORECAST_TARGET,
    cap:    float = FORECAST_CAP,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Pipeline complet : signal normalisé → FS → Scaled → Capped.

    Parameters
    ----------
    signal : signal normalisé pour une règle (DataFrame, colonnes = tickers).
    target : E[|F|] cible (défaut 10).
    cap    : valeur absolue maximale du forecast (défaut 20).

    Returns
    -------
    dict avec :
        "fs"     → pd.Series  (Forecast Scalar, une valeur par date)
        "scaled" → pd.DataFrame (Scaled Forecast, même shape que signal)
        "capped" → pd.DataFrame (Capped Forecast, même shape que signal)
    """
    fs     = forecast_scalar(signal, target)
    scaled = scale_forecast(signal, fs)
    capped = cap_forecast(scaled, cap)
    return {"fs": fs, "scaled": scaled, "capped": capped}
