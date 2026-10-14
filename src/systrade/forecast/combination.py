"""
src/systrade/forecast/combination.py
──────────────────────────────────────
Combinaison des capped forecasts — Bloc 5.

Formule :
    Avg_t = Σ w_i × CF_i_t
    FDM_t = clip(1 / √(w'Σw), 1, FDM_CAP)
    Combined_t = clip(FDM_t × Avg_t, −20, +20)

    où Σ = matrice de corrélation des capped forecasts, estimée par
    fenêtre expansive poolée sur tous les instruments du groupe.

Deux groupes :
    Trend + Carry (10 instruments) : EWMAC ×3 + Carry ×2,  FDM ~1.19
    Trend only    ( 4 instruments) : EWMAC ×3,              FDM ~1.08

Pour les instruments Trend-only, les règles carry sont absentes de
capped_forecasts → weighted_average_forecast renormalise automatiquement
les poids EWMAC de 0.233 à 0.333.  Pas de branche conditionnelle nécessaire.

Anti-instabilité :
    • Corrélations négatives → 0 avant calcul FDM (évite FDM artificiellement élevé).
    • Plafond FDM = 2.5.
    • EWMA long (span 125) sur la série FDM (évite les sauts brusques de position).

Anti look-ahead :
    • Corrélations calculées par statistiques cumulatives (cumsum) sur le passé.
    • Au temps t, on n'utilise que les observations (instrument, date') avec date' ≤ t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Constantes ────────────────────────────────────────────────────────────────

RULE_WEIGHTS_TREND_CARRY: dict[str, float] = {
    "EWMAC_8_32":   7 / 30,   # ≈ 0.2333
    "EWMAC_16_64":  7 / 30,
    "EWMAC_32_128": 7 / 30,
    "carry_medium": 3 / 20,   # 0.15
    "carry_slow":   3 / 20,
}

RULE_WEIGHTS_TREND_ONLY: dict[str, float] = {
    "EWMAC_8_32":   1 / 3,
    "EWMAC_16_64":  1 / 3,
    "EWMAC_32_128": 1 / 3,
}

TREND_ONLY_TICKERS: list[str] = ["GLD", "DBC", "UUP", "TIP"]

FDM_CAP:      float = 2.5
FORECAST_CAP: float = 20.0
FDM_POOL_MIN: int   = 500   # observations poolées minimales avant de produire un FDM
FDM_EWMA_SPAN: int  = 125   # ~6 mois de lissage


# ── FDM analytique ────────────────────────────────────────────────────────────

def fdm_from_corr(
    weights: dict[str, float],
    corr: np.ndarray,
    rule_names: list[str],
) -> float:
    """
    FDM = min(1 / √(w'Σw), FDM_CAP).

    Parameters
    ----------
    weights    : {rule_name: weight}, doit sommer à 1.
    corr       : matrice de corrélation (n_rules × n_rules), numpy array.
    rule_names : ordre des lignes/colonnes dans corr.
    """
    w = np.array([weights[r] for r in rule_names], dtype=float)

    # Corrélations négatives → 0 (évite FDM artificiellement haut)
    corr_clipped = np.where(corr < 0.0, 0.0, corr)
    np.fill_diagonal(corr_clipped, 1.0)   # la diagonale reste 1

    wTSw = float(w @ corr_clipped @ w)
    if wTSw <= 0:
        return 1.0
    fdm = 1.0 / np.sqrt(wTSw)
    return float(min(fdm, FDM_CAP))


# ── Corrélation poolée expansive ──────────────────────────────────────────────

def _pooled_expanding_corr(
    capped_forecasts: dict[str, pd.DataFrame],
    rule_names: list[str],
    tickers: list[str],
    min_obs: int = FDM_POOL_MIN,
) -> dict[pd.Timestamp, np.ndarray]:
    """
    Pour chaque date t, calcule la matrice de corrélation poolée des capped
    forecasts sur toutes les observations (instrument, date') avec date' ≤ t.

    Implémentation par statistiques cumulatives (cumsum) — O(1) par date.

    Returns
    -------
    dict {date → corr_matrix (n_rules × n_rules numpy array)}
    """
    n = len(rule_names)

    # Construire le DataFrame poolé : une ligne par (instrument × jour)
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        if all(
            r in capped_forecasts and ticker in capped_forecasts[r].columns
            for r in rule_names
        ):
            frames.append(
                pd.DataFrame({r: capped_forecasts[r][ticker] for r in rule_names})
            )

    if not frames:
        return {}

    pooled = pd.concat(frames, axis=0).sort_index()

    # Statistiques cumulatives
    cum_n     = 0
    cum_sum   = np.zeros(n)
    cum_sum2  = np.zeros(n)
    cum_cross = np.zeros((n, n))

    date_corr: dict[pd.Timestamp, np.ndarray] = {}

    for date, group in pooled.groupby(level=0):
        rows = group.values.astype(float)
        valid = rows[~np.isnan(rows).any(axis=1)]

        if len(valid) == 0:
            continue

        cum_n     += len(valid)
        cum_sum   += valid.sum(axis=0)
        cum_sum2  += (valid ** 2).sum(axis=0)
        cum_cross += valid.T @ valid

        if cum_n >= min_obs:
            mean = cum_sum / cum_n
            var  = np.maximum(cum_sum2 / cum_n - mean ** 2, 1e-12)
            std  = np.sqrt(var)
            cov  = cum_cross / cum_n - np.outer(mean, mean)
            corr = cov / np.outer(std, std)
            np.fill_diagonal(corr, 1.0)
            corr = np.clip(corr, -1.0, 1.0)
            date_corr[date] = corr

    return date_corr


def expanding_fdm(
    capped_forecasts: dict[str, pd.DataFrame],
    weights: dict[str, float],
    tickers: list[str],
    all_dates: pd.DatetimeIndex,
    min_obs: int   = FDM_POOL_MIN,
    ewma_span: int = FDM_EWMA_SPAN,
) -> pd.Series:
    """
    Série temporelle du FDM — fenêtre expansive poolée + lissage EWMA.

    1. Corrélation poolée expansive (pas de look-ahead)
    2. FDM analytique à chaque date disponible
    3. Reindex sur all_dates + forward-fill
    4. Lissage EWMA (évite les sauts de position)

    Returns
    -------
    pd.Series indexée sur all_dates.  Vaut 1.0 avant min_obs observations.
    """
    rule_names = [r for r in weights if r in capped_forecasts]
    if not rule_names:
        return pd.Series(1.0, index=all_dates)

    date_corr = _pooled_expanding_corr(capped_forecasts, rule_names, tickers, min_obs)

    if not date_corr:
        return pd.Series(1.0, index=all_dates)

    # FDM à chaque date calculée
    fdm_vals = {
        date: fdm_from_corr(
            {r: weights[r] for r in rule_names},
            corr,
            rule_names,
        )
        for date, corr in date_corr.items()
    }

    fdm_raw = pd.Series(fdm_vals).reindex(all_dates)

    # Avant la première estimation : FDM = 1 (aucune diversification assumée)
    fdm_raw = fdm_raw.ffill().fillna(1.0)

    # Lissage EWMA pour éviter les sauts de position
    return fdm_raw.ewm(span=ewma_span, min_periods=1).mean()


# ── Moyenne pondérée ──────────────────────────────────────────────────────────

def weighted_average_forecast(
    capped_forecasts: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    Avg_t(instrument) = Σ_i w_i × CF_i_t(instrument).

    Gestion des règles manquantes :
        Pour un instrument sans carry (trend-only), les colonnes carry
        sont absentes → les poids sont renormalisés sur les règles disponibles.
        Exemple : w_EWMAC = 0.233 renormalisé à 0.333 (= 1/3 exact).
    """
    available_rules = [r for r in weights if r in capped_forecasts]
    if not available_rules:
        return pd.DataFrame()

    ref_df = capped_forecasts[available_rules[0]]
    all_tickers = list(ref_df.columns)
    result = pd.DataFrame(0.0, index=ref_df.index, columns=all_tickers, dtype=float)

    for ticker in all_tickers:
        ticker_rules = [
            r for r in available_rules
            if ticker in capped_forecasts[r].columns
        ]
        if not ticker_rules:
            result[ticker] = float("nan")
            continue

        total_w = sum(weights[r] for r in ticker_rules)
        ticker_avg = sum(
            (weights[r] / total_w) * capped_forecasts[r][ticker]
            for r in ticker_rules
        )
        result[ticker] = ticker_avg

    return result


# ── Point d'entrée principal ──────────────────────────────────────────────────

def compute_combination(
    capped_forecasts: dict[str, pd.DataFrame],
    rule_weights_tc:     dict[str, float] | None = None,
    rule_weights_to:     dict[str, float] | None = None,
    trend_only_tickers:  list[str] | None = None,
    cap:       float = FORECAST_CAP,
    min_obs:   int   = FDM_POOL_MIN,
    ewma_span: int   = FDM_EWMA_SPAN,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Pipeline complet de combinaison pour tous les instruments.

    Parameters
    ----------
    capped_forecasts   : {rule_name: DataFrame(dates × tickers)} — sorties du bloc 4.
    rule_weights_tc    : poids Trend+Carry (défaut RULE_WEIGHTS_TREND_CARRY).
    rule_weights_to    : poids Trend-only (défaut RULE_WEIGHTS_TREND_ONLY).
    trend_only_tickers : instruments sans carry (défaut TREND_ONLY_TICKERS).

    Returns
    -------
    dict :
        "combined" → DataFrame — Combined Forecast ∈ [−20, +20] (tous instruments)
        "fdm_tc"   → Series    — FDM Trend+Carry (time-varying)
        "fdm_to"   → Series    — FDM Trend-only  (time-varying)
        "avg"      → DataFrame — Weighted average avant FDM
    """
    if rule_weights_tc   is None: rule_weights_tc   = RULE_WEIGHTS_TREND_CARRY
    if rule_weights_to   is None: rule_weights_to   = RULE_WEIGHTS_TREND_ONLY
    if trend_only_tickers is None: trend_only_tickers = TREND_ONLY_TICKERS

    ref_df     = next(iter(capped_forecasts.values()))
    all_dates  = ref_df.index
    all_tickers = list(ref_df.columns)

    tc_tickers = [t for t in all_tickers if t not in trend_only_tickers]
    to_tickers = [t for t in all_tickers if t     in trend_only_tickers]

    # ── FDM par groupe ────────────────────────────────────────────────────────
    fdm_tc = expanding_fdm(
        capped_forecasts, rule_weights_tc, tc_tickers, all_dates, min_obs, ewma_span
    )
    fdm_to = expanding_fdm(
        {k: v for k, v in capped_forecasts.items() if k in rule_weights_to},
        rule_weights_to, to_tickers, all_dates, min_obs, ewma_span
    )

    # ── Moyenne pondérée (une seule passe, renorm auto pour trend-only) ───────
    avg = weighted_average_forecast(capped_forecasts, rule_weights_tc)

    # ── Combined forecast par groupe ──────────────────────────────────────────
    combined_tc = avg[tc_tickers].mul(fdm_tc, axis=0).clip(-cap, cap)
    combined_to = avg[to_tickers].mul(fdm_to, axis=0).clip(-cap, cap)

    combined = pd.concat([combined_tc, combined_to], axis=1).reindex(columns=all_tickers)

    return {
        "combined": combined,
        "fdm_tc":   fdm_tc,
        "fdm_to":   fdm_to,
        "avg":      avg,
    }
