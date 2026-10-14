"""
volatility.py — estimation de la volatilité journalière des instruments.

Pipeline (4 étapes) :
    Prix ajustés → Rendements → σ_EWMA → Blend (70/30 LT) → Floor → σ_final

Toutes les grandeurs calculées au jour t n'utilisent que les données ≤ t.
Résultat principal : volatilité quotidienne en décimal (ex. 0.0094 = 0.94%).
Annualisation : σ_annual = σ_daily × √256 = σ_daily × 16.
"""

import pandas as pd
import numpy as np


# ── Étape 1 : rendements journaliers ──────────────────────────────────────────

def daily_returns(prices_adj: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements journaliers à partir des prix ajustés.

    r_t = P_adj_t / P_adj_{t-1} - 1

    Les prix ajustés (total return) capturent les dividendes dans le rendement.
    La première ligne est NaN par construction (pas de t-1 pour t=0).

    Paramètres
    ----------
    prices_adj :
        DataFrame large (dates × tickers), prix ajustés.

    Retourne
    --------
    DataFrame de même forme, valeurs en décimal (0.01 = +1%).
    """
    return prices_adj.pct_change()


# ── Étape 2 : volatilité EWMA ─────────────────────────────────────────────────

def ewma_vol(returns: pd.DataFrame, span: int = 35) -> pd.DataFrame:
    """
    Volatilité EWMA quotidienne.

    Utilise ewm(span=span).std() de pandas, qui implémente :
        λ = 1 - 2/(span+1)  →  span=35  →  λ ≈ 0.944
        σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t

    Résultat en volatilité quotidienne (décimal).
    Les premières lignes sont NaN jusqu'à ce que l'EWMA soit initialisée
    (pandas retourne une valeur dès la 2e observation, mais elle est peu
    fiable avant ~min_periods observations).

    Paramètres
    ----------
    returns :
        DataFrame de rendements journaliers (sortie de daily_returns()).
    span :
        Span de l'EWMA (défaut : 35, convention Carver).

    Retourne
    --------
    DataFrame de même forme, σ_daily en décimal.
    """
    return returns.ewm(span=span, min_periods=10).std()


# ── Étape 3 : volatilité long terme ──────────────────────────────────────────

def long_term_vol(
    ewma_daily: pd.DataFrame,
    window_years: int = 10,
    ann_factor: int = 256,
) -> pd.DataFrame:
    """
    Moyenne mobile de la volatilité EWMA sur une fenêtre de 10 ans.

    Quand l'historique disponible est inférieur à 10 ans (début de backtest),
    pandas utilise automatiquement toutes les données disponibles grâce au
    paramètre min_periods. C'est le comportement "expansive" souhaité.

    Convention Carver : on moyenne les σ (pas les σ²).
    Justification : la dispersion des σ est suffisamment faible pour que
    l'écart avec √mean(σ²) soit négligeable (< 3%).

    Paramètres
    ----------
    ewma_daily :
        Volatilité EWMA quotidienne (sortie de ewma_vol()).
    window_years :
        Fenêtre en années (défaut : 10).
    ann_factor :
        Jours de trading par an (défaut : 256).

    Retourne
    --------
    DataFrame de même forme, moyenne roulante de σ_EWMA.
    """
    window = window_years * ann_factor  # 2 560 jours
    return ewma_daily.rolling(window=window, min_periods=252).mean()


# ── Étape 4a : blend 70/30 ────────────────────────────────────────────────────

def blend_vol(
    ewma_daily: pd.DataFrame,
    lt_daily: pd.DataFrame,
    lt_weight: float = 0.30,
) -> pd.DataFrame:
    """
    Mélange la volatilité EWMA avec la volatilité long terme.

    σ_blend_t = (1 - lt_weight) × σ_EWMA_t + lt_weight × σ_LT_t

    Quand σ_LT est NaN (moins de 252 jours d'historique), on utilise
    uniquement σ_EWMA pour ne pas bloquer le début du backtest.

    Paramètres
    ----------
    ewma_daily :
        Volatilité EWMA quotidienne.
    lt_daily :
        Volatilité long terme quotidienne (sortie de long_term_vol()).
    lt_weight :
        Poids de la composante long terme (défaut : 0.30).

    Retourne
    --------
    DataFrame de même forme.
    """
    ewma_weight = 1.0 - lt_weight

    # Quand lt_daily est NaN, replier sur l'EWMA seule
    blended = ewma_weight * ewma_daily + lt_weight * lt_daily
    blended = blended.where(lt_daily.notna(), other=ewma_daily)

    return blended


# ── Étape 4b : plancher de volatilité ────────────────────────────────────────

def vol_floor(
    blended: pd.DataFrame,
    lookback: int = 500,
    percentile: float = 5.0,
) -> pd.DataFrame:
    """
    Applique un plancher à la volatilité blendée.

    Le plancher est le 5e percentile de la volatilité sur les `lookback`
    jours précédents (shift(1) pour éviter le look-ahead).

    σ_final_t = max(σ_blend_t, floor_t)

    Sans ce plancher, une longue période de faible volatilité peut produire
    des positions très grandes qui explosent quand la volatilité revient à
    la normale (ex. fin 2019 → mars 2020).

    Paramètres
    ----------
    blended :
        Volatilité blendée (sortie de blend_vol()).
    lookback :
        Fenêtre du percentile en jours (défaut : 500).
    percentile :
        Percentile utilisé comme plancher (défaut : 5).

    Retourne
    --------
    DataFrame de même forme, σ_final ≥ floor.
    """
    # Percentile roulant sur les lookback jours PRÉCÉDENTS (shift 1 = no look-ahead)
    floor = (
        blended
        .shift(1)
        .rolling(window=lookback, min_periods=100)
        .quantile(percentile / 100.0)
    )

    return blended.where(blended >= floor, other=floor)


# ── Fonction principale ───────────────────────────────────────────────────────

def compute_vol(
    prices_adj: pd.DataFrame,
    span: int = 35,
    lt_weight: float = 0.30,
    lt_window_years: int = 10,
    floor_lookback: int = 500,
    floor_percentile: float = 5.0,
    ann_factor: int = 256,
) -> dict[str, pd.DataFrame]:
    """
    Pipeline complet de volatilité.

    Enchaîne les 4 étapes :
        rendements → ewma → blend → floor

    Paramètres
    ----------
    prices_adj :
        Prix ajustés (sortie de store.load_prices_adj()).
    span, lt_weight, lt_window_years, floor_lookback, floor_percentile :
        Paramètres de chaque étape (valeurs Carver par défaut).
    ann_factor :
        Facteur d'annualisation (256 jours de trading).

    Retourne
    --------
    dict avec deux clés :
        "daily"  : σ quotidienne en décimal (pour normalisation des signaux)
        "annual" : σ annualisée = daily × √ann_factor (pour sizing et carry)
    """
    returns    = daily_returns(prices_adj)
    ewma       = ewma_vol(returns, span=span)
    lt         = long_term_vol(ewma, window_years=lt_window_years, ann_factor=ann_factor)
    blended    = blend_vol(ewma, lt, lt_weight=lt_weight)
    final      = vol_floor(blended, lookback=floor_lookback, percentile=floor_percentile)

    return {
        "daily":  final,
        "annual": final * np.sqrt(ann_factor),
    }
