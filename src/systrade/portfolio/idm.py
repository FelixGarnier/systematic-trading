"""
src/systrade/portfolio/idm.py
──────────────────────────────
Portefeuille : poids handcraftés et IDM — Bloc 7.

Poids (handcrafting, pas Markowitz) — lus dans config/instruments.yaml :

    Actions US            20 %   SPY, IWM                 10 % chacun
    Actions internationales 20 % VGK, EWJ, EEM, EPP, EWC  4 % chacun
    Obligations           25 %   IEF, TLT, TIP, LQD       6,25 % chacun
    Matières premières    25 %   GLD, DBC                 12,5 % chacun
    Devises               10 %   UUP                      10 %

IDM (même formule que le FDM, au niveau du portefeuille) :

    IDM_t = clip( 1 / √(w' Σ_t w) , 1, 2.5 )

    Σ_t = corrélation des rendements de sous-systèmes, fenêtre expansive.
    ρ < 0 → 0 avant le calcul (évite un IDM artificiellement élevé).
    Lissage EWMA span 125.

Rendement de sous-système (ce que le signal a réellement gagné) :

    r_sub_i,t = (F_i,{t-1} / 10) × r_i,t / σ_daily_i,{t-1}

Position finale, avant buffer et avant contrainte de levier :

    N_final_i,t = N*_i,t × w_i,t × IDM_t

Tant qu'un instrument n'a pas `min_obs` rendements de sous-système, il est
hors du portefeuille. Les poids des instruments actifs sont alors renormalisés
pour sommer à 1. Une fois les 14 instruments actifs, w_i,t = poids handcrafté.

Anti look-ahead : F et σ sont décalés d'un jour ; la corrélation à t n'utilise
que les rendements de sous-système ≤ t.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from systrade.data.store import PROJECT_ROOT
from systrade.portfolio.sizing import FORECAST_SCALE

# ── Constantes ────────────────────────────────────────────────────────────────

IDM_CAP: float = 2.5
IDM_MIN_OBS: int = 252       # ~1 an de recouvrement avant d'estimer Σ
IDM_EWMA_SPAN: int = 125     # ~6 mois, identique au lissage du FDM


# ── Poids ─────────────────────────────────────────────────────────────────────

def instrument_weights(config_path: Path | None = None) -> dict[str, float]:
    """
    Poids instrument handcraftés, dans l'ordre du fichier de config.

    Retourne {ticker: poids} sommant à 1 sur l'univers complet.
    """
    path = config_path or (PROJECT_ROOT / "config" / "instruments.yaml")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {
        ticker: float(spec["instrument_weight"])
        for ticker, spec in config["universe"].items()
    }


def active_weights(
    subsystem_returns: pd.DataFrame,
    weights: dict[str, float],
    min_obs: int = IDM_MIN_OBS,
) -> pd.DataFrame:
    """
    Poids du jour : instruments avec assez d'historique, renormalisés à 1.

    Un instrument absent de `weights` est ignoré. Avant `min_obs` observations
    valides, son poids est 0.
    """
    cols = [c for c in subsystem_returns.columns if c in weights]
    counts = subsystem_returns[cols].notna().cumsum()
    raw = pd.DataFrame(
        {c: np.where(counts[c] >= min_obs, weights[c], 0.0) for c in cols},
        index=subsystem_returns.index,
    )
    total = raw.sum(axis=1).replace(0.0, np.nan)
    return raw.div(total, axis=0).fillna(0.0)


# ── IDM analytique ────────────────────────────────────────────────────────────

def idm_from_corr(
    weights: dict[str, float],
    corr: np.ndarray,
    names: list[str],
) -> float:
    """
    IDM = min(1 / √(w'Σw), IDM_CAP), avec w renormalisé sur `names`.

    Les corrélations négatives sont mises à 0. La diagonale reste 1.
    """
    w = np.array([weights[n] for n in names], dtype=float)
    total = float(w.sum())
    if total <= 0.0 or len(names) == 0:
        return 1.0
    w = w / total

    corr_clipped = np.where(corr < 0.0, 0.0, corr).astype(float)
    np.fill_diagonal(corr_clipped, 1.0)

    w_sw = float(w @ corr_clipped @ w)
    if w_sw <= 0.0:
        return 1.0
    return float(min(1.0 / np.sqrt(w_sw), IDM_CAP))


# ── Rendements de sous-systèmes ───────────────────────────────────────────────

def subsystem_returns(
    forecast: pd.DataFrame,
    prices_adj: pd.DataFrame,
    vol_daily: pd.DataFrame,
    forecast_scale: float = FORECAST_SCALE,
) -> pd.DataFrame:
    """
    r_sub_t = (F_{t-1} / 10) × r_t / σ_daily_{t-1}.

    Le décalage d'un jour sur F et σ garantit que le rendement du jour t
    n'entre pas dans le signal qui l'a « produit ».
    """
    cols = forecast.columns.intersection(prices_adj.columns).intersection(vol_daily.columns)
    rets = prices_adj[cols].pct_change()
    lagged_forecast = forecast[cols].shift(1) / forecast_scale
    lagged_vol = vol_daily[cols].shift(1).replace(0.0, np.nan)
    return lagged_forecast * rets / lagged_vol


# ── IDM expansif ──────────────────────────────────────────────────────────────

def _pairwise_cumstats(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Statistiques cumulatives par paire d'instruments.

    values : (T, K), NaN déjà remplacés par 0 là où valid est False.
    Chaque sortie a la forme (T, K, K). L'entrée [t, i, j] n'agrège que les
    dates ≤ t où i ET j sont observés.
    """
    both = valid[:, :, None] & valid[:, None, :]
    both_f = both.astype(np.float64)
    xi = values[:, :, None]
    xj = values[:, None, :]
    return (
        np.cumsum(both_f, axis=0),
        np.cumsum(np.where(both, xi, 0.0), axis=0),
        np.cumsum(np.where(both, xj, 0.0), axis=0),
        np.cumsum(np.where(both, xi * xj, 0.0), axis=0),
        np.cumsum(np.where(both, xi * xi, 0.0), axis=0),
        np.cumsum(np.where(both, xj * xj, 0.0), axis=0),
    )


def _corr_at(
    t: int,
    idx: np.ndarray,
    cum_n: np.ndarray,
    cum_i: np.ndarray,
    cum_j: np.ndarray,
    cum_ij: np.ndarray,
    cum_i2: np.ndarray,
    cum_j2: np.ndarray,
    min_obs: int,
) -> np.ndarray:
    """Matrice de corrélation des instruments `idx` à la date t."""
    sl = np.ix_(idx, idx)
    n = cum_n[t][sl]
    mean_i = np.divide(cum_i[t][sl], n, out=np.zeros_like(n), where=n > 0)
    mean_j = np.divide(cum_j[t][sl], n, out=np.zeros_like(n), where=n > 0)
    var_i = np.divide(cum_i2[t][sl], n, out=np.zeros_like(n), where=n > 0) - mean_i ** 2
    var_j = np.divide(cum_j2[t][sl], n, out=np.zeros_like(n), where=n > 0) - mean_j ** 2
    cov = np.divide(cum_ij[t][sl], n, out=np.zeros_like(n), where=n > 0) - mean_i * mean_j

    std_i = np.sqrt(np.maximum(var_i, 0.0))
    std_j = np.sqrt(np.maximum(var_j, 0.0))
    denom = std_i * std_j
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-12)
    corr = np.where(n >= min_obs, corr, 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def expanding_idm(
    sub_returns: pd.DataFrame,
    weights: dict[str, float],
    min_obs: int = IDM_MIN_OBS,
    ewma_span: int = IDM_EWMA_SPAN,
) -> pd.Series:
    """
    Série d'IDM. Vaut 1 tant que moins de deux instruments sont estimables,
    puis EWMA du 1/√(w'Σw) calculé sur le passé uniquement.
    """
    cols = [c for c in sub_returns.columns if c in weights]
    frame = sub_returns[cols]
    values = frame.to_numpy(dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    cum_n, cum_i, cum_j, cum_ij, cum_i2, cum_j2 = _pairwise_cumstats(filled, valid)

    raw = np.ones(len(frame), dtype=float)
    names = list(frame.columns)
    fixed = {name: weights[name] for name in names}

    for t in range(len(frame)):
        active = np.flatnonzero(np.diag(cum_n[t]) >= min_obs)
        if len(active) < 2:
            continue
        corr = _corr_at(t, active, cum_n, cum_i, cum_j, cum_ij, cum_i2, cum_j2, min_obs)
        active_names = [names[i] for i in active]
        raw[t] = idm_from_corr(fixed, corr, active_names)

    series = pd.Series(raw, index=frame.index, name="idm")
    return series.ewm(span=ewma_span, min_periods=1).mean()


# ── Application aux positions ─────────────────────────────────────────────────

def apply_idm(
    positions: pd.DataFrame,
    weights: pd.DataFrame,
    idm: pd.Series,
) -> pd.DataFrame:
    """N_final = N* × w_t × IDM_t, aligné sur les colonnes de `weights`."""
    cols = [c for c in weights.columns if c in positions.columns]
    scaled = positions[cols].mul(weights[cols], axis=0)
    return scaled.mul(idm.reindex(scaled.index), axis=0)


def compute_idm(
    positions_target: pd.DataFrame,
    forecast: pd.DataFrame,
    prices_adj: pd.DataFrame,
    vol_daily: pd.DataFrame,
    weights: dict[str, float] | None = None,
    min_obs: int = IDM_MIN_OBS,
    ewma_span: int = IDM_EWMA_SPAN,
) -> dict[str, pd.DataFrame | pd.Series | dict[str, float]]:
    """
    Pipeline : N* → poids handcraftés → IDM → N_final.

    `positions_target` est la position avant poids et avant contrainte de levier
    (sortie `positions_target` de sizing, pas la version déjà rescalée).
    Le buffer (bloc 8) puis le levier s'appliquent ensuite.

    Returns
    -------
    positions          : N_final
    idm                : série IDM lissée
    weights            : poids du jour (renormalisés pendant la montée en charge)
    weights_fixed      : poids handcraftés
    subsystem_returns  : rendements utilisés pour Σ
    """
    if weights is None:
        weights = instrument_weights()

    sub = subsystem_returns(forecast, prices_adj, vol_daily)
    w_t = active_weights(sub, weights, min_obs=min_obs)
    idm = expanding_idm(sub, weights, min_obs=min_obs, ewma_span=ewma_span)
    positions = apply_idm(positions_target, w_t, idm)

    return {
        "positions": positions,
        "idm": idm,
        "weights": w_t,
        "weights_fixed": weights,
        "subsystem_returns": sub,
    }
