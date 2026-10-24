"""
src/systrade/backtest/metrics.py
─────────────────────────────────
Métriques du bloc 9.

Sharpe annualisé : √256 × μ / σ, écart-type échantillon (ddof=1).

Le Calmar de ce projet est Sharpe / |max drawdown|, comme validé dans
l'architecture. Ce n'est pas le Calmar usuel (CAGR / |max drawdown|).
Le CAGR est reporté à côté.

Walk-forward, sans réestimation d'aucun paramètre :

    P1  2009 → 2012   sortie de crise (la chauffe couvre 2007-2008)
    P2  2012 → 2017   marché haussier
    P3  2017 → 2022   COVID et reprise
    P4  2022 → fin    inflation et hausse des taux

Les métriques commencent en 2009 : deux ans de chauffe pour l'EWMA,
les scalaires et les corrélations. 2008 n'entre donc pas dans P1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from systrade.portfolio.costs import ANN_FACTOR

EVAL_START = "2009-01-01"

# (nom, début inclus, fin inclus)
WALK_FORWARD: tuple[tuple[str, str, str], ...] = (
    ("P1 2009-2012", "2009-01-01", "2011-12-31"),
    ("P2 2012-2017", "2012-01-01", "2016-12-31"),
    ("P3 2017-2022", "2017-01-01", "2021-12-31"),
    ("P4 2022-2026", "2022-01-01", "2026-12-31"),
)


def sharpe_ratio(returns: pd.Series, ann_factor: int = ANN_FACTOR) -> float:
    """√256 × μ / σ. NaN s'il n'y a pas assez de jours ou si σ = 0."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sigma = float(r.std(ddof=1))
    if sigma == 0.0 or not np.isfinite(sigma):
        return float("nan")
    return float(np.sqrt(ann_factor) * r.mean() / sigma)


def equity_curve(returns: pd.Series) -> pd.Series:
    """Capital d'un euro, composé quotidiennement."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    """Pire perte depuis un sommet. Négatif ou nul."""
    equity = equity_curve(returns)
    drawdown = equity / equity.cummax() - 1.0
    if drawdown.empty:
        return float("nan")
    return float(drawdown.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown courant, en fraction du dernier sommet."""
    equity = equity_curve(returns)
    return equity / equity.cummax() - 1.0


def cagr(returns: pd.Series, ann_factor: int = ANN_FACTOR) -> float:
    """Taux de croissance annuel composé."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    growth = float((1.0 + r).prod())
    years = len(r) / ann_factor
    if years <= 0 or growth <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def annual_volatility(returns: pd.Series, ann_factor: int = ANN_FACTOR) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(ann_factor))


def skewness(returns: pd.Series) -> float:
    """Skewness de Fisher (biais corrigé), comme pandas."""
    r = returns.dropna()
    if len(r) < 3:
        return float("nan")
    return float(r.skew())


def hit_rate(returns: pd.Series) -> float:
    """Part des jours strictement positifs."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def calmar_ratio(returns: pd.Series, ann_factor: int = ANN_FACTOR) -> float:
    """Sharpe / |max drawdown|, définition validée au bloc 9."""
    dd = max_drawdown(returns)
    sharpe = sharpe_ratio(returns, ann_factor)
    if not np.isfinite(dd) or dd == 0.0 or not np.isfinite(sharpe):
        return float("nan")
    return float(sharpe / abs(dd))


def summarize(
    returns: pd.Series,
    ann_factor: int = ANN_FACTOR,
    leverage: pd.Series | None = None,
    turnover_annual: float | None = None,
) -> dict[str, float]:
    """Métriques primaires et secondaires sur une série de rendements nets."""
    stats = {
        "sharpe": sharpe_ratio(returns, ann_factor),
        "cagr": cagr(returns, ann_factor),
        "volatility": annual_volatility(returns, ann_factor),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, ann_factor),
        "skewness": skewness(returns),
        "hit_rate": hit_rate(returns),
        "n_days": float(returns.dropna().shape[0]),
    }
    if leverage is not None and len(leverage.dropna()):
        stats["leverage_mean"] = float(leverage.mean())
        stats["leverage_max"] = float(leverage.max())
    if turnover_annual is not None:
        stats["turnover_annual"] = float(turnover_annual)
    return stats


def rolling_sharpe(
    returns: pd.Series,
    window: int = ANN_FACTOR * 3,
    ann_factor: int = ANN_FACTOR,
) -> pd.Series:
    """Sharpe annualisé sur une fenêtre glissante. Défaut : 3 ans de 256 jours."""
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std(ddof=1)
    return (np.sqrt(ann_factor) * mu / sigma).rename("rolling_sharpe")


def walk_forward(
    returns: pd.Series,
    periods: tuple[tuple[str, str, str], ...] = WALK_FORWARD,
    ann_factor: int = ANN_FACTOR,
) -> pd.DataFrame:
    """Sharpe, CAGR et max drawdown sur chaque sous-période. Aucun paramètre refité."""
    rows = []
    for name, start, end in periods:
        window = returns.loc[start:end]
        rows.append({
            "period": name,
            "sharpe": sharpe_ratio(window, ann_factor),
            "cagr": cagr(window, ann_factor),
            "max_drawdown": max_drawdown(window),
            "n_days": int(window.dropna().shape[0]),
        })
    return pd.DataFrame(rows).set_index("period")


def block_bootstrap_sharpe_difference(
    left: pd.Series,
    right: pd.Series,
    block: int = 21,
    n_draws: int = 2000,
    seed: int = 0,
    ann_factor: int = ANN_FACTOR,
) -> dict[str, float]:
    """
    Écart de Sharpe annualisé, left − right, et son intervalle à 95 %.

    Les deux séries sont alignées sur les dates communes, puis retirées
    ensemble par blocs circulaires. L'appariement des jours est conservé.
    `seed` fixe les tirages. Le résultat du papier est block=21 et block=63,
    n_draws=2000, seed=0, sur les rendements excédentaires.
    """
    if block < 1:
        raise ValueError(f"block doit être >= 1, reçu {block}")
    paired = pd.concat(
        [left.rename("left"), right.rename("right")], axis=1
    ).dropna()
    y = paired["left"].to_numpy(dtype=float)
    x = paired["right"].to_numpy(dtype=float)
    nobs = len(y)
    if nobs < 2:
        raise ValueError("il faut au moins deux jours communs")

    def _sharpe(sample: np.ndarray) -> float:
        sigma = float(sample.std(ddof=1))
        if sigma == 0.0 or not np.isfinite(sigma):
            return float("nan")
        return float(np.sqrt(ann_factor) * sample.mean() / sigma)

    observed = _sharpe(y) - _sharpe(x)
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(nobs / block))
    diffs = np.empty(n_draws)
    for i in range(n_draws):
        starts = rng.integers(0, nobs, size=nblocks)
        take = np.concatenate([(np.arange(block) + s) % nobs for s in starts])[:nobs]
        diffs[i] = _sharpe(y[take]) - _sharpe(x[take])
    finite = diffs[np.isfinite(diffs)]
    if len(finite) == 0:
        low = high = fraction = float("nan")
    else:
        low, high = np.quantile(finite, [0.025, 0.975])
        fraction = float(np.mean(finite >= 0.0))
    return {
        "difference": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "fraction_nonnegative": float(fraction),
        "n_days": float(nobs),
        "block": float(block),
        "n_draws": float(n_draws),
    }
