"""
src/systrade/backtest/engine.py
────────────────────────────────
Moteur de backtest — Bloc 9.

Timing
------
Le signal de la clôture t utilise uniquement des données ≤ t.
La position qui en sort gagne le rendement total de t vers t+1 :

    PnL_{t+1} = N_t × P_brut_t × r_total_{t+1}

Les fichiers de prix n'ont pas d'ouverture. On ne simule donc pas un
passage d'ordre à l'open de t+1 : le trade est porté par le rendement
clôture-à-clôture suivant. C'est le décalage d'un jour, sans look-ahead.

Capital
-------
N* est recalculé chaque soir sur le capital courant, après le P&L du jour.
Le buffer puis le plafond de levier s'appliquent dans cet ordre.
Les coûts (spread × 1,5, emprunt des shorts) sont déduits du capital.

La décomposition par règle relance le même moteur avec le forecast cappé
de cette seule règle. La décomposition par classe attribue le P&L net
du portefeuille ; le Sharpe d'une série c × r est celui de r quand c > 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from systrade.backtest.metrics import EVAL_START, summarize, walk_forward
from systrade.data.store import PROJECT_ROOT
from systrade.forecast.combination import compute_combination
from systrade.forecast.scalar import compute_scaled_capped
from systrade.portfolio.buffer import BUFFER_DELTA
from systrade.portfolio.costs import (
    ANN_FACTOR,
    DEFAULT_SPREAD,
    SLIPPAGE_MULTIPLIER,
    annual_turnover,
    borrow_rates,
    sharpe_cost,
)
from systrade.portfolio.idm import (
    active_weights,
    expanding_idm,
    instrument_weights,
    subsystem_returns,
)
from systrade.portfolio.sizing import FORECAST_SCALE, MAX_LEVERAGE, VOL_TARGET
from systrade.risk.volatility import compute_vol
from systrade.rules.carry import all_carry_signals
from systrade.rules.ewmac import all_ewmac_signals

RULES: tuple[str, ...] = (
    "EWMAC_8_32",
    "EWMAC_16_64",
    "EWMAC_32_128",
    "carry_medium",
    "carry_slow",
)

# Biais de survie : l'univers est choisi en 2026 parmi des ETF encore listés.
SURVIVAL_BIAS = (
    "Univers choisi en 2026 parmi des ETF encore cotés. "
    "Les fonds fermés entre 2007 et 2026 n'y figurent pas."
)


def asset_classes(config_path: Path | None = None) -> dict[str, str]:
    """{ticker: classe} depuis instruments.yaml."""
    path = config_path or (PROJECT_ROOT / "config" / "instruments.yaml")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {
        ticker: str(spec["class"])
        for ticker, spec in config["universe"].items()
    }


def shares_per_capital(
    forecast: pd.DataFrame,
    vol_annual: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    weights: pd.DataFrame,
    idm: pd.Series,
    vol_target: float = VOL_TARGET,
    forecast_scale: float = FORECAST_SCALE,
) -> pd.DataFrame:
    """
    Actions par dollar de capital.

        k = τ × F × w × IDM / (10 × σ_annual × P_brut)

    N* = capital × k. k ne dépend pas du capital.
    """
    cols = (
        forecast.columns
        .intersection(vol_annual.columns)
        .intersection(prices_unadj.columns)
        .intersection(weights.columns)
    )
    vol = vol_annual[cols].replace(0.0, np.nan)
    price = prices_unadj[cols].replace(0.0, np.nan)
    scaled = forecast[cols].mul(weights[cols]).mul(idm.reindex(forecast.index), axis=0)
    return scaled * vol_target / (forecast_scale * vol * price)


def simulate(
    shares_per_dollar: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    total_returns: pd.DataFrame,
    capital0: float = 100_000.0,
    delta: float = BUFFER_DELTA,
    max_leverage: float = MAX_LEVERAGE,
    spread: float = DEFAULT_SPREAD,
    slippage: float = SLIPPAGE_MULTIPLIER,
    rates: dict[str, float] | None = None,
    ann_factor: int = ANN_FACTOR,
) -> dict[str, pd.DataFrame | pd.Series | float]:
    """
    Boucle quotidienne. La position fixée à la clôture t gagne r_{t+1}.

    Un k non fini laisse la position bufferisée inchangée (donnée manquante).
    Un k nul est une cible plate : on solde.
    """
    if rates is None:
        rates = {}

    cols = list(shares_per_dollar.columns)
    dates = shares_per_dollar.index
    k = shares_per_dollar.to_numpy(dtype=float)
    px = prices_unadj.reindex(index=dates, columns=cols).to_numpy(dtype=float)
    rets = total_returns.reindex(index=dates, columns=cols).to_numpy(dtype=float)
    borrow = np.array([rates.get(c, 0.0) for c in cols], dtype=float)

    n_rows, n_cols = k.shape
    capital = np.zeros(n_rows)
    capital_start = np.zeros(n_rows)
    port_ret = np.zeros(n_rows)
    leverage = np.zeros(n_rows)
    turnover = np.zeros(n_rows)
    trade_costs = np.zeros(n_rows)
    borrow_costs = np.zeros(n_rows)
    held_hist = np.zeros((n_rows, n_cols))
    pnl = np.zeros((n_rows, n_cols))

    cap = float(capital0)
    held = np.zeros(n_cols)
    buffered = np.zeros(n_cols)

    for t in range(n_rows):
        cap_start = cap
        capital_start[t] = cap_start
        px_prev = np.where(np.isfinite(px[t - 1]), px[t - 1], 0.0) if t else np.zeros(n_cols)
        r = np.where(np.isfinite(rets[t]), rets[t], 0.0)

        price_pnl = held * px_prev * r
        borrow_pnl = np.maximum(-held, 0.0) * px_prev * borrow / ann_factor
        cap = cap_start + float(price_pnl.sum() - borrow_pnl.sum())

        if cap < 1.0:
            held[:] = 0.0
            buffered[:] = 0.0
            capital[t] = max(cap, 0.0)
            port_ret[t] = capital[t] / cap_start - 1.0 if cap_start else 0.0
            pnl[t] = price_pnl - borrow_pnl
            borrow_costs[t] = float(borrow_pnl.sum())
            continue

        target = cap * k[t]
        valid = np.isfinite(target) & np.isfinite(px[t]) & (px[t] > 0)
        updated = buffered.copy()
        if valid.any():
            width = delta * np.abs(target[valid])
            updated[valid] = np.clip(
                buffered[valid], target[valid] - width, target[valid] + width
            )
        buffered = updated

        px_now = np.where(np.isfinite(px[t]), px[t], 0.0)
        gross = float(np.abs(buffered * px_now).sum())
        lev = gross / cap if cap else 0.0
        scale = min(1.0, max_leverage / lev) if lev > 0 else 1.0
        new_held = buffered * scale

        trades = new_held - held
        cost = np.abs(trades) * px_now * spread * slippage
        cap -= float(cost.sum())

        pnl[t] = price_pnl - borrow_pnl - cost
        held = new_held
        held_hist[t] = held
        capital[t] = cap
        leverage[t] = float(np.abs(held * px_now).sum()) / cap if cap else 0.0
        trade_costs[t] = float(cost.sum())
        borrow_costs[t] = float(borrow_pnl.sum())
        turnover[t] = float((np.abs(trades) * px_now).sum()) / cap_start if cap_start else 0.0
        port_ret[t] = cap / cap_start - 1.0 if cap_start else 0.0

    index = shares_per_dollar.index
    return {
        "returns": pd.Series(port_ret, index=index, name="return"),
        "capital": pd.Series(capital, index=index, name="capital"),
        "capital_start": pd.Series(capital_start, index=index, name="capital_start"),
        "positions": pd.DataFrame(held_hist, index=index, columns=cols),
        "pnl": pd.DataFrame(pnl, index=index, columns=cols),
        "leverage": pd.Series(leverage, index=index, name="leverage"),
        "turnover": pd.Series(turnover, index=index, name="turnover"),
        "trade_costs": pd.Series(trade_costs, index=index, name="trade_costs"),
        "borrow_costs": pd.Series(borrow_costs, index=index, name="borrow_costs"),
        "capital0": capital0,
    }


def backtest_from_forecast(
    forecast: pd.DataFrame,
    prices_adj: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    vol_daily: pd.DataFrame,
    vol_annual: pd.DataFrame,
    weights: dict[str, float] | None = None,
    rates: dict[str, float] | None = None,
    capital0: float = 100_000.0,
    min_obs: int = 252,
    ewma_span: int = 125,
    delta: float = BUFFER_DELTA,
    max_leverage: float = MAX_LEVERAGE,
    spread: float = DEFAULT_SPREAD,
    slippage: float = SLIPPAGE_MULTIPLIER,
) -> dict[str, pd.DataFrame | pd.Series | float]:
    """Forecast combiné (ou forecast d'une seule règle) → chemin de capital."""
    if weights is None:
        weights = instrument_weights()
    if rates is None:
        rates = borrow_rates()

    sub = subsystem_returns(forecast, prices_adj, vol_daily)
    w = active_weights(sub, weights, min_obs=min_obs)
    idm = expanding_idm(sub, weights, min_obs=min_obs, ewma_span=ewma_span)
    k = shares_per_capital(forecast, vol_annual, prices_unadj, w, idm)
    total_returns = prices_adj.reindex(columns=k.columns).pct_change()
    result = simulate(
        k,
        prices_unadj,
        total_returns,
        capital0=capital0,
        delta=delta,
        max_leverage=max_leverage,
        spread=spread,
        slippage=slippage,
        rates=rates,
    )
    result["idm"] = idm
    result["weights"] = w
    return result


def class_contributions(
    pnl: pd.DataFrame,
    capital_start: pd.Series,
    classes: dict[str, str],
) -> pd.DataFrame:
    """Rendement net attribué à chaque classe : P&L de la classe / capital de début de jour."""
    contrib = pnl.div(capital_start.replace(0.0, np.nan), axis=0)
    grouped: dict[str, pd.Series] = {}
    for ticker in contrib.columns:
        name = classes.get(ticker)
        if name is None:
            continue
        grouped[name] = grouped.get(name, 0.0) + contrib[ticker]
    return pd.DataFrame(grouped)


def buy_and_hold(prices_adj: pd.DataFrame, ticker: str) -> pd.Series:
    """Rendement total d'un ETF, sans coût."""
    return prices_adj[ticker].pct_change().rename(ticker)


def sixty_forty(prices_adj: pd.DataFrame, equity: str = "SPY", bond: str = "IEF") -> pd.Series:
    """60/40 rééquilibré chaque jour, sans coût."""
    rets = prices_adj[[equity, bond]].pct_change()
    return (0.60 * rets[equity] + 0.40 * rets[bond]).rename("60/40")


def long_only_forecast(prices_adj: pd.DataFrame, vol_annual: pd.DataFrame) -> pd.DataFrame:
    """Forecast constant +10 là où le prix et la volatilité existent."""
    level = pd.DataFrame(FORECAST_SCALE, index=prices_adj.index, columns=prices_adj.columns)
    return level.where(prices_adj.notna() & vol_annual.reindex_like(prices_adj).notna())


def _window(result: dict, start: str) -> dict:
    """Sous-ensemble des séries à partir de `start`, pour les métriques."""
    returns = result["returns"].loc[start:]
    leverage = result["leverage"].loc[start:]
    turnover = annual_turnover(result["turnover"].loc[start:])
    return {"returns": returns, "leverage": leverage, "turnover_annual": turnover}


def _stats(result: dict, start: str) -> dict[str, float]:
    window = _window(result, start)
    stats = summarize(
        window["returns"],
        leverage=window["leverage"],
        turnover_annual=window["turnover_annual"],
    )
    vol = stats["volatility"]
    if np.isfinite(vol) and vol > 0:
        stats["sharpe_cost"] = sharpe_cost(
            DEFAULT_SPREAD * SLIPPAGE_MULTIPLIER,
            window["turnover_annual"],
            vol,
        )
    return stats


def assemble_book(
    prices_adj: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    dividends: pd.DataFrame,
    rates: pd.DataFrame,
) -> dict:
    """
    Signaux, poids, IDM et actions par dollar de capital.

    C'est le calcul commun au backtest et à la boucle paper. `k` ne dépend
    pas du capital : N* = capital courant × k, puis buffer et levier.
    """
    vol = compute_vol(prices_adj)
    ewmac = all_ewmac_signals(prices_adj, vol["daily"])
    carry = all_carry_signals(dividends, prices_unadj, rates, vol["annual"])
    capped = {
        name: compute_scaled_capped(signal)["capped"]
        for name, signal in {**ewmac, **carry}.items()
    }
    forecast = compute_combination(capped)["combined"]
    weights = instrument_weights()
    sub = subsystem_returns(forecast, prices_adj, vol["daily"])
    w = active_weights(sub, weights)
    idm = expanding_idm(sub, weights)
    k = shares_per_capital(forecast, vol["annual"], prices_unadj, w, idm)
    return {
        "vol": vol,
        "capped": capped,
        "forecast": forecast,
        "weights": w,
        "idm": idm,
        "shares_per_dollar": k,
        "borrow": borrow_rates(),
        "classes": asset_classes(),
    }


def run_backtest(
    prices_adj: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    dividends: pd.DataFrame,
    rates: pd.DataFrame,
    capital0: float = 100_000.0,
    eval_start: str = EVAL_START,
    decompose: bool = True,
) -> dict:
    """
    Pipeline complet : signaux → portefeuille → métriques, benchmarks, décompositions.

    `decompose=False` ne calcule que le livre principal. Utile pour les tests.
    """
    assembled = assemble_book(prices_adj, prices_unadj, dividends, rates)
    vol = assembled["vol"]
    forecast = assembled["forecast"]
    capped = assembled["capped"]

    weights = instrument_weights()
    borrow = assembled["borrow"]
    classes = assembled["classes"]
    common = dict(
        prices_adj=prices_adj,
        prices_unadj=prices_unadj,
        vol_daily=vol["daily"],
        vol_annual=vol["annual"],
        weights=weights,
        rates=borrow,
        capital0=capital0,
    )

    cols = assembled["shares_per_dollar"].columns
    strategy = simulate(
        assembled["shares_per_dollar"],
        prices_unadj,
        prices_adj.reindex(columns=cols).pct_change(),
        capital0=capital0,
        rates=borrow,
    )
    strategy["idm"] = assembled["idm"]
    strategy["weights"] = assembled["weights"]
    strategy_stats = _stats(strategy, eval_start)

    bench_returns = {
        "SPY": buy_and_hold(prices_adj, "SPY").loc[eval_start:],
        "60/40": sixty_forty(prices_adj).loc[eval_start:],
    }
    long_only = backtest_from_forecast(long_only_forecast(prices_adj, vol["annual"]), **common)
    benchmarks = {
        name: summarize(series) for name, series in bench_returns.items()
    }
    benchmarks["long-only vol target"] = _stats(long_only, eval_start)

    contrib = class_contributions(
        strategy["pnl"], strategy["capital_start"], classes
    ).loc[eval_start:]
    by_rule: dict[str, dict[str, float]] = {}
    by_class = {
        name: summarize(contrib[name]) for name in contrib.columns
    }
    if decompose:
        for rule in RULES:
            if rule not in capped:
                continue
            rule_book = backtest_from_forecast(capped[rule], **common)
            by_rule[rule] = _stats(rule_book, eval_start)

    evaluated = strategy["returns"].loc[eval_start:]
    curves = {
        "strategy": evaluated,
        "SPY": bench_returns["SPY"],
        "60/40": bench_returns["60/40"],
        "long-only vol target": long_only["returns"].loc[eval_start:],
    }
    return {
        "strategy": strategy,
        "forecast": forecast,
        "metrics": strategy_stats,
        "walk_forward": walk_forward(evaluated),
        "benchmarks": benchmarks,
        "by_rule": by_rule,
        "by_class": by_class,
        "curves": curves,
        "class_returns": contrib,
        "survival_bias": SURVIVAL_BIAS,
        "eval_start": eval_start,
    }
