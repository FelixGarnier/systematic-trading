"""
src/systrade/portfolio/costs.py
────────────────────────────────
Coûts de transaction et d'emprunt — Bloc 8.

Trois postes :

    spread     coût d'un trade = |ΔN| × prix × spread × slippage
    commission 0 $ chez Alpaca, paramétrable
    borrow     taux annuel × notionnel short / 256, longs exonérés

Le spread V1 est une hypothèse plate de 2 bp (bid-ask / prix). Les ETF du
livre sont liquides ; le chiffre sera remplacé par le bid/ask Alpaca en live.
Le slippage de backtest (bloc 9) est 1,5× ce spread.

Coût en unités de Sharpe (speed limit de Carver) :

    SR_coût = spread × turnover_annuel / σ

    turnover_annuel = moyenne( |ΔN| × prix / capital ) × 256

La règle reste acceptable si SR_coût ≤ SR_attendu / 3.
Les dividendes payés sur les shorts ne sont pas ici : ils sont dans le
rendement total, au bloc 9.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from systrade.data.store import PROJECT_ROOT

# Spread bid-ask / prix. Hypothèse plate V1, pas une estimation historique.
DEFAULT_SPREAD: float = 0.0002
SLIPPAGE_MULTIPLIER: float = 1.5
COMMISSION: float = 0.0
ANN_FACTOR: int = 256
SPEED_LIMIT_FRACTION: float = 1.0 / 3.0


def borrow_rates(config_path: Path | None = None) -> dict[str, float]:
    """Taux d'emprunt annuel par instrument, lus dans instruments.yaml."""
    path = config_path or (PROJECT_ROOT / "config" / "instruments.yaml")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {
        ticker: float(spec["borrow_rate"])
        for ticker, spec in config["universe"].items()
    }


def borrow_cost(
    positions: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    rates: dict[str, float] | None = None,
    ann_factor: int = ANN_FACTOR,
) -> pd.DataFrame:
    """
    Coût d'emprunt quotidien en dollars. Seulement les positions courtes.

        coût_t = max(−N_t, 0) × P_t × taux_annuel / 256
    """
    if rates is None:
        rates = borrow_rates()

    cols = [c for c in positions.columns if c in rates]
    prices = prices_unadj.reindex(index=positions.index, columns=cols)
    short_shares = (-positions[cols]).clip(lower=0.0)
    rate_row = pd.Series({c: rates[c] for c in cols})
    return short_shares * prices * rate_row / ann_factor


def trade_cost(
    trades: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    spread: float = DEFAULT_SPREAD,
    slippage: float = SLIPPAGE_MULTIPLIER,
    commission: float = COMMISSION,
) -> pd.DataFrame:
    """
    Coût de transaction en dollars, par instrument et par jour.

        coût = |ΔN| × prix × spread × slippage  +  commission si ΔN ≠ 0
    """
    prices = prices_unadj.reindex(index=trades.index, columns=trades.columns)
    notional_cost = trades.abs() * prices * spread * slippage
    commissions = (trades != 0).astype(float) * commission
    return notional_cost + commissions


def daily_turnover(
    trades: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    capital: float | pd.Series,
) -> pd.Series:
    """Turnover du jour : Σ |ΔN| × prix / capital."""
    prices = prices_unadj.reindex(index=trades.index, columns=trades.columns)
    traded = (trades.abs() * prices).sum(axis=1)
    if isinstance(capital, pd.Series):
        cap = capital.reindex(trades.index)
    else:
        cap = pd.Series(float(capital), index=trades.index)
    return traded / cap.replace(0.0, np.nan)


def annual_turnover(daily: pd.Series, ann_factor: int = ANN_FACTOR) -> float:
    """Turnover annuel = moyenne des turnovers quotidiens × 256."""
    return float(daily.mean(skipna=True) * ann_factor)


def sharpe_cost(spread: float, turnover_annual: float, vol_annual: float) -> float:
    """SR_coût = spread × turnover_annuel / σ."""
    if vol_annual <= 0:
        raise ValueError(f"vol_annual doit être > 0, reçu {vol_annual}")
    return spread * turnover_annual / vol_annual


def within_speed_limit(
    cost_in_sharpe: float,
    expected_sharpe: float,
    fraction: float = SPEED_LIMIT_FRACTION,
) -> bool:
    """True si les coûts restent sous fraction × Sharpe attendu (défaut 1/3)."""
    return cost_in_sharpe <= expected_sharpe * fraction
