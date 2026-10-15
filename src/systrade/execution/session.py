"""
src/systrade/execution/session.py
──────────────────────────────────
Décision d'une séance, sans réseau et sans base.

Même ordre que le backtest : N* sur le capital courant, buffer
(mémoire d'avant levier), plafond 2×, puis arrondi à l'action entière.
Le risque live s'applique ensuite sur l'écart avec le courtier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from systrade.execution.risk import apply_risk
from systrade.execution.types import Order
from systrade.portfolio.buffer import BUFFER_DELTA, apply_buffer
from systrade.portfolio.sizing import (
    MAX_LEVERAGE,
    apply_leverage_constraint,
    gross_leverage,
)


class IncompleteMarket(Exception):
    """Un prix de l'univers manque. On ne touche ni au buffer ni aux ordres."""

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers
        super().__init__("prix manquants : " + ", ".join(tickers))


@dataclass
class ClosePlan:
    as_of: pd.Timestamp
    equity: float
    last_equity: float | None
    day_pnl: float | None
    leverage: float
    vol_upper: float
    idm: float
    circuit_breaker: bool
    alerts: list[str]
    targets: list[dict]
    orders: list[Order]
    blocked: list[Order]
    buffered: pd.Series = field(repr=False)


def round_shares(value: float) -> int:
    """Arrondi half-away-from-zero. 0,5 action part vers l'extérieur, pas vers l'entier pair."""
    if not np.isfinite(value):
        return 0
    return int(np.sign(value) * np.floor(abs(value) + 0.5))


def _prices_ok(prices: pd.Series) -> list[str]:
    missing = []
    for ticker, price in prices.items():
        if not np.isfinite(price) or price <= 0.0:
            missing.append(str(ticker))
    return missing


def _n_star(shares_per_dollar: pd.Series, equity: float) -> pd.Series:
    return shares_per_dollar.astype(float) * float(equity)


def _step_buffer(n_star: pd.Series, buffered: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    frame = pd.DataFrame([n_star.to_numpy(dtype=float)], index=[as_of], columns=list(n_star.index))
    initial = None
    if len(buffered):
        initial = buffered.reindex(n_star.index).fillna(0.0)
    return apply_buffer(frame, delta=BUFFER_DELTA, initial=initial).iloc[0]


def _apply_cap(
    buffered: pd.Series,
    prices: pd.Series,
    equity: float,
    as_of: pd.Timestamp,
) -> pd.Series:
    cols = list(buffered.index)
    frame = pd.DataFrame([buffered.to_numpy(dtype=float)], index=[as_of], columns=cols)
    px = pd.DataFrame([prices.reindex(cols).to_numpy(dtype=float)], index=[as_of], columns=cols)
    return apply_leverage_constraint(frame, px, equity, max_leverage=MAX_LEVERAGE).iloc[0]


def _shave(qty: dict[str, int], prices: pd.Series, equity: float) -> dict[str, int]:
    """Ramène le livre entier sous 2× en enlevant une action au plus gros notionnel."""
    if equity <= 0.0:
        return {ticker: 0 for ticker in qty}

    def leverage() -> float:
        gross = sum(abs(shares) * float(prices[ticker]) for ticker, shares in qty.items())
        return gross / equity

    guard = sum(abs(shares) for shares in qty.values()) + 1
    while leverage() > MAX_LEVERAGE + 1e-9 and guard > 0:
        guard -= 1
        ticker = max(qty, key=lambda name: abs(qty[name]) * float(prices[name]))
        if qty[ticker] == 0:
            break
        qty[ticker] -= int(np.sign(qty[ticker]))
    return {ticker: shares for ticker, shares in qty.items() if shares != 0}


def integer_book(desired: pd.Series, prices: pd.Series, equity: float) -> dict[str, int]:
    qty = {
        str(ticker): round_shares(float(value))
        for ticker, value in desired.items()
        if np.isfinite(value)
    }
    return _shave(qty, prices, equity)


def diff_orders(desired: dict[str, int], held: dict[str, int]) -> list[Order]:
    """Écart entre le livre cible entier et le courtier. Achat si la cible est au-dessus."""
    orders: list[Order] = []
    for ticker in sorted(set(desired) | set(held)):
        delta = int(desired.get(ticker, 0)) - int(held.get(ticker, 0))
        if delta == 0:
            continue
        orders.append(
            Order(
                ticker=ticker,
                side="buy" if delta > 0 else "sell",
                qty=abs(delta),
            )
        )
    return orders


def _vol_upper(held_float: pd.Series, prices: pd.Series, vol: pd.Series, equity: float) -> float:
    """Borne de vol si toutes les positions étaient parfaitement corrélées."""
    if equity <= 0.0:
        return float("nan")
    total = 0.0
    for ticker, shares in held_float.items():
        price = prices.get(ticker, np.nan)
        sigma = vol.get(ticker, np.nan)
        if not (np.isfinite(shares) and np.isfinite(price) and np.isfinite(sigma)):
            continue
        total += abs(float(shares) * float(price)) * float(sigma)
    return total / equity


def build_close_plan(
    *,
    as_of: pd.Timestamp,
    shares_per_dollar: pd.Series,
    prices: pd.Series,
    forecast: pd.Series,
    vol_annual: pd.Series,
    weight: pd.Series,
    idm: float,
    equity: float,
    last_equity: float | None,
    held: dict[str, int],
    buffered: pd.Series,
    adv: dict[str, float] | None = None,
) -> ClosePlan:
    """
    Plan de clôture. Lève IncompleteMarket si un prix manque.

    Le buffer renvoyé est celui d'avant levier, à persister même si le
    risque bloque les ordres : le lendemain, l'écart avec le courtier
    refera l'ordre.
    """
    prices = prices.reindex(shares_per_dollar.index)
    missing = _prices_ok(prices)
    if missing:
        raise IncompleteMarket(missing)

    as_of = pd.Timestamp(as_of)
    n_star = _n_star(shares_per_dollar, equity)
    new_buffer = _step_buffer(n_star, buffered, as_of)
    desired_float = _apply_cap(new_buffer, prices, equity, as_of)
    desired = integer_book(desired_float, prices, equity)

    universe = {str(ticker) for ticker in shares_per_dollar.index}
    held_book = {ticker: qty for ticker, qty in held.items() if ticker in universe}
    raw_orders = diff_orders(desired, held_book)
    price_map = {str(k): float(v) for k, v in prices.items()}
    decision = apply_risk(raw_orders, held, price_map, equity, last_equity, adv)
    allowed = {(order.ticker, order.side) for order in decision.orders}
    blocked = [order for order in raw_orders if (order.ticker, order.side) not in allowed]
    # Une réduction d'ADV change la quantité : on journalise l'ordre gardé, pas l'original.
    kept = list(decision.orders)

    frame = pd.DataFrame(
        [desired_float.to_numpy(dtype=float)],
        index=[as_of],
        columns=list(desired_float.index),
    )
    px = pd.DataFrame(
        [prices.reindex(desired_float.index).to_numpy(dtype=float)],
        index=[as_of],
        columns=list(desired_float.index),
    )
    leverage = float(gross_leverage(frame, px, equity).iloc[0])
    pnl = None if last_equity is None or last_equity <= 0 else equity / last_equity - 1.0

    targets = []
    for ticker in shares_per_dollar.index:
        name = str(ticker)
        targets.append(
            {
                "ticker": name,
                "forecast": _finite(forecast.get(ticker, np.nan)),
                "vol_annual": _finite(vol_annual.get(ticker, np.nan)),
                "weight": _finite(weight.get(ticker, np.nan)),
                "idm": float(idm) if np.isfinite(idm) else None,
                "n_star": _finite(n_star.get(ticker, np.nan)),
                "buffered": _finite(new_buffer.get(ticker, np.nan)),
                "desired": int(desired.get(name, 0)),
                "held": int(held.get(name, 0)),
                "price": float(prices[ticker]),
            }
        )

    alerts = list(decision.alerts)
    unknown = sorted(set(held) - set(map(str, shares_per_dollar.index)))
    if unknown:
        alerts.append("positions hors univers : " + ", ".join(unknown))

    return ClosePlan(
        as_of=as_of,
        equity=float(equity),
        last_equity=None if last_equity is None else float(last_equity),
        day_pnl=pnl,
        leverage=leverage,
        vol_upper=_vol_upper(desired_float, prices, vol_annual, equity),
        idm=float(idm) if np.isfinite(idm) else float("nan"),
        circuit_breaker=decision.circuit_breaker,
        alerts=alerts,
        targets=targets,
        orders=kept,
        blocked=[
            Order(ticker=order.ticker, side=order.side, qty=order.qty, status="blocked")
            for order in blocked
        ],
        buffered=new_buffer,
    )


def _finite(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)
