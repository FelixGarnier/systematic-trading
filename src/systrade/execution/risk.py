"""
src/systrade/execution/risk.py
────────────────────────────────
Contrôles live, plus stricts que le plafond du backtest.

Le backtest autorise 2×. En paper, trois gardes s'ajoutent :

- circuit breaker : P&L du jour < −5 % du capital → aucun ordre
- levier préventif : si le levier détenu dépasse 1,5×, on bloque
  les ordres qui augmentent l'exposition, on laisse passer les réductions
- liquidité : un ordre ne dépasse pas 1 % de l'ADV

Le seuil d'ADV n'était pas chiffré dans l'architecture. 1 % ne contraint
pas ces ETF, et il arrête un ordre aberrant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from systrade.execution.types import Order

PREVENTIVE_LEVERAGE = 1.5
CIRCUIT_BREAKER = -0.05
ADV_FRACTION = 0.01


@dataclass
class RiskDecision:
    orders: list[Order]
    circuit_breaker: bool
    leverage: float
    alerts: list[str] = field(default_factory=list)


def day_return(equity: float, last_equity: float | None) -> float | None:
    """P&L du jour en fraction du capital de la veille. None le premier jour."""
    if last_equity is None or last_equity <= 0.0 or not np.isfinite(equity):
        return None
    return equity / last_equity - 1.0


def gross_leverage(qty: dict[str, int], prices: dict[str, float], equity: float) -> float:
    """Σ |N × P| / capital. Infini si le capital est nul et qu'il reste une position."""
    gross = 0.0
    for ticker, shares in qty.items():
        price = prices.get(ticker, float("nan"))
        if not np.isfinite(price):
            continue
        gross += abs(shares) * price
    if equity <= 0.0:
        return float("inf") if gross > 0.0 else 0.0
    return gross / equity


def _signed_delta(order: Order) -> int:
    return order.qty if order.side == "buy" else -order.qty


def increases_exposure(held: int, order: Order) -> bool:
    """Vrai si |position| augmente. Couvrir un short ou alléger un long : faux."""
    return abs(held + _signed_delta(order)) > abs(held)


def _clip_adv(order: Order, price: float, adv: float) -> Order | None:
    max_notional = ADV_FRACTION * adv
    max_qty = int(np.floor(max_notional / price)) if price > 0.0 else 0
    if order.qty <= max_qty:
        return order
    if max_qty <= 0:
        return None
    return Order(
        ticker=order.ticker,
        side=order.side,
        qty=max_qty,
        order_type=order.order_type,
        limit_price=order.limit_price,
        reason="adv",
        status="planned",
    )


def apply_risk(
    orders: list[Order],
    held: dict[str, int],
    prices: dict[str, float],
    equity: float,
    last_equity: float | None,
    adv: dict[str, float] | None = None,
) -> RiskDecision:
    """
    Filtre la liste d'ordres. Les ordres refusés ne sont pas dans `orders` :
    l'appelant les journalise à part s'il les a encore.
    """
    alerts: list[str] = []
    leverage = gross_leverage(held, prices, equity)
    pnl = day_return(equity, last_equity)

    # isclose : 95/100 − 1 vaut −0,050000000000000044, qui ne doit pas couper.
    tripped = (
        pnl is not None
        and pnl < CIRCUIT_BREAKER
        and not np.isclose(pnl, CIRCUIT_BREAKER, rtol=0.0, atol=1e-9)
    )
    if tripped:
        alerts.append(
            f"circuit breaker : P&L jour {pnl:.2%} < {CIRCUIT_BREAKER:.0%}"
        )
        return RiskDecision(
            orders=[],
            circuit_breaker=True,
            leverage=leverage,
            alerts=alerts,
        )

    kept: list[Order] = []
    adv_missing = False
    for order in orders:
        if leverage > PREVENTIVE_LEVERAGE and increases_exposure(held.get(order.ticker, 0), order):
            alerts.append(
                f"{order.ticker} bloqué : levier {leverage:.2f} > {PREVENTIVE_LEVERAGE}"
            )
            continue
        price = prices.get(order.ticker, float("nan"))
        if adv is None or order.ticker not in adv or not np.isfinite(adv.get(order.ticker, np.nan)):
            adv_missing = True
            kept.append(order)
            continue
        clipped = _clip_adv(order, price, float(adv[order.ticker]))
        if clipped is None:
            alerts.append(f"{order.ticker} bloqué : ordre au-dessus de {ADV_FRACTION:.0%} de l'ADV")
            continue
        if clipped.qty != order.qty:
            alerts.append(
                f"{order.ticker} réduit de {order.qty} à {clipped.qty} "
                f"actions ({ADV_FRACTION:.0%} ADV)"
            )
        kept.append(clipped)

    if adv_missing and orders:
        alerts.append("ADV indisponible : contrôle de liquidité non appliqué")

    return RiskDecision(
        orders=kept,
        circuit_breaker=False,
        leverage=leverage,
        alerts=alerts,
    )
