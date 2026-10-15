"""
src/systrade/execution/passive.py
──────────────────────────────────
Exécution passive de l'ouverture.

Achat limite au bid, vente limite à l'ask. Si un côté de la fourchette
manque, on prend le mid. Re-peg toutes les 5 minutes si l'ordre est
encore ouvert. Passage en market au bout de 30 minutes.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from systrade.execution.types import Order, Quote

REPEG_AFTER = timedelta(minutes=5)
MARKET_AFTER = timedelta(minutes=30)


def passive_limit(side: str, quote: Quote) -> float | None:
    """
    Prix limite passif, arrondi au cent dans le sens qui reste passif.

    Un achat est arrondi vers le bas, une vente vers le haut.
    """
    if side == "buy":
        raw = quote.bid if quote.bid > 0.0 else quote.mid
        if raw <= 0.0:
            return None
        return math.floor(raw * 100.0 + 1e-9) / 100.0
    raw = quote.ask if quote.ask > 0.0 else quote.mid
    if raw <= 0.0:
        return None
    return math.ceil(raw * 100.0 - 1e-9) / 100.0


def with_limit(order: Order, quote: Quote) -> Order | None:
    """Copie l'ordre avec son prix limite. None si aucune cotation."""
    price = passive_limit(order.side, quote)
    if price is None:
        return None
    return Order(
        ticker=order.ticker,
        side=order.side,
        qty=order.qty,
        order_type="limit",
        limit_price=price,
        reason=order.reason,
        status="planned",
    )


def next_action(
    submitted_at: datetime,
    last_peg_at: datetime | None,
    now: datetime,
) -> str:
    """
    hold, repeg, ou market.

    L'horloge part de la soumission. Le re-peg repart de `last_peg_at`.
    """
    if now - submitted_at >= MARKET_AFTER:
        return "market"
    pegged = last_peg_at or submitted_at
    if now - pegged >= REPEG_AFTER:
        return "repeg"
    return "hold"
