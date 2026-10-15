"""
src/systrade/execution/types.py
────────────────────────────────
Objets échangés entre le calcul, le risque et le courtier.

Le backtest porte des actions fractionnaires. Alpaca, pour un short,
demande des actions entières. Le passage à l'entier se fait ici, après
le buffer et après le plafond de levier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Account:
    equity: float
    cash: float
    last_equity: float | None


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        if self.bid > 0.0 and self.ask > 0.0:
            return (self.bid + self.ask) / 2.0
        return max(self.bid, self.ask, 0.0)


@dataclass(frozen=True)
class Order:
    """Ordre voulu. `side` vaut buy ou sell, `qty` est un entier positif."""

    ticker: str
    side: str
    qty: int
    order_type: str = "limit"
    limit_price: float | None = None
    reason: str = ""
    status: str = "planned"


@dataclass(frozen=True)
class BrokerOrder:
    """État d'un ordre déjà chez le courtier."""

    broker_id: str
    status: str
    qty: int
    filled_qty: int
