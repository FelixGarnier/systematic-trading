"""
src/systrade/execution/broker.py
─────────────────────────────────
Interface courtier et implémentation en mémoire.

Alpaca n'est pas importé ici. Les tests et le dry-run passent par
MemoryBroker. Le compte paper réel est dans alpaca_broker.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from systrade.execution.types import Account, BrokerOrder, Order, Quote


class BrokerInterface(ABC):
    @abstractmethod
    def get_account(self) -> Account:
        """Equity, cash, et equity de la veille pour le P&L du jour."""

    @abstractmethod
    def get_positions(self) -> dict[str, int]:
        """Actions signées. Long > 0, short < 0. Les lignes à zéro sont omises."""

    @abstractmethod
    def get_price(self, ticker: str) -> float:
        """Dernier prix négociable. Le sizing, lui, utilise la clôture Tiingo."""

    @abstractmethod
    def get_quote(self, ticker: str) -> Quote:
        """Bid et ask pour l'ordre limite passif."""

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Envoie l'ordre. Retourne l'identifiant courtier."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Annule un ordre encore ouvert."""

    @abstractmethod
    def get_order(self, order_id: str) -> BrokerOrder:
        """Statut et quantité déjà exécutée."""


class MemoryBroker(BrokerInterface):
    """
    Courtier sans réseau.

    Les ordres restent ouverts tant qu'on n'appelle pas `fill`. Utile pour
    le dry-run et pour vérifier que la clôture ne soumet rien.
    """

    def __init__(
        self,
        equity: float = 100_000.0,
        cash: float | None = None,
        last_equity: float | None = None,
        positions: dict[str, int] | None = None,
        quotes: dict[str, Quote] | None = None,
    ) -> None:
        self.equity = float(equity)
        self.cash = float(equity if cash is None else cash)
        self.last_equity = last_equity
        self.positions = {k: int(v) for k, v in (positions or {}).items() if int(v) != 0}
        self.quotes = dict(quotes or {})
        self.orders: dict[str, Order] = {}
        self._status: dict[str, str] = {}
        self._filled: dict[str, int] = {}
        self._seq = 0
        self.submitted = 0
        self.cancelled: list[str] = []

    def get_account(self) -> Account:
        return Account(equity=self.equity, cash=self.cash, last_equity=self.last_equity)

    def get_positions(self) -> dict[str, int]:
        return dict(self.positions)

    def get_price(self, ticker: str) -> float:
        return self.get_quote(ticker).mid

    def get_quote(self, ticker: str) -> Quote:
        return self.quotes.get(ticker, Quote(0.0, 0.0))

    def submit_order(self, order: Order) -> str:
        if order.qty <= 0:
            raise ValueError(f"quantité nulle pour {order.ticker}")
        self._seq += 1
        self.submitted += 1
        broker_id = f"mem-{self._seq}"
        self.orders[broker_id] = order
        self._status[broker_id] = "open"
        self._filled[broker_id] = 0
        return broker_id

    def cancel_order(self, order_id: str) -> None:
        if self._status.get(order_id) == "filled":
            return
        self._status[order_id] = "cancelled"
        self.cancelled.append(order_id)

    def get_order(self, order_id: str) -> BrokerOrder:
        order = self.orders[order_id]
        return BrokerOrder(
            broker_id=order_id,
            status=self._status[order_id],
            qty=order.qty,
            filled_qty=self._filled[order_id],
        )

    def fill(self, order_id: str) -> None:
        """Exécute le reliquat et met à jour les positions. Réservé aux tests."""
        order = self.orders[order_id]
        remaining = order.qty - self._filled[order_id]
        signed = remaining if order.side == "buy" else -remaining
        self.positions[order.ticker] = self.positions.get(order.ticker, 0) + signed
        if self.positions[order.ticker] == 0:
            del self.positions[order.ticker]
        self._filled[order_id] = order.qty
        self._status[order_id] = "filled"
