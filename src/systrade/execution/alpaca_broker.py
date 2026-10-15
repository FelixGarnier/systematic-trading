"""
src/systrade/execution/alpaca_broker.py
────────────────────────────────────────
Alpaca en paper uniquement.

Le point d'entrée de la boucle construit toujours ce courtier avec
paper=True. Les clés viennent de ALPACA_API_KEY et ALPACA_SECRET_KEY.
Les cotations passent par le flux IEX, celui du compte paper sans
abonnement SIP.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

from dotenv import load_dotenv

from systrade.execution.broker import BrokerInterface
from systrade.execution.types import Account, BrokerOrder, Order, Quote


class AlpacaBroker(BrokerInterface):
    def __init__(self, api_key: str | None = None, secret_key: str | None = None) -> None:
        load_dotenv()
        api_key = api_key or os.getenv("ALPACA_API_KEY")
        secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY et ALPACA_SECRET_KEY manquants. "
                "Le paper trading ne démarre pas sans elles."
            )
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        # paper=True est figé : ce projet ne soumet pas d'ordres sur un compte réel.
        self._trading = TradingClient(api_key, secret_key, paper=True)
        self._data = StockHistoricalDataClient(api_key, secret_key)

    def get_account(self) -> Account:
        account = self._trading.get_account()
        last = getattr(account, "last_equity", None)
        return Account(
            equity=float(account.equity),
            cash=float(account.cash),
            last_equity=float(last) if last not in (None, "") else None,
        )

    def get_positions(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for pos in self._trading.get_all_positions():
            qty = int(float(pos.qty))
            side = str(getattr(pos, "side", "")).lower()
            signed = -qty if "short" in side else qty
            if signed != 0:
                out[str(pos.symbol)] = signed
        return out

    def get_price(self, ticker: str) -> float:
        return self.get_quote(ticker).mid

    def get_quote(self, ticker: str) -> Quote:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestQuoteRequest

        raw = self._data.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=ticker, feed=DataFeed.IEX)
        )
        quote = raw[ticker]
        return Quote(bid=float(quote.bid_price or 0.0), ask=float(quote.ask_price or 0.0))

    def submit_order(self, order: Order) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        if order.order_type == "market" or order.limit_price is None:
            request = MarketOrderRequest(
                symbol=order.ticker,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            request = LimitOrderRequest(
                symbol=order.ticker,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=order.limit_price,
            )
        submitted = self._trading.submit_order(request)
        return str(submitted.id)

    def cancel_order(self, order_id: str) -> None:
        self._trading.cancel_order_by_id(order_id)

    def get_order(self, order_id: str) -> BrokerOrder:
        raw = self._trading.get_order_by_id(order_id)
        status = str(raw.status).lower()
        if "filled" in status and "partial" not in status:
            state = "filled"
        elif any(word in status for word in ("cancel", "expire", "reject", "suspend")):
            state = "cancelled" if "reject" not in status else "rejected"
        else:
            state = "open"
        filled = int(float(getattr(raw, "filled_qty", 0) or 0))
        qty = int(float(raw.qty))
        return BrokerOrder(broker_id=order_id, status=state, qty=qty, filled_qty=filled)

    def market_calendar(self, start: date, end: date) -> list[tuple[date, datetime, datetime]]:
        """Séances NYSE entre deux dates. Ouverture et clôture sont naïves, heure de New York."""
        from alpaca.trading.requests import GetCalendarRequest

        rows = self._trading.get_calendar(GetCalendarRequest(start=start, end=end))
        return [(row.date, row.open, row.close) for row in rows]

    def average_dollar_volume(self, tickers: list[str], lookback: int = 20) -> dict[str, float]:
        """ADV en dollars sur les `lookback` dernières séances quotidiennes."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        if not tickers:
            return {}
        start = datetime.now(timezone.utc) - timedelta(days=lookback * 3)
        bars = self._data.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=TimeFrame.Day,
                start=start,
                feed=DataFeed.IEX,
            )
        )
        frame = bars.df
        if frame is None or len(frame) == 0 or frame.index.nlevels != 2:
            return {}
        out: dict[str, float] = {}
        for ticker, block in frame.groupby(level=0):
            tail = block.tail(lookback)
            dollar = float((tail["close"] * tail["volume"]).mean())
            if dollar == dollar:
                out[str(ticker)] = dollar
        return out
