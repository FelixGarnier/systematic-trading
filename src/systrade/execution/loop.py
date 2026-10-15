"""
src/systrade/execution/loop.py
────────────────────────────────
Boucle paper en trois temps.

close  Après la clôture. Met à jour Tiingo et FRED, calcule le livre,
       journalise les ordres. N'envoie rien.
open   À l'ouverture du lendemain. Limite au bid (achat) ou à l'ask (vente).
work   Toutes les 5 minutes : re-peg. À 30 minutes : market sur le reliquat.

Le sizing utilise l'equity du compte paper, pas le capital terminal du
backtest. Le buffer démarre à plat le premier soir.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd

from systrade.backtest.engine import assemble_book
from systrade.data import store
from systrade.execution.broker import BrokerInterface
from systrade.execution.journal import Journal
from systrade.execution.passive import next_action, with_limit
from systrade.execution.session import ClosePlan, IncompleteMarket, build_close_plan
from systrade.execution.types import Order

logger = logging.getLogger(__name__)


def snapshot_from_book(book: dict, prices_unadj: pd.DataFrame) -> dict:
    """Dernière ligne du livre assemblé, prête pour build_close_plan."""
    k = book["shares_per_dollar"]
    as_of = pd.Timestamp(k.index[-1])
    cols = k.columns
    return {
        "as_of": as_of,
        "shares_per_dollar": k.loc[as_of],
        "prices": prices_unadj.reindex(columns=cols).loc[as_of],
        "forecast": book["forecast"].reindex(columns=cols).loc[as_of],
        "vol_annual": book["vol"]["annual"].reindex(columns=cols).loc[as_of],
        "weight": book["weights"].reindex(columns=cols).loc[as_of],
        "idm": float(book["idm"].loc[as_of]),
    }


def _through(frame: pd.DataFrame, session_date: date | datetime | None) -> pd.DataFrame:
    """Coupe l'historique à une séance incluse. None laisse la série entière."""
    if session_date is None:
        return frame
    return frame.loc[: pd.Timestamp(session_date)]


def _load_snapshot(refresh: bool, session_date: date | datetime | None = None) -> dict:
    if refresh:
        from systrade.data.loader import update

        update()
    prices_adj = _through(store.load_prices_adj(), session_date)
    prices_unadj = _through(store.load_prices_unadj(), session_date)
    book = assemble_book(
        prices_adj,
        prices_unadj,
        _through(store.load_dividends(), session_date),
        _through(store.load_rates(), session_date),
    )
    return snapshot_from_book(book, prices_unadj)


def _adv_for(broker: BrokerInterface, tickers: list[str]) -> dict[str, float] | None:
    fn = getattr(broker, "average_dollar_volume", None)
    if fn is None:
        return None
    try:
        return fn(tickers)
    except Exception:
        logger.exception("ADV indisponible")
        return None


def close_session(
    broker: BrokerInterface,
    journal: Journal,
    *,
    refresh: bool = False,
    snapshot: dict | None = None,
    adv: dict[str, float] | None = None,
    session_date: date | datetime | None = None,
) -> ClosePlan | None:
    """
    Calcule et journalise. Retourne None si le marché est incomplet :
    le buffer n'est pas mis à jour, aucun ordre n'est prévu.
    """
    snap = snapshot if snapshot is not None else _load_snapshot(refresh, session_date)
    account = broker.get_account()
    held = broker.get_positions()
    if adv is None:
        adv = _adv_for(broker, [str(t) for t in snap["shares_per_dollar"].index])
    try:
        plan = build_close_plan(
            as_of=snap["as_of"],
            shares_per_dollar=snap["shares_per_dollar"],
            prices=snap["prices"],
            forecast=snap["forecast"],
            vol_annual=snap["vol_annual"],
            weight=snap["weight"],
            idm=snap["idm"],
            equity=account.equity,
            last_equity=account.last_equity,
            held=held,
            buffered=journal.load_buffer(),
            adv=adv,
        )
    except IncompleteMarket as exc:
        journal.record_anomaly("data", str(exc))
        logger.error("%s", exc)
        return None
    floor = session_date.date() if isinstance(session_date, datetime) else session_date
    if floor is not None and pd.Timestamp(plan.as_of).date() < floor:
        logger.info(
            "barre %s avant la séance %s, clôture non enregistrée",
            plan.as_of.date(),
            floor,
        )
        return None
    journal.record_close(plan)
    logger.info(
        "clôture %s : %d ordres, levier %.2f, equity %.0f",
        pd.Timestamp(plan.as_of).date(),
        len(plan.orders),
        plan.leverage,
        plan.equity,
    )
    return plan


def open_session(
    broker: BrokerInterface,
    journal: Journal,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Soumet les ordres planned de la dernière clôture. Retourne les ids courtier."""
    stamp = now or datetime.now(timezone.utc)
    broker_ids: list[str] = []
    for row in journal.planned_orders():
        order = Order(ticker=row["ticker"], side=row["side"], qty=int(row["qty"]))
        quote = broker.get_quote(order.ticker)
        live = with_limit(order, quote)
        if live is None:
            journal.mark_status(row["id"], "rejected", "pas de cotation")
            journal.record_anomaly("quote", f"{order.ticker} : pas de bid/ask")
            continue
        try:
            broker_id = broker.submit_order(live)
        except Exception as exc:
            journal.mark_status(row["id"], "rejected", str(exc))
            journal.record_anomaly("broker", f"{order.ticker} : {exc}")
            continue
        journal.mark_submitted(row["id"], broker_id, live.limit_price, "limit", stamp)
        broker_ids.append(broker_id)
    return broker_ids


def work_session(
    broker: BrokerInterface,
    journal: Journal,
    *,
    now: datetime | None = None,
) -> int:
    """
    Re-peg ou conversion market des ordres encore ouverts.

    Retourne le nombre d'ordres modifiés.
    """
    stamp = now or datetime.now(timezone.utc)
    changed = 0
    for row in journal.working_orders():
        try:
            view = broker.get_order(row["broker_id"])
        except Exception as exc:
            journal.record_anomaly("broker", f"{row['ticker']} : {exc}")
            continue
        if view.status == "filled":
            journal.mark_status(row["id"], "filled")
            continue
        if view.status in {"cancelled", "rejected"}:
            journal.mark_status(row["id"], view.status)
            continue
        submitted_at = datetime.fromisoformat(row["submitted_at"])
        last_peg = (
            datetime.fromisoformat(row["last_peg_at"]) if row["last_peg_at"] else None
        )
        action = next_action(submitted_at, last_peg, stamp)
        if action == "hold" or (action == "market" and row["order_type"] == "market"):
            continue
        remaining = view.qty - view.filled_qty
        if remaining <= 0:
            journal.mark_status(row["id"], "filled")
            continue
        try:
            broker.cancel_order(row["broker_id"])
        except Exception as exc:
            journal.record_anomaly("broker", f"annulation {row['ticker']} : {exc}")
            continue
        if action == "market":
            replacement = Order(
                ticker=row["ticker"],
                side=row["side"],
                qty=remaining,
                order_type="market",
            )
        else:
            quote = broker.get_quote(row["ticker"])
            replacement = with_limit(
                Order(ticker=row["ticker"], side=row["side"], qty=remaining),
                quote,
            )
            if replacement is None:
                journal.mark_status(row["id"], "rejected", "pas de cotation au re-peg")
                continue
        try:
            broker_id = broker.submit_order(replacement)
        except Exception as exc:
            journal.mark_status(row["id"], "rejected", str(exc))
            journal.record_anomaly("broker", f"remplacement {row['ticker']} : {exc}")
            continue
        journal.mark_submitted(
            row["id"],
            broker_id,
            replacement.limit_price,
            replacement.order_type,
            stamp,
        )
        changed += 1
    return changed
