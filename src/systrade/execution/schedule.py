"""
src/systrade/execution/schedule.py
───────────────────────────────────
Décide, à un instant donné, s'il faut clôturer, envoyer ou reprendre.

L'horloge est celle de New York, pas celle de Paris : le décalage
change deux fois par an, et les jours fériés US ne sont pas des
jours de bourse. Une séance n'est close que 20 minutes après
l'heure de clôture du calendrier Alpaca (16:00, ou plus tôt les
jours courts), le temps que la barre quotidienne existe.

`tick` est idempotent. Lancé toutes les cinq minutes, il rattrape
une soirée manquée au prochain passage, y compris le lendemain matin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from systrade.execution.broker import BrokerInterface
from systrade.execution.journal import Journal
from systrade.execution.loop import close_session, open_session, work_session

logger = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
CLOSE_LAG = timedelta(minutes=20)


@dataclass(frozen=True)
class Session:
    day: date
    open: datetime
    close: datetime


def as_ny(moment: datetime) -> datetime:
    """Normalise un instant en heure de New York."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=NY)
    return moment.astimezone(NY)


def due_actions(
    now: datetime,
    sessions: list[Session],
    last_close: date | None,
    n_planned: int,
    n_working: int,
) -> list[str]:
    """
    close, open, work, dans cet ordre, ou une liste vide.

    close : la dernière séance terminée n'a pas encore de plan.
    open  : le marché est ouvert et des ordres attendent.
    work  : des ordres sont déjà chez le courtier pendant la séance.
    """
    now = as_ny(now)
    actions: list[str] = []

    finished = [session for session in sessions if now >= as_ny(session.close) + CLOSE_LAG]
    if finished:
        target = max(finished, key=lambda session: session.day)
        if last_close is None or last_close < target.day:
            actions.append("close")

    current = next(
        (
            session
            for session in sessions
            if as_ny(session.open) <= now <= as_ny(session.close)
        ),
        None,
    )
    if current is not None and n_planned > 0:
        actions.append("open")
    if current is not None and n_working > 0:
        actions.append("work")
    return actions


def _sessions_from(rows: list[tuple[date, datetime, datetime]]) -> list[Session]:
    return [
        Session(day=day, open=as_ny(open_at), close=as_ny(close_at))
        for day, open_at, close_at in rows
    ]


def run_tick(
    broker: BrokerInterface,
    journal: Journal,
    sessions: list[Session],
    now: datetime | None = None,
) -> list[str]:
    """
    Fait ce qui est dû, puis revérifie.

    Une clôture rattrapée pendant que le marché est ouvert produit
    des ordres planned : le même passage les envoie.
    """
    moment = as_ny(now or datetime.now(NY))
    done: list[str] = []

    def pending() -> list[str]:
        return due_actions(
            moment,
            sessions,
            journal.last_close_date(),
            len(journal.planned_orders()),
            len(journal.working_orders()),
        )

    actions = pending()
    if "close" in actions:
        target = max(
            (session for session in sessions if moment >= as_ny(session.close) + CLOSE_LAG),
            key=lambda session: session.day,
        )
        plan = close_session(broker, journal, refresh=True, session_date=target.day)
        if plan is not None and plan.as_of.date() >= target.day:
            done.append("close")
            logger.info("clôture %s enregistrée", target.day)
        else:
            got = None if plan is None else plan.as_of.date()
            logger.info("clôture %s en attente, dernière barre %s", target.day, got)
            return done

    actions = pending()
    if "open" in actions:
        ids = open_session(broker, journal, now=moment)
        done.append("open")
        logger.info("%d ordres soumis", len(ids))
    if "work" in actions:
        changed = work_session(broker, journal, now=moment)
        done.append("work")
        logger.info("%d ordres repris", changed)
    if not done:
        logger.info("rien à faire")
    return done


def fetch_sessions(broker: BrokerInterface, today: date) -> list[Session]:
    """Calendrier Alpaca sur une fenêtre de dix jours autour d'aujourd'hui."""
    calendar = getattr(broker, "market_calendar", None)
    if calendar is None:
        raise RuntimeError("ce courtier ne fournit pas le calendrier NYSE")
    rows = calendar(today - timedelta(days=10), today + timedelta(days=10))
    return _sessions_from(rows)
