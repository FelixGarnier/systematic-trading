"""
tests/test_schedule.py — le tick ne dépend pas de l'heure de Paris.
"""

from datetime import date, datetime

import pandas as pd

from systrade.execution.broker import MemoryBroker
from systrade.execution.journal import Journal
from systrade.execution.schedule import NY, Session, due_actions, run_tick
from systrade.execution.session import ClosePlan
from systrade.execution.types import Order, Quote


def _at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY)


def _sessions() -> list[Session]:
    wednesday = date(2026, 10, 14)
    thursday = date(2026, 10, 15)
    return [
        Session(wednesday, _at(wednesday, 9, 30), _at(wednesday, 16, 0)),
        Session(thursday, _at(thursday, 9, 30), _at(thursday, 16, 0)),
    ]


def test_after_the_close_plans_once():
    thursday = date(2026, 10, 15)
    moment = _at(thursday, 16, 40)
    assert due_actions(moment, _sessions(), None, 0, 0) == ["close"]
    assert due_actions(moment, _sessions(), thursday, 1, 0) == []


def test_twenty_minutes_before_the_bar_exists_does_not_close():
    """Mercredi est déjà clos. Jeudi 16h10, la barre du jour n'est pas encore due."""
    thursday = date(2026, 10, 15)
    moment = _at(thursday, 16, 10)
    assert due_actions(moment, _sessions(), date(2026, 10, 14), 0, 0) == []


def test_open_and_work_only_during_the_session():
    thursday = date(2026, 10, 15)
    morning = _at(thursday, 9, 40)
    done = due_actions(morning, _sessions(), date(2026, 10, 14), 2, 1)
    assert done == ["open", "work"]
    evening = due_actions(_at(thursday, 20, 0), _sessions(), thursday, 2, 1)
    assert evening == []


def test_a_missed_close_is_due_the_next_morning_before_any_order():
    """Jeudi 9h40, la clôture de mercredi n'est pas faite : on close, on n'envoie pas encore."""
    morning = _at(date(2026, 10, 15), 9, 40)
    assert due_actions(morning, _sessions(), None, 0, 0) == ["close"]


def test_catchup_close_then_sends_while_the_market_is_open(tmp_path, monkeypatch):
    journal = Journal(tmp_path / "trading.db")
    broker = MemoryBroker(quotes={"SPY": Quote(100.0, 100.05)})
    plan = ClosePlan(
        as_of=pd.Timestamp("2026-10-14"),
        equity=100_000.0,
        last_equity=100_000.0,
        day_pnl=0.0,
        leverage=1.0,
        vol_upper=0.1,
        idm=1.0,
        circuit_breaker=False,
        alerts=[],
        targets=[
            {
                "ticker": "SPY",
                "forecast": 10.0,
                "vol_annual": 0.16,
                "weight": 1.0,
                "idm": 1.0,
                "n_star": 10.0,
                "buffered": 9.0,
                "desired": 1,
                "held": 0,
                "price": 100.0,
            }
        ],
        orders=[Order("SPY", "buy", 1)],
        blocked=[],
        buffered=pd.Series({"SPY": 9.0}),
    )

    def fake_close(broker, journal, **kwargs):
        journal.record_close(plan)
        return plan

    monkeypatch.setattr("systrade.execution.schedule.close_session", fake_close)
    morning = _at(date(2026, 10, 15), 9, 40)
    done = run_tick(broker, journal, _sessions(), now=morning)
    assert done == ["close", "open"]
    assert broker.submitted == 1


def test_evening_close_does_not_send(tmp_path, monkeypatch):
    journal = Journal(tmp_path / "trading.db")
    broker = MemoryBroker(quotes={"SPY": Quote(100.0, 100.05)})
    plan = ClosePlan(
        as_of=pd.Timestamp("2026-10-15"),
        equity=100_000.0,
        last_equity=100_000.0,
        day_pnl=0.0,
        leverage=1.0,
        vol_upper=0.1,
        idm=1.0,
        circuit_breaker=False,
        alerts=[],
        targets=[
            {
                "ticker": "SPY",
                "forecast": 10.0,
                "vol_annual": 0.16,
                "weight": 1.0,
                "idm": 1.0,
                "n_star": 10.0,
                "buffered": 9.0,
                "desired": 1,
                "held": 0,
                "price": 100.0,
            }
        ],
        orders=[Order("SPY", "buy", 1)],
        blocked=[],
        buffered=pd.Series({"SPY": 9.0}),
    )

    def fake_close(broker, journal, **kwargs):
        journal.record_close(plan)
        return plan

    monkeypatch.setattr("systrade.execution.schedule.close_session", fake_close)
    done = run_tick(broker, journal, _sessions(), now=_at(date(2026, 10, 15), 16, 40))
    assert done == ["close"]
    assert broker.submitted == 0
