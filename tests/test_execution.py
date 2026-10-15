"""
tests/test_execution.py — bloc 10, sans réseau.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from systrade.execution.broker import MemoryBroker
from systrade.execution.journal import Journal
from systrade.execution.loop import close_session, open_session, work_session
from systrade.execution.passive import next_action, passive_limit
from systrade.execution.risk import apply_risk
from systrade.execution.session import (
    IncompleteMarket,
    build_close_plan,
    integer_book,
    round_shares,
)
from systrade.execution.types import Order, Quote
from systrade.portfolio.sizing import MAX_LEVERAGE

AS_OF = pd.Timestamp("2024-06-03")


def _plan(**overrides):
    base = dict(
        as_of=AS_OF,
        shares_per_dollar=pd.Series({"SPY": 0.05}),
        prices=pd.Series({"SPY": 100.0}),
        forecast=pd.Series({"SPY": 10.0}),
        vol_annual=pd.Series({"SPY": 0.16}),
        weight=pd.Series({"SPY": 1.0}),
        idm=1.0,
        equity=100_000.0,
        last_equity=100_000.0,
        held={},
        buffered=pd.Series(dtype=float),
        adv={"SPY": 1e12},
    )
    base.update(overrides)
    return build_close_plan(**base)


def _snap():
    return {
        "as_of": AS_OF,
        "shares_per_dollar": pd.Series({"SPY": 0.05}),
        "prices": pd.Series({"SPY": 100.0}),
        "forecast": pd.Series({"SPY": 10.0}),
        "vol_annual": pd.Series({"SPY": 0.16}),
        "weight": pd.Series({"SPY": 1.0}),
        "idm": 1.0,
    }


# ── Décision ──────────────────────────────────────────────────────────────────

def test_from_flat_buffer_then_leverage_matches_the_backtest():
    """Cible 5 000 actions, buffer à 4 500, plafond 2× → 2 000."""
    plan = _plan()
    assert plan.targets[0]["buffered"] == pytest.approx(4_500.0)
    assert plan.targets[0]["desired"] == 2_000
    assert plan.leverage == pytest.approx(MAX_LEVERAGE)
    assert plan.orders == [Order("SPY", "buy", 2_000)]


def test_short_from_flat_sells_the_near_edge():
    plan = _plan(shares_per_dollar=pd.Series({"SPY": -0.001}), prices=pd.Series({"SPY": 50.0}))
    assert plan.targets[0]["desired"] == -90
    assert plan.orders == [Order("SPY", "sell", 90)]


def test_position_inside_the_band_does_not_order():
    plan = _plan(
        shares_per_dollar=pd.Series({"SPY": 0.001}),
        prices=pd.Series({"SPY": 50.0}),
        held={"SPY": 100},
        buffered=pd.Series({"SPY": 100.0}),
    )
    assert plan.orders == []
    assert plan.targets[0]["desired"] == 100


def test_round_half_away_from_zero():
    assert round_shares(1.5) == 2
    assert round_shares(-1.5) == -2
    assert round_shares(0.4) == 0


def test_integer_book_is_shaved_back_under_the_leverage_cap():
    qty = integer_book(pd.Series({"SPY": 2.6}), pd.Series({"SPY": 100.0}), equity=100.0)
    assert qty == {"SPY": 2}
    assert abs(2 * 100) / 100 == pytest.approx(MAX_LEVERAGE)


def test_missing_price_refuses_the_plan():
    with pytest.raises(IncompleteMarket):
        _plan(prices=pd.Series({"SPY": float("nan")}))


def test_position_outside_the_universe_is_not_flattened():
    plan = _plan(held={"OTHER": 10})
    assert all(order.ticker != "OTHER" for order in plan.orders)
    assert any("OTHER" in alert for alert in plan.alerts)


# ── Risque ────────────────────────────────────────────────────────────────────

def test_circuit_breaker_blocks_every_order_strictly_below_minus_five_percent():
    blocked = apply_risk(
        [Order("SPY", "buy", 1)],
        held={},
        prices={"SPY": 100.0},
        equity=94.0,
        last_equity=100.0,
    )
    assert blocked.circuit_breaker
    assert blocked.orders == []

    allowed = apply_risk(
        [Order("SPY", "buy", 1)],
        held={},
        prices={"SPY": 100.0},
        equity=95.0,
        last_equity=100.0,
    )
    assert not allowed.circuit_breaker
    assert allowed.orders == [Order("SPY", "buy", 1)]


def test_leverage_gate_blocks_increases_and_keeps_reductions():
    decision = apply_risk(
        [Order("SPY", "buy", 10), Order("IEF", "sell", 5)],
        held={"SPY": 20, "IEF": 20},
        prices={"SPY": 100.0, "IEF": 100.0},
        equity=1_000.0,
        last_equity=1_000.0,
        adv={"SPY": 1e12, "IEF": 1e12},
    )
    assert decision.leverage == pytest.approx(4.0)
    assert [order.ticker for order in decision.orders] == ["IEF"]


def test_adv_clips_the_order_to_one_percent():
    decision = apply_risk(
        [Order("SPY", "buy", 10)],
        held={},
        prices={"SPY": 50.0},
        equity=100_000.0,
        last_equity=100_000.0,
        adv={"SPY": 10_000.0},
    )
    assert decision.orders[0].qty == 2
    assert any("réduit" in alert for alert in decision.alerts)


def test_missing_adv_keeps_the_order_and_says_so():
    decision = apply_risk(
        [Order("SPY", "buy", 1)],
        held={},
        prices={"SPY": 50.0},
        equity=100_000.0,
        last_equity=100_000.0,
        adv=None,
    )
    assert decision.orders[0].qty == 1
    assert any("ADV indisponible" in alert for alert in decision.alerts)


# ── Exécution passive ─────────────────────────────────────────────────────────

def test_buy_lifts_the_bid_and_sell_hits_the_ask():
    quote = Quote(100.129, 100.121)
    assert passive_limit("buy", quote) == pytest.approx(100.12)
    assert passive_limit("sell", quote) == pytest.approx(100.13)


def test_missing_side_falls_back_to_mid():
    assert passive_limit("buy", Quote(0.0, 100.009)) == pytest.approx(100.00)
    assert passive_limit("sell", Quote(0.0, 0.0)) is None


def test_repeg_at_five_minutes_and_market_at_thirty():
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    assert next_action(start, None, start + timedelta(minutes=4)) == "hold"
    assert next_action(start, None, start + timedelta(minutes=5)) == "repeg"
    assert next_action(start, start + timedelta(minutes=5), start + timedelta(minutes=9)) == "hold"
    later = start + timedelta(minutes=30)
    assert next_action(start, start + timedelta(minutes=25), later) == "market"


# ── Journal et boucle ─────────────────────────────────────────────────────────

def test_close_writes_the_plan_and_does_not_submit(tmp_path):
    broker = MemoryBroker(equity=100_000.0, last_equity=100_000.0)
    journal = Journal(tmp_path / "trading.db")
    plan = close_session(broker, journal, snapshot=_snap(), adv={"SPY": 1e12})
    assert plan is not None
    assert broker.submitted == 0
    assert journal.load_buffer()["SPY"] == pytest.approx(4_500.0)
    assert len(journal.planned_orders()) == 1


def test_a_second_close_supersedes_unsent_orders(tmp_path):
    broker = MemoryBroker()
    journal = Journal(tmp_path / "trading.db")
    close_session(broker, journal, snapshot=_snap(), adv={"SPY": 1e12})
    close_session(broker, journal, snapshot=_snap(), adv={"SPY": 1e12})
    assert len(journal.planned_orders()) == 1
    with journal._connect() as conn:
        superseded = conn.execute(
            "SELECT COUNT(*) AS n FROM orders WHERE status = 'superseded'"
        ).fetchone()["n"]
    assert superseded == 1


def test_incomplete_close_leaves_the_buffer_alone(tmp_path):
    broker = MemoryBroker()
    journal = Journal(tmp_path / "trading.db")
    snap = _snap()
    snap["prices"] = pd.Series({"SPY": float("nan")})
    assert close_session(broker, journal, snapshot=snap) is None
    assert journal.load_buffer().empty
    assert broker.submitted == 0


def test_open_submits_a_passive_limit_and_work_turns_it_into_a_market(tmp_path):
    broker = MemoryBroker(quotes={"SPY": Quote(100.0, 100.05)})
    journal = Journal(tmp_path / "trading.db")
    close_session(broker, journal, snapshot=_snap(), adv={"SPY": 1e12})
    start = datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    ids = open_session(broker, journal, now=start)
    assert ids == ["mem-1"]
    assert broker.orders["mem-1"].limit_price == pytest.approx(100.0)
    assert broker.orders["mem-1"].order_type == "limit"

    assert work_session(broker, journal, now=start + timedelta(minutes=5)) == 1
    assert broker.cancelled == ["mem-1"]
    assert broker.orders["mem-2"].order_type == "limit"

    assert work_session(broker, journal, now=start + timedelta(minutes=30)) == 1
    assert broker.orders["mem-3"].order_type == "market"
    assert work_session(broker, journal, now=start + timedelta(minutes=31)) == 0


def test_circuit_breaker_is_journaled_and_nothing_is_planned(tmp_path):
    broker = MemoryBroker(equity=90_000.0, last_equity=100_000.0)
    journal = Journal(tmp_path / "trading.db")
    plan = close_session(broker, journal, snapshot=_snap(), adv={"SPY": 1e12})
    assert plan is not None and plan.circuit_breaker
    assert journal.planned_orders() == []
    assert not journal.load_buffer().empty
