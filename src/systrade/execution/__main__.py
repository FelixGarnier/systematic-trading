"""
Boucle paper Alpaca.

    python -m systrade.execution close
    python -m systrade.execution open
    python -m systrade.execution work

`--dry-run` calcule le livre sur les données locales et écrit dans
data/live/dry_run.db. Aucun appel Alpaca.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from systrade.execution.broker import MemoryBroker
from systrade.execution.journal import DRY_RUN_PATH, JOURNAL_PATH, LIVE_DIR, Journal
from systrade.execution.loop import close_session, open_session, work_session
from systrade.execution.schedule import fetch_sessions, run_tick


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Boucle paper du bloc 10")
    sub = parser.add_subparsers(dest="phase", required=True)
    for name in ("close", "open", "work", "tick"):
        command = sub.add_parser(name)
        command.add_argument(
            "--dry-run",
            action="store_true",
            help="courtier en mémoire, pas d'appel Alpaca",
        )
        if name == "close":
            command.add_argument(
                "--no-refresh",
                action="store_true",
                help="ne pas retélécharger Tiingo et FRED",
            )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.phase == "tick":
        return _tick()

    journal = Journal(DRY_RUN_PATH if args.dry_run else JOURNAL_PATH)
    if args.dry_run:
        broker = MemoryBroker(equity=100_000.0, last_equity=100_000.0)
    else:
        from systrade.execution.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker()

    if args.phase == "close":
        refresh = not args.dry_run and not args.no_refresh
        plan = close_session(broker, journal, refresh=refresh)
        if plan is None:
            return 1
        print(
            f"{plan.as_of.date()}  equity {plan.equity:,.0f}  "
            f"levier {plan.leverage:.2f}  ordres {len(plan.orders)}"
        )
        for order in plan.orders:
            print(f"  {order.side:4} {order.qty:6d}  {order.ticker}")
        return 0
    if args.phase == "open":
        ids = open_session(broker, journal)
        print(f"{len(ids)} ordres soumis")
        return 0
    if args.phase == "work":
        changed = work_session(broker, journal)
        print(f"{changed} ordres repris")
        return 0
    return _tick()


def _tick() -> int:
    """Un passage. Le verrou empêche deux ticks de se chevaucher."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LIVE_DIR / "tick.lock"
    with open(lock_path, "w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logging.info("tick déjà en cours")
            return 0
        from systrade.execution.alpaca_broker import AlpacaBroker

        now = datetime.now(ZoneInfo("America/New_York"))
        broker = AlpacaBroker()
        journal = Journal(JOURNAL_PATH)
        sessions = fetch_sessions(broker, now.date())
        done = run_tick(broker, journal, sessions, now=now)
        print(now.strftime("%Y-%m-%d %H:%M ET"), ", ".join(done) if done else "rien")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
