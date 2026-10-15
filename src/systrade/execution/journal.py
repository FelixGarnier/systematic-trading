"""
src/systrade/execution/journal.py
──────────────────────────────────
Journal SQLite de la boucle paper.

Une clôture est une transaction : run, cibles, ordres, buffer. Si
l'écriture casse, rien n'est gardé. Les ordres `planned` d'une clôture
précédente passent en `superseded` dans la même transaction.

data/live/trading.db pour le compte paper, data/live/dry_run.db pour
un essai local. Les deux sont ignorés par git.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from systrade.data.store import PROJECT_ROOT
from systrade.execution.session import ClosePlan

LIVE_DIR = PROJECT_ROOT / "data" / "live"
JOURNAL_PATH = LIVE_DIR / "trading.db"
DRY_RUN_PATH = LIVE_DIR / "dry_run.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else JOURNAL_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    ts TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    as_of TEXT,
                    equity REAL,
                    last_equity REAL,
                    day_pnl REAL,
                    leverage REAL,
                    vol_upper REAL,
                    idm REAL,
                    circuit_breaker INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS targets (
                    run_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    forecast REAL,
                    vol_annual REAL,
                    weight REAL,
                    idm REAL,
                    n_star REAL,
                    buffered REAL,
                    desired INTEGER,
                    held INTEGER,
                    price REAL,
                    PRIMARY KEY (run_id, ticker)
                );
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL,
                    status TEXT NOT NULL,
                    broker_id TEXT,
                    reason TEXT,
                    submitted_at TEXT,
                    last_peg_at TEXT
                );
                CREATE TABLE IF NOT EXISTS buffer_state (
                    ticker TEXT PRIMARY KEY,
                    buffered REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER,
                    ts TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _sql(value: float | None) -> float | None:
        if value is None or value != value:
            return None
        return float(value)

    def record_close(self, plan: ClosePlan) -> int:
        """Écrit la clôture et remplace le buffer. Retourne l'id du run."""
        ts = _utc_now()
        as_of = pd.Timestamp(plan.as_of).strftime("%Y-%m-%d")
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "UPDATE orders SET status = 'superseded' WHERE status = 'planned'"
            )
            cursor = conn.execute(
                """
                INSERT INTO runs (
                    ts, phase, as_of, equity, last_equity, day_pnl,
                    leverage, vol_upper, idm, circuit_breaker
                ) VALUES (?, 'close', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    as_of,
                    self._sql(plan.equity),
                    self._sql(plan.last_equity),
                    self._sql(plan.day_pnl),
                    self._sql(plan.leverage),
                    self._sql(plan.vol_upper),
                    self._sql(plan.idm),
                    int(plan.circuit_breaker),
                ),
            )
            run_id = int(cursor.lastrowid)
            conn.executemany(
                """
                INSERT INTO targets (
                    run_id, ticker, forecast, vol_annual, weight, idm,
                    n_star, buffered, desired, held, price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        row["ticker"],
                        row["forecast"],
                        row["vol_annual"],
                        row["weight"],
                        row["idm"],
                        row["n_star"],
                        row["buffered"],
                        row["desired"],
                        row["held"],
                        row["price"],
                    )
                    for row in plan.targets
                ],
            )
            conn.executemany(
                """
                INSERT INTO orders (
                    run_id, ticker, side, qty, order_type, limit_price, status, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        order.ticker,
                        order.side,
                        order.qty,
                        order.order_type,
                        order.limit_price,
                        "planned" if order.status != "blocked" else "blocked",
                        order.reason,
                    )
                    for order in [*plan.orders, *plan.blocked]
                ],
            )
            for ticker, value in plan.buffered.items():
                conn.execute(
                    """
                    INSERT INTO buffer_state (ticker, buffered, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        buffered = excluded.buffered,
                        updated_at = excluded.updated_at
                    """,
                    (str(ticker), float(value), ts),
                )
            for message in plan.alerts:
                conn.execute(
                    "INSERT INTO anomalies (run_id, ts, kind, message) VALUES (?, ?, ?, ?)",
                    (run_id, ts, "risk", message),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return run_id

    def record_anomaly(self, kind: str, message: str, run_id: int | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO anomalies (run_id, ts, kind, message) VALUES (?, ?, ?, ?)",
                (run_id, _utc_now(), kind, message),
            )

    def last_close_date(self) -> date | None:
        """Date de marché de la dernière clôture journalisée."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT as_of FROM runs WHERE phase = 'close' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None or not row["as_of"]:
            return None
        return date.fromisoformat(row["as_of"])

    def load_buffer(self) -> pd.Series:
        with self._connect() as conn:
            rows = conn.execute("SELECT ticker, buffered FROM buffer_state").fetchall()
        if not rows:
            return pd.Series(dtype=float)
        return pd.Series({row["ticker"]: row["buffered"] for row in rows}, dtype=float)

    def planned_orders(self) -> list[sqlite3.Row]:
        """Ordres planned du dernier run de clôture."""
        with self._connect() as conn:
            run = conn.execute(
                "SELECT id FROM runs WHERE phase = 'close' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if run is None:
                return []
            return conn.execute(
                "SELECT * FROM orders WHERE run_id = ? AND status = 'planned' ORDER BY id",
                (run["id"],),
            ).fetchall()

    def mark_submitted(
        self,
        order_id: int,
        broker_id: str,
        limit_price: float | None,
        order_type: str,
        submitted_at: datetime,
    ) -> None:
        stamp = submitted_at.isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE orders
                SET status = 'submitted', broker_id = ?, limit_price = ?,
                    order_type = ?,
                    submitted_at = COALESCE(submitted_at, ?),
                    last_peg_at = ?
                WHERE id = ?
                """,
                (broker_id, limit_price, order_type, stamp, stamp, order_id),
            )

    def mark_status(self, order_id: int, status: str, reason: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE orders SET status = ?, reason = ? WHERE id = ?",
                (status, reason, order_id),
            )

    def working_orders(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT * FROM orders
                WHERE status = 'submitted' AND broker_id IS NOT NULL
                ORDER BY id
                """
            ).fetchall()
