"""
checks.py — contrôles d'intégrité des données.

Principe : signaler sans modifier.
Chaque contrôle retourne une liste de DataWarning.
Aucune exception n'est levée, aucune donnée n'est corrigée.

Usage :
    from systrade.data import checks, store
    warnings = checks.run_all_checks(
        store.load_prices_adj(),
        store.load_prices_unadj(),
    )
    checks.print_warnings(warnings)
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ── Structure d'un avertissement ──────────────────────────────────────────────

@dataclass
class DataWarning:
    """Représente une anomalie détectée dans les données."""
    check:   str             # nom du contrôle ("missing_days", "stale", ...)
    ticker:  str             # instrument concerné
    dates:   list            # dates affectées
    message: str             # description lisible

    def __str__(self) -> str:
        n = len(self.dates)
        first = self.dates[0] if n else "—"
        last  = self.dates[-1] if n > 1 else ""
        span  = f"{first} … {last}" if last else str(first)
        return f"[{self.check}] {self.ticker:6s} | {n} occurrence(s) | {span} | {self.message}"


# ── Contrôle 1 : jours manquants ──────────────────────────────────────────────

def check_missing_days(prices: pd.DataFrame) -> list[DataWarning]:
    """
    Détecte les jours où un ticker a une valeur NaN alors que d'autres
    tickers ont des données ce jour-là.

    Logique
    -------
    On construit un masque "jour actif" : tout jour où au moins un ticker
    a une valeur non-NaN. Pour chaque ticker, on cherche les jours actifs
    où ce ticker est NaN.

    Note : les NaN en début de série (avant le lancement de l'ETF, ex. UUP
    avant mars 2007) sont attendus et signalés comme tels.
    """
    warnings = []
    active_days = prices.notna().any(axis=1)

    for ticker in prices.columns:
        missing = prices.index[active_days & prices[ticker].isna()]
        if len(missing) == 0:
            continue

        # Distinguer les NaN en début de série (ETF non lancé)
        # des NaN au milieu (données vraiment manquantes)
        first_valid = prices[ticker].first_valid_index()
        if first_valid is None:
            pre_launch = missing
            mid_series = pd.DatetimeIndex([])
        else:
            pre_launch = missing[missing < first_valid]
            mid_series = missing[missing >= first_valid]

        if len(pre_launch):
            warnings.append(DataWarning(
                check="missing_days",
                ticker=ticker,
                dates=pre_launch.tolist(),
                message=f"ETF non lancé ({len(pre_launch)} jours avant {first_valid.date()})",
            ))
        if len(mid_series):
            warnings.append(DataWarning(
                check="missing_days",
                ticker=ticker,
                dates=mid_series.tolist(),
                message=f"données manquantes en cours de série ({len(mid_series)} jours)",
            ))

    return warnings


# ── Contrôle 2 : prix figés ───────────────────────────────────────────────────

def check_stale_prices(prices: pd.DataFrame, max_consecutive: int = 5) -> list[DataWarning]:
    """
    Détecte les séquences où le prix ne change pas pendant N jours consécutifs.

    Un prix figé sur plusieurs jours signale une donnée corrompue ou un
    instrument non tradé (possible pour des ETF très peu liquides).

    Paramètres
    ----------
    max_consecutive :
        Nombre minimum de jours consécutifs sans changement pour déclencher
        un avertissement (défaut : 5, soit une semaine entière).
    """
    warnings = []

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < max_consecutive:
            continue

        # True quand le prix change, False quand il reste identique
        changed  = series.diff().abs() > 0
        # Numéro de "groupe" : incrémente à chaque changement
        group_id = changed.cumsum()
        # Taille de chaque groupe de jours sans changement
        group_sizes = group_id.groupby(group_id).transform("count")

        stale_dates = series.index[group_sizes >= max_consecutive].tolist()
        if stale_dates:
            warnings.append(DataWarning(
                check="stale_prices",
                ticker=ticker,
                dates=stale_dates,
                message=(
                    f"prix inchangé ≥ {max_consecutive} jours consécutifs "
                    f"({len(stale_dates)} jours concernés)"
                ),
            ))

    return warnings


# ── Contrôle 3 : rendements extrêmes ──────────────────────────────────────────

def check_extreme_returns(
    prices: pd.DataFrame,
    threshold_sigma: float = 8.0,
    vol_window: int = 60,
    min_periods: int = 30,
) -> list[DataWarning]:
    """
    Détecte les rendements dont la valeur absolue dépasse threshold_sigma
    fois l'écart-type roulant.

    Paramètres
    ----------
    threshold_sigma :
        Seuil en nombre d'écarts-types (défaut : 8).
    vol_window :
        Fenêtre de l'écart-type roulant en jours (défaut : 60).
    min_periods :
        Nombre minimum d'observations pour produire un écart-type
        (défaut : 30 — les premières semaines sont ignorées).

    Note
    ----
    On ne supprime jamais ces rendements. Certains sont légitimes
    (crash 2008, COVID mars 2020). Le but est de les inspecter manuellement.
    """
    warnings = []
    returns = prices.pct_change()

    for ticker in prices.columns:
        r  = returns[ticker].dropna()
        if len(r) < min_periods:
            continue

        rolling_std = r.rolling(window=vol_window, min_periods=min_periods).std().shift(1)
        extreme     = r.index[r.abs() > threshold_sigma * rolling_std]

        if len(extreme):
            examples = [
                f"{d.date()}: {returns.loc[d, ticker]:.1%}"
                for d in extreme[:3]   # afficher au plus 3 exemples
            ]
            warnings.append(DataWarning(
                check="extreme_returns",
                ticker=ticker,
                dates=extreme.tolist(),
                message=(
                    f"|r| > {threshold_sigma}σ détecté {len(extreme)} fois. "
                    f"Exemples : {', '.join(examples)}"
                ),
            ))

    return warnings


# ── Contrôle 4 : prix non-positifs ────────────────────────────────────────────

def check_non_positive_prices(prices: pd.DataFrame) -> list[DataWarning]:
    """
    Détecte les prix nuls ou négatifs.

    Un prix ≤ 0 est une corruption de données — aucun ETF ne peut valoir 0.
    """
    warnings = []

    for ticker in prices.columns:
        bad_dates = prices.index[prices[ticker] <= 0].tolist()
        if bad_dates:
            warnings.append(DataWarning(
                check="non_positive_price",
                ticker=ticker,
                dates=bad_dates,
                message=f"prix ≤ 0 sur {len(bad_dates)} jour(s)",
            ))

    return warnings


# ── Runner principal ──────────────────────────────────────────────────────────

def run_all_checks(
    prices_adj:   pd.DataFrame,
    prices_unadj: pd.DataFrame,
) -> list[DataWarning]:
    """
    Lance les quatre contrôles et retourne la liste consolidée des avertissements.

    Paramètres
    ----------
    prices_adj :
        Prix ajustés (pour rendements et prix figés).
    prices_unadj :
        Prix bruts (pour la vérification des prix non-positifs).
    """
    warnings: list[DataWarning] = []
    warnings += check_missing_days(prices_adj)
    warnings += check_stale_prices(prices_adj)
    warnings += check_extreme_returns(prices_adj)
    warnings += check_non_positive_prices(prices_unadj)
    return warnings


# ── Affichage ─────────────────────────────────────────────────────────────────

def print_warnings(warnings: list[DataWarning]) -> None:
    """Affiche les avertissements groupés par type de contrôle."""
    if not warnings:
        print("✓ Aucun problème détecté.")
        return

    by_check: dict[str, list[DataWarning]] = {}
    for w in warnings:
        by_check.setdefault(w.check, []).append(w)

    for check_name, items in by_check.items():
        print(f"\n── {check_name} ({len(items)} avertissement(s)) ──")
        for w in items:
            print(f"  {w}")
