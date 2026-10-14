"""
loader.py — téléchargement des données depuis Tiingo et FRED.

Responsabilités :
    - Appeler l'API Tiingo pour télécharger les prix EOD
    - Appeler l'API FRED pour télécharger les taux
    - Appliquer le décalage d'un jour sur les taux FRED (no look-ahead)
    - Déléguer le stockage à store.py

Ce module est le seul à accéder au réseau.
"""

import logging
import os
import time
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from fredapi import Fred
from tiingo import TiingoClient

from systrade.data import store

logger = logging.getLogger(__name__)

# Colonnes Tiingo qu'on conserve (les autres sont ignorées)
_TIINGO_COLS = ["close", "adjClose", "divCash", "splitFactor"]

# ── Helpers de configuration ──────────────────────────────────────────────────

def _load_config() -> dict:
    """Lit config/instruments.yaml et retourne le dictionnaire Python."""
    config_path = store.PROJECT_ROOT / "config" / "instruments.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_fred_series(config: dict) -> list[str]:
    """
    Extrait toutes les séries FRED nécessaires depuis la config.

    Inclut le taux cash (DTB3) et les proxys de carry obligataire
    (DGS7, DGS20, BAMLC0A0CMEY).
    """
    series: set[str] = set()
    series.add(config["carry"]["fred_cash_rate"])
    for instrument in config["universe"].values():
        if "carry_fred_series" in instrument:
            series.add(instrument["carry_fred_series"])
    return sorted(series)


# ── Normalisation des timestamps ──────────────────────────────────────────────

def _to_tz_naive(s: pd.Series) -> pd.Series:
    """
    Convertit une Series de timestamps en tz-naive.

    Tiingo renvoie des timestamps UTC ("2024-01-02T00:00:00+00:00").
    pd.to_datetime() les parse en tz-aware. On supprime le fuseau horaire
    ici car on ne travaille qu'avec des dates de clôture (pas d'heure).
    """
    if s.dt.tz is not None:
        return s.dt.tz_convert(None)
    return s


# ── Téléchargement des prix Tiingo ────────────────────────────────────────────

def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    api_key: str,
    sleep_between: float = 0.5,
) -> None:
    """
    Télécharge les prix EOD depuis Tiingo et les sauvegarde via store.save_raw().

    Paramètres
    ----------
    tickers :
        Liste de tickers (ex. ["SPY", "IWM", ...]).
    start_date, end_date :
        Dates ISO "YYYY-MM-DD".
    api_key :
        Clé API Tiingo.
    sleep_between :
        Pause en secondes entre chaque requête (tier gratuit Tiingo).

    Garantie no look-ahead
    ----------------------
    On sauvegarde exactement ce que Tiingo renvoie pour chaque date.
    Aucune transformation des prix ici.

    Gestion d'erreurs
    -----------------
    En cas d'échec sur un ticker, l'erreur est loggée et on continue.
    Le caller peut vérifier store.raw_path(ticker).exists() pour chaque ticker.
    """
    client = TiingoClient({"api_key": api_key, "session": True})

    for ticker in tickers:
        try:
            raw = client.get_ticker_price(
                ticker,
                startDate=start_date,
                endDate=end_date,
                fmt="json",
                frequency="daily",
            )
            if not raw:
                logger.warning("réponse vide pour %s", ticker)
                continue

            df = pd.DataFrame(raw)

            # Normaliser la colonne date en DatetimeIndex tz-naive
            df["date"] = _to_tz_naive(pd.to_datetime(df["date"]))
            df = df.set_index("date")

            # Vérifier que les colonnes attendues sont présentes
            missing = set(_TIINGO_COLS) - set(df.columns)
            if missing:
                raise ValueError(f"Colonnes manquantes : {missing}")

            store.save_raw(df[_TIINGO_COLS], ticker)
            logger.info("prix téléchargés : %s (%d lignes)", ticker, len(df))

        except Exception as exc:
            logger.error("échec du téléchargement de %s : %s", ticker, exc)

        time.sleep(sleep_between)


# ── Téléchargement des taux FRED ──────────────────────────────────────────────

def download_rates(
    series_ids: list[str],
    start_date: str,
    end_date: str,
    api_key: str,
) -> None:
    """
    Télécharge les taux FRED, applique le lag d'un jour, et sauvegarde
    via store.save_rates().

    No look-ahead
    -------------
    FRED publie ses séries avec un délai d'environ un jour ouvré.
    On applique .shift(1) pour que la valeur assignée au jour t soit
    celle publiée à t-1 (la seule connue à la clôture de t-1).

    Traitement des NaN
    ------------------
    FRED ne publie pas les week-ends ni les jours fériés.
    On applique ffill(limit=5) avant le shift pour propager la dernière
    valeur connue sur ces jours (maximum 5 jours consécutifs, soit une
    semaine entière de jours fériés).

    Paramètres
    ----------
    series_ids :
        Identifiants FRED (ex. ["DTB3", "DGS7", "DGS20", "BAMLC0A0CMEY"]).
    start_date, end_date :
        Dates ISO "YYYY-MM-DD".
    api_key :
        Clé API FRED.
    """
    fred = Fred(api_key=api_key)
    frames: dict[str, pd.Series] = {}

    # On recule de 5 jours pour avoir une valeur valide après le shift
    start_with_buffer = (
        pd.Timestamp(start_date) - pd.Timedelta(days=5)
    ).strftime("%Y-%m-%d")

    for series_id in series_ids:
        try:
            s = fred.get_series(
                series_id,
                observation_start=start_with_buffer,
                observation_end=end_date,
            )
            s.name = series_id
            frames[series_id] = s
            logger.info("taux FRED téléchargés : %s (%d obs.)", series_id, len(s))
        except Exception as exc:
            logger.error("échec FRED %s : %s", series_id, exc)

    if not frames:
        raise RuntimeError("Aucune série FRED téléchargée avec succès.")

    df = pd.DataFrame(frames)
    df.index.name = "date"

    # Étape 1 : propager les valeurs sur les week-ends / jours fériés
    df = df.ffill(limit=5)

    # Étape 2 : décaler d'une période (lag de publication)
    df = df.shift(1)

    # Étape 3 : tronquer à la période demandée (après le shift)
    df = df.loc[start_date:end_date]

    store.save_rates(df)


# ── Mise à jour quotidienne ───────────────────────────────────────────────────

def update(
    tiingo_key: Optional[str] = None,
    fred_key: Optional[str] = None,
) -> None:
    """
    Mise à jour quotidienne : télécharge uniquement les données manquantes.

    Détecte la dernière date disponible dans les fichiers raw/ et télécharge
    à partir de cette date moins 5 jours (buffer pour corriger d'éventuelles
    révisions de Tiingo).

    À appeler chaque soir après la clôture (~16h30 ET).
    """
    load_dotenv()
    tiingo_key = tiingo_key or os.getenv("TIINGO_API_KEY")
    fred_key   = fred_key   or os.getenv("FRED_API_KEY")

    if not tiingo_key or not fred_key:
        raise ValueError("Clés API manquantes. Vérifier le fichier .env.")

    config  = _load_config()
    tickers = list(config["universe"].keys())

    # Trouver la dernière date disponible dans les fichiers raw
    last_date = _get_last_available_date(tickers)
    start_date = (last_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")

    logger.info("mise à jour depuis %s jusqu'à %s", start_date, end_date)

    download_prices(tickers, start_date, end_date, tiingo_key, sleep_between=0.2)
    download_rates(_get_fred_series(config), start_date, end_date, fred_key)
    store.build_clean_layers()


def _get_last_available_date(tickers: list[str]) -> pd.Timestamp:
    """
    Retourne la date la plus récente disponible dans les fichiers raw.
    Utilisé par update() pour ne télécharger que le delta manquant.
    """
    last_dates = []
    for ticker in tickers:
        path = store.raw_path(ticker)
        if path.exists():
            df = pd.read_parquet(path)
            last_dates.append(df.index.max())

    if not last_dates:
        raise FileNotFoundError(
            "Aucun fichier raw trouvé. Lancer download_all() d'abord."
        )
    return min(last_dates)  # date min pour couvrir tous les tickers


# ── Point d'entrée principal (setup initial) ──────────────────────────────────

def download_all(
    start_date: str = "2007-01-01",
    end_date: Optional[str] = None,
    tiingo_key: Optional[str] = None,
    fred_key: Optional[str] = None,
) -> None:
    """
    Télécharge l'intégralité des données (prix + taux) et reconstruit
    les couches clean.

    À appeler une seule fois lors du setup initial. Ensuite, utiliser
    update() pour les mises à jour quotidiennes.

    Paramètres
    ----------
    start_date : date de début (défaut : "2007-01-01")
    end_date   : date de fin (défaut : aujourd'hui)
    tiingo_key : clé Tiingo (défaut : variable d'env TIINGO_API_KEY)
    fred_key   : clé FRED (défaut : variable d'env FRED_API_KEY)
    """
    load_dotenv()
    tiingo_key = tiingo_key or os.getenv("TIINGO_API_KEY")
    fred_key   = fred_key   or os.getenv("FRED_API_KEY")

    if not tiingo_key:
        raise ValueError("TIINGO_API_KEY manquante. Vérifier le fichier .env.")
    if not fred_key:
        raise ValueError("FRED_API_KEY manquante. Vérifier le fichier .env.")

    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    config   = _load_config()

    logger.info("=== Téléchargement des prix (%d tickers) ===", len(config["universe"]))
    download_prices(
        tickers    = list(config["universe"].keys()),
        start_date = start_date,
        end_date   = end_date,
        api_key    = tiingo_key,
    )

    fred_series = _get_fred_series(config)
    logger.info("=== Téléchargement des taux FRED (%d séries) ===", len(fred_series))
    download_rates(
        series_ids = fred_series,
        start_date = start_date,
        end_date   = end_date,
        api_key    = fred_key,
    )

    logger.info("=== Construction des couches clean ===")
    store.build_clean_layers()
    logger.info("=== Téléchargement terminé ===")
