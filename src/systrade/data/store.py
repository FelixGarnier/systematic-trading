"""
store.py — lecture et écriture des fichiers Parquet.

Responsabilités :
    - Définir les chemins de tous les fichiers de données
    - Sauvegarder les données brutes Tiingo (une par ticker)
    - Construire les couches clean (format large : dates × tickers)
    - Charger les données clean pour les modules en aval

Aucun calcul ici. Ce module ne fait que déplacer des données.
"""

from pathlib import Path

import pandas as pd

# ── Chemins ───────────────────────────────────────────────────────────────────
#
# Arborescence depuis src/systrade/data/store.py :
#   parents[0] = src/systrade/data/
#   parents[1] = src/systrade/
#   parents[2] = src/
#   parents[3] = racine du projet (là où est pyproject.toml)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
CLEAN_DIR    = DATA_DIR / "clean"

# Fichiers clean (format large : DatetimeIndex × tickers)
PRICES_ADJ_PATH   = CLEAN_DIR / "prices_adj.parquet"    # adjClose
PRICES_UNADJ_PATH = CLEAN_DIR / "prices_unadj.parquet"  # close
DIVIDENDS_PATH    = CLEAN_DIR / "dividends.parquet"      # divCash
RATES_PATH        = CLEAN_DIR / "rates.parquet"          # taux FRED


def raw_path(ticker: str) -> Path:
    """Chemin du fichier raw pour un ticker donné."""
    return RAW_DIR / f"{ticker}_raw.parquet"


# ── Écriture ──────────────────────────────────────────────────────────────────

def save_raw(df: pd.DataFrame, ticker: str) -> None:
    """
    Sauvegarde les données brutes Tiingo pour un ticker.

    Paramètres
    ----------
    df :
        DataFrame retourné par loader.download_prices(), avec DatetimeIndex
        et colonnes : close, adjClose, divCash, splitFactor, ...
    ticker :
        Symbole du ticker (ex. "SPY").

    Garantie
    --------
    Écrase le fichier existant. La couche raw est immuable par convention :
    on ne modifie jamais les données après les avoir sauvegardées ici.
    Appeler build_clean_layers() après pour mettre à jour la couche clean.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path(ticker))


def save_rates(df: pd.DataFrame) -> None:
    """
    Sauvegarde les taux FRED dans clean/rates.parquet.

    Paramètres
    ----------
    df :
        DataFrame retourné par loader.download_rates(), avec DatetimeIndex
        tz-naive et une colonne par série FRED (ex. "DTB3", "DGS7", ...).
        Les taux sont déjà décalés d'un jour (lag de publication FRED).
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RATES_PATH)


def build_clean_layers() -> None:
    """
    Construit les trois fichiers clean/ à partir des fichiers raw/.

    Lit tous les fichiers {TICKER}_raw.parquet présents dans data/raw/ et
    produit trois DataFrames en format large (dates × tickers) :
        - clean/prices_adj.parquet   ← adjClose
        - clean/prices_unadj.parquet ← close
        - clean/dividends.parquet    ← divCash

    Cette fonction est idempotente : l'appeler plusieurs fois donne le même
    résultat. À appeler après chaque téléchargement (setup initial ou mise
    à jour quotidienne en live).

    Raises
    ------
    FileNotFoundError
        Si aucun fichier raw n'est trouvé dans RAW_DIR.
    KeyError
        Si un fichier raw ne contient pas les colonnes attendues.
    """
    raw_files = sorted(RAW_DIR.glob("*_raw.parquet"))
    if not raw_files:
        raise FileNotFoundError(
            f"Aucun fichier raw trouvé dans {RAW_DIR}. "
            "Lancer d'abord loader.download_all()."
        )

    adj:   dict[str, pd.Series] = {}
    unadj: dict[str, pd.Series] = {}
    div:   dict[str, pd.Series] = {}

    for path in raw_files:
        ticker = path.stem.replace("_raw", "")
        df = pd.read_parquet(path)

        # Normaliser l'index en DatetimeIndex tz-naive
        # Tiingo renvoie des timestamps UTC ; on supprime le fuseau horaire
        # car on ne travaille qu'avec des dates de clôture (pas d'heure)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "date"

        adj[ticker]   = df["adjClose"]
        unadj[ticker] = df["close"]
        div[ticker]   = df["divCash"]

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(adj).sort_index().to_parquet(PRICES_ADJ_PATH)
    pd.DataFrame(unadj).sort_index().to_parquet(PRICES_UNADJ_PATH)
    pd.DataFrame(div).sort_index().to_parquet(DIVIDENDS_PATH)


# ── Lecture ───────────────────────────────────────────────────────────────────

def load_prices_adj() -> pd.DataFrame:
    """
    Prix ajustés (total return), format large (dates × tickers).

    Utiliser pour : calcul des rendements, volatilité, signaux EWMAC et carry.
    Ne jamais utiliser un niveau de prix ajusté seul pour le sizing.
    """
    return pd.read_parquet(PRICES_ADJ_PATH)


def load_prices_unadj() -> pd.DataFrame:
    """
    Prix bruts, format large (dates × tickers).

    Utiliser pour : conversion dollars ↔ nombre d'actions (sizing et P&L).
    """
    return pd.read_parquet(PRICES_UNADJ_PATH)


def load_dividends() -> pd.DataFrame:
    """
    Dividendes versés par date, format large (dates × tickers).

    Utiliser pour : calcul du carry des ETF actions (dividendes 12 derniers mois).
    0.0 les jours sans dividende.
    """
    return pd.read_parquet(DIVIDENDS_PATH)


def load_rates() -> pd.DataFrame:
    """
    Taux FRED décalés d'un jour, format large (dates × séries).

    Colonnes : DTB3, DGS7, DGS20, BAMLC0A0CMEY.
    Utiliser pour : taux cash (DTB3) et proxys de carry obligataire.
    """
    return pd.read_parquet(RATES_PATH)
