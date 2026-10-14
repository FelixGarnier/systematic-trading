"""
tests/test_store.py — tests unitaires pour src/systrade/data/store.py

Chaque test est isolé : on redirige les chemins vers tmp_path (dossier
temporaire pytest) pour ne jamais toucher aux vrais fichiers data/.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

import systrade.data.store as store


# ── Fixture : données factices ────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """DataFrame Tiingo minimal pour les tests (3 jours, colonnes essentielles)."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {
            "close":       [470.0, 471.5, 469.0],
            "adjClose":    [468.0, 469.5, 467.0],
            "divCash":     [0.0,   1.58,  0.0],
            "splitFactor": [1.0,   1.0,   1.0],
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


@pytest.fixture
def sample_raw_df_utc(sample_raw_df):
    """Même DataFrame mais avec un index UTC (comme Tiingo le renvoie parfois)."""
    df = sample_raw_df.copy()
    df.index = df.index.tz_localize("UTC")
    return df


@pytest.fixture
def patch_paths(tmp_path, monkeypatch):
    """Redirige tous les chemins store vers tmp_path."""
    raw_dir   = tmp_path / "raw"
    clean_dir = tmp_path / "clean"
    raw_dir.mkdir()
    clean_dir.mkdir()

    monkeypatch.setattr(store, "RAW_DIR",            raw_dir)
    monkeypatch.setattr(store, "CLEAN_DIR",          clean_dir)
    monkeypatch.setattr(store, "PRICES_ADJ_PATH",    clean_dir / "prices_adj.parquet")
    monkeypatch.setattr(store, "PRICES_UNADJ_PATH",  clean_dir / "prices_unadj.parquet")
    monkeypatch.setattr(store, "DIVIDENDS_PATH",     clean_dir / "dividends.parquet")
    monkeypatch.setattr(store, "RATES_PATH",         clean_dir / "rates.parquet")

    return {"raw": raw_dir, "clean": clean_dir}


# ── Tests save_raw ────────────────────────────────────────────────────────────

def test_save_raw_creates_file(patch_paths, sample_raw_df):
    """save_raw() doit créer le fichier {TICKER}_raw.parquet."""
    store.save_raw(sample_raw_df, "SPY")
    assert (patch_paths["raw"] / "SPY_raw.parquet").exists()


def test_save_raw_roundtrip(patch_paths, sample_raw_df):
    """Les données sauvegardées doivent être identiques après relecture."""
    store.save_raw(sample_raw_df, "SPY")
    loaded = pd.read_parquet(store.raw_path("SPY"))
    pd.testing.assert_frame_equal(loaded, sample_raw_df)


# ── Tests build_clean_layers ──────────────────────────────────────────────────

def test_build_clean_layers_creates_three_files(patch_paths, sample_raw_df):
    """build_clean_layers() doit produire les 3 fichiers clean."""
    store.save_raw(sample_raw_df, "SPY")
    store.save_raw(sample_raw_df, "IWM")
    store.build_clean_layers()

    assert store.PRICES_ADJ_PATH.exists()
    assert store.PRICES_UNADJ_PATH.exists()
    assert store.DIVIDENDS_PATH.exists()


def test_build_clean_layers_wide_format(patch_paths, sample_raw_df):
    """Les fichiers clean doivent être en format large (dates × tickers)."""
    store.save_raw(sample_raw_df, "SPY")
    store.save_raw(sample_raw_df, "IWM")
    store.build_clean_layers()

    adj = pd.read_parquet(store.PRICES_ADJ_PATH)
    assert set(adj.columns) == {"SPY", "IWM"}
    assert adj.index.name == "date"
    assert len(adj) == 3  # 3 jours dans sample_raw_df


def test_build_clean_layers_correct_values(patch_paths, sample_raw_df):
    """adjClose, close et divCash doivent être correctement extraits."""
    store.save_raw(sample_raw_df, "SPY")
    store.build_clean_layers()

    adj   = pd.read_parquet(store.PRICES_ADJ_PATH)
    unadj = pd.read_parquet(store.PRICES_UNADJ_PATH)
    div   = pd.read_parquet(store.DIVIDENDS_PATH)

    assert adj["SPY"].iloc[0]   == pytest.approx(468.0)
    assert unadj["SPY"].iloc[0] == pytest.approx(470.0)
    assert div["SPY"].iloc[1]   == pytest.approx(1.58)   # dividende le 2e jour


def test_build_clean_layers_utc_normalization(patch_paths, sample_raw_df_utc):
    """Un index UTC doit être converti en tz-naive dans la couche clean."""
    store.save_raw(sample_raw_df_utc, "SPY")
    store.build_clean_layers()

    adj = pd.read_parquet(store.PRICES_ADJ_PATH)
    assert adj.index.tz is None


def test_build_clean_layers_raises_if_no_raw(patch_paths):
    """build_clean_layers() doit lever FileNotFoundError si raw/ est vide."""
    with pytest.raises(FileNotFoundError):
        store.build_clean_layers()


def test_build_clean_layers_is_idempotent(patch_paths, sample_raw_df):
    """Appeler build_clean_layers() deux fois doit donner le même résultat."""
    store.save_raw(sample_raw_df, "SPY")
    store.build_clean_layers()
    adj_first = pd.read_parquet(store.PRICES_ADJ_PATH)

    store.build_clean_layers()
    adj_second = pd.read_parquet(store.PRICES_ADJ_PATH)

    pd.testing.assert_frame_equal(adj_first, adj_second)


# ── Tests load_* ──────────────────────────────────────────────────────────────

def test_load_prices_adj(patch_paths, sample_raw_df):
    """load_prices_adj() doit retourner le même DataFrame que build_clean_layers()."""
    store.save_raw(sample_raw_df, "SPY")
    store.build_clean_layers()

    df = store.load_prices_adj()
    assert isinstance(df, pd.DataFrame)
    assert "SPY" in df.columns


def test_load_rates(patch_paths):
    """load_rates() doit retourner le DataFrame de taux sauvegardé par save_rates()."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    rates = pd.DataFrame(
        {"DTB3": [5.32, 5.31], "DGS7": [4.18, 4.19]},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    store.save_rates(rates)

    loaded = store.load_rates()
    pd.testing.assert_frame_equal(loaded, rates)


def test_save_rates_keeps_history_outside_update_window(patch_paths):
    """Une mise à jour courte ne doit pas effacer l'historique déjà stocké."""
    history = pd.DataFrame(
        {"DTB3": [1.0, 5.0]},
        index=pd.to_datetime(["2007-01-03", "2024-01-03"]),
    )
    store.save_rates(history)

    update = pd.DataFrame(
        {"DTB3": [5.5], "BAMLC0A0CMEY": [6.0]},
        index=pd.to_datetime(["2024-01-03"]),
    )
    store.save_rates(update)

    loaded = store.load_rates()
    assert loaded.loc["2007-01-03", "DTB3"] == pytest.approx(1.0)
    assert loaded.loc["2024-01-03", "DTB3"] == pytest.approx(5.5)
    assert loaded.loc["2024-01-03", "BAMLC0A0CMEY"] == pytest.approx(6.0)
