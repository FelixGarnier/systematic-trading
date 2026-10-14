"""
tests/test_loader.py — tests unitaires pour src/systrade/data/loader.py

Stratégie : les appels réseau (Tiingo, FRED) sont mockés.
On vérifie que loader transforme correctement les réponses API
et délègue le stockage à store.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import systrade.data.loader as loader
import systrade.data.store as store


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def patch_store_paths(tmp_path, monkeypatch):
    """Redirige les chemins store vers tmp_path (réutilisé depuis test_store)."""
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


def _mock_tiingo_response():
    """Réponse Tiingo simulée pour SPY sur 3 jours."""
    return [
        {"date": "2024-01-02T00:00:00+00:00", "close": 470.0, "adjClose": 468.0,
         "divCash": 0.0, "splitFactor": 1.0},
        {"date": "2024-01-03T00:00:00+00:00", "close": 471.5, "adjClose": 469.5,
         "divCash": 1.58, "splitFactor": 1.0},
        {"date": "2024-01-04T00:00:00+00:00", "close": 469.0, "adjClose": 467.0,
         "divCash": 0.0, "splitFactor": 1.0},
    ]


def _mock_fred_series(name: str) -> pd.Series:
    """Série FRED simulée sur 5 jours (y compris un week-end)."""
    # FRED ne publie pas le week-end : le vendredi 5 jan et le lundi 8 jan
    # ont la même valeur, mais le samedi/dimanche sont absents
    dates  = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04",
                              "2024-01-05", "2024-01-08"])
    values = [5.32, 5.31, 5.30, 5.30, 5.29]
    return pd.Series(values, index=dates, name=name)


# ── Tests _to_tz_naive ────────────────────────────────────────────────────────

def test_to_tz_naive_removes_utc():
    """Les timestamps UTC doivent être convertis en tz-naive."""
    s = pd.to_datetime(["2024-01-02T00:00:00+00:00"])
    s = pd.Series(s)
    result = loader._to_tz_naive(s)
    assert result.dt.tz is None


def test_to_tz_naive_passthrough_naive():
    """Les timestamps déjà tz-naive ne doivent pas être modifiés."""
    s = pd.Series(pd.to_datetime(["2024-01-02"]))
    result = loader._to_tz_naive(s)
    assert result.dt.tz is None
    assert result.iloc[0] == pd.Timestamp("2024-01-02")


# ── Tests _get_fred_series ────────────────────────────────────────────────────

def test_get_fred_series_includes_cash_rate():
    """DTB3 (taux cash) doit toujours être inclus."""
    config = loader._load_config()
    series = loader._get_fred_series(config)
    assert "DTB3" in series


def test_get_fred_series_includes_bond_proxies():
    """Les proxys obligataires (DGS7, DGS20, BAMLC0A0CMEY) doivent être inclus."""
    config = loader._load_config()
    series = loader._get_fred_series(config)
    for expected in ["DGS7", "DGS20", "BAMLC0A0CMEY"]:
        assert expected in series


# ── Tests download_prices ─────────────────────────────────────────────────────

@patch("systrade.data.loader.TiingoClient")
def test_download_prices_creates_raw_file(mock_client_class, patch_store_paths):
    """download_prices() doit créer le fichier raw pour chaque ticker."""
    mock_client = mock_client_class.return_value
    mock_client.get_ticker_price.return_value = _mock_tiingo_response()

    loader.download_prices(["SPY"], "2024-01-01", "2024-01-10", "fake_key", sleep_between=0)

    assert (patch_store_paths["raw"] / "SPY_raw.parquet").exists()


@patch("systrade.data.loader.TiingoClient")
def test_download_prices_correct_values(mock_client_class, patch_store_paths):
    """Les valeurs adjClose et divCash doivent être correctement extraites."""
    mock_client = mock_client_class.return_value
    mock_client.get_ticker_price.return_value = _mock_tiingo_response()

    loader.download_prices(["SPY"], "2024-01-01", "2024-01-10", "fake_key", sleep_between=0)

    df = pd.read_parquet(store.raw_path("SPY"))
    assert df["adjClose"].iloc[0] == pytest.approx(468.0)
    assert df["divCash"].iloc[1]  == pytest.approx(1.58)


@patch("systrade.data.loader.TiingoClient")
def test_download_prices_index_is_tz_naive(mock_client_class, patch_store_paths):
    """L'index du fichier raw doit être tz-naive après normalisation."""
    mock_client = mock_client_class.return_value
    mock_client.get_ticker_price.return_value = _mock_tiingo_response()

    loader.download_prices(["SPY"], "2024-01-01", "2024-01-10", "fake_key", sleep_between=0)

    df = pd.read_parquet(store.raw_path("SPY"))
    assert df.index.tz is None


@patch("systrade.data.loader.TiingoClient")
def test_download_prices_continues_on_failure(mock_client_class, patch_store_paths):
    """Un échec sur un ticker ne doit pas bloquer les autres."""
    mock_client = mock_client_class.return_value

    def side_effect(ticker, **kwargs):
        if ticker == "BAD":
            raise RuntimeError("ticker introuvable")
        return _mock_tiingo_response()

    mock_client.get_ticker_price.side_effect = side_effect

    # Pas d'exception levée
    loader.download_prices(["BAD", "SPY"], "2024-01-01", "2024-01-10", "fake_key", sleep_between=0)

    # SPY doit être sauvegardé malgré l'échec de BAD
    assert (patch_store_paths["raw"] / "SPY_raw.parquet").exists()
    assert not (patch_store_paths["raw"] / "BAD_raw.parquet").exists()


# ── Tests download_rates ──────────────────────────────────────────────────────

@patch("systrade.data.loader.Fred")
def test_download_rates_creates_rates_file(mock_fred_class, patch_store_paths):
    """download_rates() doit créer clean/rates.parquet."""
    mock_fred = mock_fred_class.return_value
    mock_fred.get_series.return_value = _mock_fred_series("DTB3")

    loader.download_rates(["DTB3"], "2024-01-02", "2024-01-08", "fake_key")

    assert (patch_store_paths["clean"] / "rates.parquet").exists()


@patch("systrade.data.loader.Fred")
def test_download_rates_shift_applied(mock_fred_class, patch_store_paths):
    """
    Le shift d'un jour doit être appliqué.

    La valeur du 2 jan (première ligne après shift) doit être NaN :
    il n'y a pas de valeur "la veille" du premier jour.
    """
    mock_fred = mock_fred_class.return_value
    mock_fred.get_series.return_value = _mock_fred_series("DTB3")

    loader.download_rates(["DTB3"], "2024-01-02", "2024-01-08", "fake_key")

    df = pd.read_parquet(store.RATES_PATH)
    # La première ligne doit être NaN (résultat du shift)
    assert pd.isna(df["DTB3"].iloc[0])


@patch("systrade.data.loader.Fred")
def test_download_rates_second_row_is_first_value(mock_fred_class, patch_store_paths):
    """
    Après le shift, la valeur du jour t doit être celle publiée à t-1.
    La 2e ligne doit contenir la valeur originale de la 1ère ligne.
    """
    mock_fred = mock_fred_class.return_value
    mock_fred.get_series.return_value = _mock_fred_series("DTB3")

    loader.download_rates(["DTB3"], "2024-01-02", "2024-01-08", "fake_key")

    df = pd.read_parquet(store.RATES_PATH)
    # Valeur originale du 2 jan = 5.32, doit apparaître le 3 jan après shift
    assert df["DTB3"].iloc[1] == pytest.approx(5.32)


# ── Tests download_all ────────────────────────────────────────────────────────

@patch("systrade.data.loader.store.build_clean_layers")
@patch("systrade.data.loader.download_rates")
@patch("systrade.data.loader.download_prices")
def test_download_all_calls_both_downloaders(mock_prices, mock_rates, mock_build, patch_store_paths):
    """download_all() doit appeler download_prices ET download_rates."""
    loader.download_all(
        start_date="2024-01-01",
        end_date="2024-01-10",
        tiingo_key="fake_tiingo",
        fred_key="fake_fred",
    )
    assert mock_prices.called
    assert mock_rates.called
    assert mock_build.called


@patch("systrade.data.loader.load_dotenv")
@patch("systrade.data.loader.os.getenv", return_value=None)
def test_download_all_raises_without_keys(mock_getenv, mock_dotenv):
    """download_all() sans clés API doit lever ValueError."""
    with pytest.raises(ValueError, match="TIINGO_API_KEY"):
        loader.download_all(tiingo_key=None, fred_key=None)
