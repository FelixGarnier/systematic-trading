"""
tests/test_idm.py — tests unitaires pour src/systrade/portfolio/idm.py
"""

import numpy as np
import pandas as pd
import pytest

from systrade.portfolio.idm import (
    IDM_CAP,
    active_weights,
    apply_idm,
    compute_idm,
    expanding_idm,
    idm_from_corr,
    instrument_weights,
    subsystem_returns,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

N = 400
_DATES = pd.bdate_range("2018-01-02", periods=N)
_TICKERS = ["SPY", "IEF", "GLD", "UUP"]


def _frame(values: np.ndarray, tickers: list[str] | None = None) -> pd.DataFrame:
    cols = tickers or _TICKERS[: values.shape[1]]
    return pd.DataFrame(values, index=_DATES[: values.shape[0]], columns=cols)


def _equal_weights(tickers: list[str]) -> dict[str, float]:
    return {t: 1.0 / len(tickers) for t in tickers}


# ── Poids handcraftés ─────────────────────────────────────────────────────────

def test_instrument_weights_sum_to_one():
    """Les poids de la config doivent sommer à 1."""
    weights = instrument_weights()
    assert sum(weights.values()) == pytest.approx(1.0)


def test_instrument_weights_universe_size():
    """L'univers V1 compte 14 instruments."""
    assert len(instrument_weights()) == 14


def test_instrument_weights_known_values():
    """Quelques poids de la table handcraftée."""
    weights = instrument_weights()
    assert weights["SPY"] == pytest.approx(0.10)
    assert weights["VGK"] == pytest.approx(0.04)
    assert weights["IEF"] == pytest.approx(0.0625)
    assert weights["GLD"] == pytest.approx(0.125)
    assert weights["UUP"] == pytest.approx(0.10)


def test_active_weights_renormalize_single_instrument():
    """Un seul instrument actif reçoit un poids de 1."""
    sub = pd.DataFrame(
        {"SPY": np.ones(N), "IEF": np.nan},
        index=_DATES,
    )
    w = active_weights(sub, {"SPY": 0.10, "IEF": 0.0625}, min_obs=10)
    assert w["SPY"].iloc[-1] == pytest.approx(1.0)
    assert w["IEF"].iloc[-1] == pytest.approx(0.0)


def test_active_weights_full_universe_keeps_handcrafted():
    """Quand tout le monde est actif, les proportions handcraftées sont conservées."""
    sub = pd.DataFrame(1.0, index=_DATES, columns=["SPY", "IEF"])
    fixed = {"SPY": 0.10, "IEF": 0.0625}
    w = active_weights(sub, fixed, min_obs=10)
    total = 0.10 + 0.0625
    assert w["SPY"].iloc[-1] == pytest.approx(0.10 / total)
    assert w["IEF"].iloc[-1] == pytest.approx(0.0625 / total)


def test_active_weights_zero_before_min_obs():
    """Avant min_obs, tous les poids sont nuls."""
    sub = pd.DataFrame(1.0, index=_DATES, columns=["SPY", "IEF"])
    w = active_weights(sub, {"SPY": 0.5, "IEF": 0.5}, min_obs=50)
    assert w.iloc[:49].to_numpy().sum() == pytest.approx(0.0)
    assert w.iloc[49].sum() == pytest.approx(1.0)


# ── IDM analytique ────────────────────────────────────────────────────────────

def test_idm_perfect_correlation_is_one():
    """Corrélation parfaite → aucune diversification → IDM = 1."""
    names = ["A", "B", "C"]
    weights = {n: 1 / 3 for n in names}
    idm = idm_from_corr(weights, np.ones((3, 3)), names)
    assert idm == pytest.approx(1.0)


def test_idm_identity_correlation():
    """Règles indépendantes, poids égaux : IDM = √n."""
    names = ["A", "B", "C", "D"]
    weights = {n: 0.25 for n in names}
    idm = idm_from_corr(weights, np.eye(4), names)
    assert idm == pytest.approx(2.0)


def test_idm_negative_corr_zeroed():
    """Une corrélation négative est traitée comme 0, pas comme de la diversification en plus."""
    weights = {"A": 0.5, "B": 0.5}
    corr_neg = np.array([[1.0, -0.8], [-0.8, 1.0]])
    corr_zero = np.eye(2)
    assert idm_from_corr(weights, corr_neg, ["A", "B"]) == pytest.approx(
        idm_from_corr(weights, corr_zero, ["A", "B"])
    )


def test_idm_capped():
    """20 instruments indépendants donneraient IDM ≈ 4.47, plafonné à 2.5."""
    n = 20
    names = [str(i) for i in range(n)]
    weights = {name: 1 / n for name in names}
    idm = idm_from_corr(weights, np.eye(n), names)
    assert idm == pytest.approx(IDM_CAP)


def test_idm_renormalizes_weights():
    """Des poids qui ne somment pas à 1 sont renormalisés avant le calcul."""
    weights = {"A": 0.10, "B": 0.10}
    idm = idm_from_corr(weights, np.eye(2), ["A", "B"])
    assert idm == pytest.approx(np.sqrt(2))


# ── Rendements de sous-systèmes ───────────────────────────────────────────────

def test_subsystem_returns_uses_lagged_forecast():
    """Le rendement du jour t dépend de F_{t-1}, pas de F_t."""
    prices = pd.DataFrame({"SPY": np.linspace(100, 110, 5)}, index=_DATES[:5])
    vol = pd.DataFrame({"SPY": 0.01}, index=_DATES[:5])
    forecast = pd.DataFrame({"SPY": [10.0, 10.0, 10.0, 10.0, 0.0]}, index=_DATES[:5])
    forecast_alt = forecast.copy()
    forecast_alt.iloc[-1, 0] = 20.0  # change seulement F au dernier jour

    base = subsystem_returns(forecast, prices, vol)
    alt = subsystem_returns(forecast_alt, prices, vol)
    # Le dernier rendement de sous-système utilise F de l'avant-dernier jour.
    assert base["SPY"].iloc[-1] == pytest.approx(alt["SPY"].iloc[-1])
    assert pd.isna(base["SPY"].iloc[0])


def test_subsystem_returns_first_rows_nan():
    """pct_change et le shift rendent les deux premières lignes incomplètes."""
    prices = pd.DataFrame({"SPY": np.linspace(100, 140, N)}, index=_DATES)
    vol = pd.DataFrame({"SPY": 0.01}, index=_DATES)
    forecast = pd.DataFrame({"SPY": 10.0}, index=_DATES)
    sub = subsystem_returns(forecast, prices, vol)
    assert pd.isna(sub["SPY"].iloc[0])
    assert sub["SPY"].iloc[1:].notna().all()


def test_subsystem_returns_scales_with_forecast():
    """Doubler le forecast décalé double le rendement de sous-système."""
    prices = pd.DataFrame({"SPY": 100 * np.cumprod(1 + np.full(N, 0.001))}, index=_DATES)
    vol = pd.DataFrame({"SPY": 0.01}, index=_DATES)
    f1 = pd.DataFrame({"SPY": 10.0}, index=_DATES)
    f2 = pd.DataFrame({"SPY": 20.0}, index=_DATES)
    s1 = subsystem_returns(f1, prices, vol)
    s2 = subsystem_returns(f2, prices, vol)
    ratio = (s2["SPY"] / s1["SPY"]).dropna()
    assert ratio.iloc[-1] == pytest.approx(2.0)


# ── IDM expansif ──────────────────────────────────────────────────────────────

def test_expanding_idm_one_when_perfectly_correlated():
    """Des sous-systèmes identiques convergent vers un IDM de 1."""
    rng = np.random.default_rng(0)
    common = rng.normal(0, 1, N)
    sub = _frame(np.column_stack([common, common, common, common]))
    idm = expanding_idm(sub, _equal_weights(_TICKERS), min_obs=30, ewma_span=1)
    assert idm.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_expanding_idm_above_one_when_independent():
    """Des sous-systèmes indépendants donnent un IDM supérieur à 1."""
    rng = np.random.default_rng(1)
    sub = _frame(rng.normal(0, 1, (N, 4)))
    idm = expanding_idm(sub, _equal_weights(_TICKERS), min_obs=30, ewma_span=1)
    assert idm.iloc[-1] > 1.5


def test_expanding_idm_starts_at_one():
    """Avant deux instruments estimables, l'IDM vaut 1."""
    rng = np.random.default_rng(2)
    sub = _frame(rng.normal(0, 1, (N, 4)))
    idm = expanding_idm(sub, _equal_weights(_TICKERS), min_obs=80, ewma_span=1)
    # À l'index 78 on a 79 observations : pas encore assez. À 79, on en a 80.
    assert idm.iloc[78] == pytest.approx(1.0)
    assert idm.iloc[79] > 1.0


def test_expanding_idm_no_lookahead():
    """Changer le futur ne change pas l'IDM passé."""
    rng = np.random.default_rng(3)
    sub = _frame(rng.normal(0, 1, (N, 4)))
    cut = 250
    full = expanding_idm(sub, _equal_weights(_TICKERS), min_obs=40, ewma_span=1)
    trunc = expanding_idm(sub.iloc[:cut], _equal_weights(_TICKERS), min_obs=40, ewma_span=1)
    assert full.iloc[cut - 1] == pytest.approx(trunc.iloc[-1])


def test_expanding_idm_capped():
    """L'IDM expansif ne dépasse pas 2.5."""
    rng = np.random.default_rng(4)
    tickers = [f"T{i}" for i in range(12)]
    sub = pd.DataFrame(rng.normal(0, 1, (N, 12)), index=_DATES, columns=tickers)
    idm = expanding_idm(sub, _equal_weights(tickers), min_obs=30, ewma_span=1)
    assert idm.max() <= IDM_CAP + 1e-9


# ── Application ───────────────────────────────────────────────────────────────

def test_apply_idm_multiplies():
    """N_final = N* × w × IDM."""
    positions = pd.DataFrame({"SPY": 100.0, "IEF": 40.0}, index=_DATES[:3])
    weights = pd.DataFrame({"SPY": 0.10, "IEF": 0.0625}, index=_DATES[:3])
    idm = pd.Series(2.0, index=_DATES[:3])
    out = apply_idm(positions, weights, idm)
    assert out["SPY"].iloc[0] == pytest.approx(20.0)
    assert out["IEF"].iloc[0] == pytest.approx(5.0)


def test_compute_idm_keys_and_scales_positions():
    """Le pipeline retourne les pièces attendues et réduit N* par le poids."""
    rng = np.random.default_rng(5)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (N, 2)), axis=0),
        index=_DATES,
        columns=["SPY", "IEF"],
    )
    vol = pd.DataFrame(0.01, index=_DATES, columns=["SPY", "IEF"])
    forecast = pd.DataFrame(10.0, index=_DATES, columns=["SPY", "IEF"])
    positions = pd.DataFrame(100.0, index=_DATES, columns=["SPY", "IEF"])
    weights = {"SPY": 0.6, "IEF": 0.4}

    result = compute_idm(
        positions, forecast, prices, vol, weights=weights, min_obs=30, ewma_span=1
    )
    assert set(result) == {
        "positions", "idm", "weights", "weights_fixed", "subsystem_returns"
    }
    # Univers complet → poids d'origine, et |N_final| < |N*| car w < 1 et IDM est borné.
    assert result["weights"]["SPY"].iloc[-1] == pytest.approx(0.6)
    assert result["positions"]["SPY"].abs().iloc[-1] < 100.0
    assert result["idm"].iloc[-1] <= IDM_CAP
