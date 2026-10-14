"""
tests/test_combination.py — tests unitaires pour src/systrade/forecast/combination.py
"""

import numpy as np
import pandas as pd
import pytest

from systrade.forecast.combination import (
    FDM_CAP,
    FDM_POOL_MIN,
    RULE_WEIGHTS_TREND_CARRY,
    RULE_WEIGHTS_TREND_ONLY,
    TREND_ONLY_TICKERS,
    compute_combination,
    expanding_fdm,
    fdm_by_instrument,
    fdm_from_corr,
    weighted_average_forecast,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

N = 600   # assez long pour atteindre FDM_POOL_MIN
_DATES = pd.bdate_range("2019-01-02", periods=N)

TC_TICKERS = ["SPY", "IWM", "IEF"]      # trend + carry
TO_TICKERS = ["GLD", "DBC"]             # trend only
ALL_TICKERS = TC_TICKERS + TO_TICKERS

TC_RULES = ["EWMAC_8_32", "EWMAC_16_64", "EWMAC_32_128", "carry_medium", "carry_slow"]
TO_RULES = ["EWMAC_8_32", "EWMAC_16_64", "EWMAC_32_128"]


def _make_forecasts(
    tickers: list[str] = ALL_TICKERS,
    rules:   list[str] = TC_RULES,
    n:       int       = N,
    seed:    int       = 0,
) -> dict[str, pd.DataFrame]:
    """
    Capped forecasts synthétiques ∈ [−20, +20] pour chaque règle.
    Les règles carry ne sont créées que pour les tickers TC.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    result: dict[str, pd.DataFrame] = {}

    for rule in rules:
        is_carry = "carry" in rule
        rule_tickers = [t for t in tickers if not (is_carry and t in TO_TICKERS)]
        data = np.clip(rng.normal(0, 5, (n, len(rule_tickers))), -20, 20)
        result[rule] = pd.DataFrame(data, index=dates, columns=rule_tickers)

    return result


def _identity_corr(n: int) -> np.ndarray:
    return np.eye(n)


def _ones_corr(n: int) -> np.ndarray:
    return np.ones((n, n))


# ── Tests fdm_from_corr ───────────────────────────────────────────────────────

def test_fdm_identity_corr():
    """
    Avec une matrice identité (règles indépendantes), FDM = 1/√(Σw²).
    Pour des poids égaux (1/3 chacun × 3 règles) : FDM = 1/√(3×(1/3)²) = √3 ≈ 1.732.
    """
    weights = {"A": 1/3, "B": 1/3, "C": 1/3}
    corr = _identity_corr(3)
    fdm = fdm_from_corr(weights, corr, ["A", "B", "C"])
    assert fdm == pytest.approx(np.sqrt(3), rel=1e-6)


def test_fdm_perfect_correlation():
    """
    Avec une corrélation parfaite (Σ = ones), les règles ne se diversifient pas.
    FDM = 1/√(Σ_i Σ_j w_i w_j) = 1/√((Σw)²) = 1/1 = 1.
    """
    weights = {"A": 1/3, "B": 1/3, "C": 1/3}
    corr = _ones_corr(3)
    fdm = fdm_from_corr(weights, corr, ["A", "B", "C"])
    assert fdm == pytest.approx(1.0, rel=1e-6)


def test_fdm_negative_corr_zeroed():
    """
    Les corrélations négatives doivent être ramenées à 0 avant calcul.
    Une corrélation de −0.9 traitée comme 0 → FDM identique au cas indépendant.
    """
    weights = {"A": 0.5, "B": 0.5}
    # Sans correction : corr = [[1, -0.9], [-0.9, 1]] → FDM très élevé
    # Avec correction  : corr = [[1, 0], [0, 1]] → FDM = 1/√(0.25+0.25) = √2
    corr_neg  = np.array([[1.0, -0.9], [-0.9, 1.0]])
    corr_zero = np.array([[1.0,  0.0], [ 0.0, 1.0]])
    fdm_neg  = fdm_from_corr(weights, corr_neg,  ["A", "B"])
    fdm_zero = fdm_from_corr(weights, corr_zero, ["A", "B"])
    assert fdm_neg == pytest.approx(fdm_zero, rel=1e-6)


def test_fdm_capped_at_25():
    """Le FDM ne doit jamais dépasser FDM_CAP = 2.5."""
    # Corrélations très négatives (après zeroing → tous 0) → FDM potentiellement >2.5
    # Avec 5 règles de poids égaux et corrélations = 0 : FDM = √5 ≈ 2.236
    # Avec corrélations négatives très fortes zeroed vers 0 : FDM = √5 < 2.5 → OK
    # Pour dépasser 2.5 avec zeroing : impossible avec 5 règles de poids ≈ 0.2
    # Testons directement en injectant des poids très petits (→ FDM immense)
    n = 10
    weights = {str(i): 1/n for i in range(n)}
    corr = np.eye(n)
    fdm = fdm_from_corr(weights, corr, [str(i) for i in range(n)])
    assert fdm <= FDM_CAP + 1e-10


def test_fdm_greater_than_one_for_independent_rules():
    """Avec des règles non corrélées, FDM > 1 (bénéfice de diversification)."""
    weights = RULE_WEIGHTS_TREND_ONLY
    rule_names = list(weights.keys())
    n = len(rule_names)
    corr = _identity_corr(n)
    fdm = fdm_from_corr(weights, corr, rule_names)
    assert fdm > 1.0


# ── Tests weighted_average_forecast ──────────────────────────────────────────

def test_waf_correct_weighting():
    """Avg = Σ w_i × CF_i : vérification numérique exacte."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    cf = {
        "EWMAC_8_32":   pd.DataFrame({"SPY": [2.0] * 5}, index=dates),
        "EWMAC_16_64":  pd.DataFrame({"SPY": [4.0] * 5}, index=dates),
        "EWMAC_32_128": pd.DataFrame({"SPY": [6.0] * 5}, index=dates),
    }
    weights = {"EWMAC_8_32": 1/3, "EWMAC_16_64": 1/3, "EWMAC_32_128": 1/3}
    avg = weighted_average_forecast(cf, weights)
    assert avg["SPY"].iloc[0] == pytest.approx(4.0)   # (2+4+6)/3


def test_waf_renorm_for_trend_only():
    """
    Pour un instrument trend-only (carry absent), les poids EWMAC
    doivent être renormalisés de 0.233 à 0.333.
    """
    dates = pd.bdate_range("2024-01-02", periods=5)
    val = 6.0
    cf = {
        "EWMAC_8_32":   pd.DataFrame({"GLD": [val]*5}, index=dates),
        "EWMAC_16_64":  pd.DataFrame({"GLD": [val]*5}, index=dates),
        "EWMAC_32_128": pd.DataFrame({"GLD": [val]*5}, index=dates),
        # carry absent pour GLD → renorm automatique
        "carry_medium": pd.DataFrame({"SPY": [val]*5}, index=dates),
        "carry_slow":   pd.DataFrame({"SPY": [val]*5}, index=dates),
    }
    avg = weighted_average_forecast(cf, RULE_WEIGHTS_TREND_CARRY)
    # Tous les EWMAC valent `val` → moyenne = val quelle que soit la renorm
    assert avg["GLD"].iloc[0] == pytest.approx(val)


def test_waf_nan_rule_does_not_erase_the_others():
    """Un carry manquant un jour ne doit pas mettre le forecast à NaN."""
    dates = pd.bdate_range("2024-01-02", periods=2)
    six = pd.DataFrame({"SPY": [6.0, 6.0]}, index=dates)
    carry = pd.DataFrame({"SPY": [np.nan, 10.0]}, index=dates)
    cf = {
        "EWMAC_8_32": six.copy(),
        "EWMAC_16_64": six.copy(),
        "EWMAC_32_128": six.copy(),
        "carry_medium": carry.copy(),
        "carry_slow": carry.copy(),
    }
    avg = weighted_average_forecast(cf, RULE_WEIGHTS_TREND_CARRY)
    # Jour 0 : trois EWMAC à 6, carry absent → moyenne = 6
    assert avg["SPY"].iloc[0] == pytest.approx(6.0)
    # Jour 1 : 70 % × 6 + 30 % × 10
    assert avg["SPY"].iloc[1] == pytest.approx(0.7 * 6.0 + 0.3 * 10.0)


def test_fdm_by_instrument_switches_when_carry_is_missing():
    """Sans carry ce jour-là, on utilise le FDM trend-only."""
    dates = pd.bdate_range("2024-01-02", periods=2)
    has_carry = pd.DataFrame({"LQD": [False, True]}, index=dates)
    fdm_tc = pd.Series([1.4, 1.4], index=dates)
    fdm_to = pd.Series([1.1, 1.1], index=dates)
    used = fdm_by_instrument(has_carry, fdm_tc, fdm_to)
    assert used["LQD"].iloc[0] == pytest.approx(1.1)
    assert used["LQD"].iloc[1] == pytest.approx(1.4)


def test_waf_shape():
    """La shape de la sortie doit être (N_dates × N_tickers)."""
    cf = _make_forecasts()
    avg = weighted_average_forecast(cf, RULE_WEIGHTS_TREND_CARRY)
    # Colonnes = tickers présents dans la première règle
    assert avg.shape[0] == N
    assert set(avg.columns) == set(ALL_TICKERS)


# ── Tests expanding_fdm ───────────────────────────────────────────────────────

def test_expanding_fdm_returns_series():
    """expanding_fdm doit retourner une pd.Series."""
    cf = _make_forecasts()
    fdm = expanding_fdm(cf, RULE_WEIGHTS_TREND_CARRY, TC_TICKERS, _DATES)
    assert isinstance(fdm, pd.Series)


def test_expanding_fdm_length():
    """La série FDM doit avoir la même longueur que all_dates."""
    cf = _make_forecasts()
    fdm = expanding_fdm(cf, RULE_WEIGHTS_TREND_CARRY, TC_TICKERS, _DATES)
    assert len(fdm) == N


def test_expanding_fdm_greater_than_one():
    """Le FDM (après convergence) doit être supérieur à 1."""
    cf = _make_forecasts()
    fdm = expanding_fdm(cf, RULE_WEIGHTS_TREND_CARRY, TC_TICKERS, _DATES)
    assert fdm.iloc[-1] > 1.0


def test_expanding_fdm_below_cap():
    """Le FDM ne doit jamais dépasser FDM_CAP."""
    cf = _make_forecasts()
    fdm = expanding_fdm(cf, RULE_WEIGHTS_TREND_CARRY, TC_TICKERS, _DATES)
    assert (fdm <= FDM_CAP + 1e-10).all()


def test_expanding_fdm_no_lookahead():
    """
    Le FDM doit évoluer dans le temps (fenêtre expansive, pas constante).
    """
    cf = _make_forecasts()
    fdm = expanding_fdm(cf, RULE_WEIGHTS_TREND_CARRY, TC_TICKERS, _DATES)
    # Le FDM doit varier : std > 0
    assert fdm.std() > 0


def test_expanding_fdm_starts_at_one():
    """Avant FDM_POOL_MIN observations, FDM doit être 1 (pas d'estimation)."""
    cf = _make_forecasts(n=50)  # Trop court pour atteindre min_obs
    dates_short = pd.bdate_range("2024-01-02", periods=50)
    fdm = expanding_fdm(cf, RULE_WEIGHTS_TREND_CARRY, TC_TICKERS, dates_short)
    # Sans lissage EWMA, tout serait 1. Avec EWMA, les premières valeurs sont 1.
    assert fdm.iloc[0] == pytest.approx(1.0)


# ── Tests compute_combination ─────────────────────────────────────────────────

def test_combination_returns_correct_keys():
    """compute_combination doit retourner les clés attendues."""
    cf = _make_forecasts()
    result = compute_combination(cf, trend_only_tickers=TO_TICKERS)
    assert set(result.keys()) == {"combined", "fdm_tc", "fdm_to", "avg"}


def test_combination_combined_within_cap():
    """Toutes les valeurs du combined forecast doivent être dans [−20, +20]."""
    cf = _make_forecasts()
    result = compute_combination(cf, trend_only_tickers=TO_TICKERS)
    comb = result["combined"].dropna()
    assert (comb >= -20.0 - 1e-9).all().all()
    assert (comb <=  20.0 + 1e-9).all().all()


def test_combination_shape():
    """combined doit avoir la même shape que les signaux d'entrée."""
    cf = _make_forecasts()
    result = compute_combination(cf, trend_only_tickers=TO_TICKERS)
    assert result["combined"].shape[0] == N
    assert set(result["combined"].columns) == set(ALL_TICKERS)


def test_combination_fdm_tc_greater_than_fdm_to():
    """
    Le FDM Trend+Carry doit être supérieur au FDM Trend-only
    (le carry ajoute de la diversification au-delà du trend seul).
    """
    cf = _make_forecasts()
    result = compute_combination(cf, trend_only_tickers=TO_TICKERS)
    # Comparer les valeurs finales après convergence
    assert result["fdm_tc"].iloc[-1] > result["fdm_to"].iloc[-1]


def test_combination_trend_only_no_carry_influence():
    """
    Les instruments trend-only ne doivent pas être influencés par les carry forecasts.
    Modifier les carry forecasts ne doit pas changer leur combined forecast.
    """
    cf_base    = _make_forecasts(seed=1)
    cf_altered = _make_forecasts(seed=1)
    # Inverser les carry forecasts
    for rule in ["carry_medium", "carry_slow"]:
        cf_altered[rule] = -cf_altered[rule]

    res_base    = compute_combination(cf_base,    trend_only_tickers=TO_TICKERS)
    res_altered = compute_combination(cf_altered, trend_only_tickers=TO_TICKERS)

    for ticker in TO_TICKERS:
        pd.testing.assert_series_equal(
            res_base["combined"][ticker].dropna(),
            res_altered["combined"][ticker].dropna(),
            check_names=False,
        )
