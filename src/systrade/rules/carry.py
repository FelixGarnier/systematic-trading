"""
src/systrade/rules/carry.py
────────────────────────────
Règle de carry — Bloc 3.

Instruments éligibles :
    Equity (dividendes - taux cash) : SPY, IWM, VGK, EWJ, EEM, EPP, EWC
    Bonds  (spread de taux)         : IEF (DGS7), TLT (DGS20), LQD (BAMLC0A0CMEY)
    Trend-only (pas de carry)       : GLD, DBC, UUP, TIP

Formule :
    Carry brut équity  = sum(dividendes 252j) / P_brut  − DTB3 / 100
    Carry brut bond    = (taux_instrument − DTB3) / 100
    Carry normalisé    = Carry brut / σ_annual
    Carry Medium       = EMA_5 (Carry normalisé)
    Carry Slow         = EMA_20(Carry normalisé)

Anti look-ahead :
    • sum(dividendes) : rolling sur les 252 DERNIERS jours (rolling window passé).
    • Taux FRED déjà décalés d'1 jour ouvré à l'ingestion (bloc 1b) : point-in-time.
    • P_brut : prix de clôture de t, connu à t → pas de problème.
    • σ_annual : calculé sur données ≤ t (bloc 2).
"""

from __future__ import annotations

import pandas as pd

# ── Constantes (reflètent la config instruments.yaml) ────────────────────────

EQUITY_CARRY_TICKERS: list[str] = ["SPY", "IWM", "VGK", "EWJ", "EEM", "EPP", "EWC"]

# instrument → colonne FRED correspondante
BOND_CARRY_MAP: dict[str, str] = {
    "IEF": "DGS7",
    "TLT": "DGS20",
    "LQD": "BAMLC0A0CMEY",
}

CARRY_SPANS: dict[str, int] = {
    "carry_medium": 5,
    "carry_slow":   20,
}


# ── Carry brut ────────────────────────────────────────────────────────────────

def equity_carry_raw(
    dividends: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    cash_rate: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """
    Carry brut pour les ETF actions.

        Carry = TTM_div_yield − cash_rate
        TTM_div_yield = rolling_sum(dividendes, 252j) / P_brut

    Parameters
    ----------
    dividends    : dividendes bruts ($/action). NaN quand pas de versement.
    prices_unadj : prix bruts (non ajustés).
    cash_rate    : série DTB3 / 100 (taux sans risque journalier).
    window       : jours ouvrés pour la somme TTM (défaut 252).
    """
    tickers = [t for t in EQUITY_CARRY_TICKERS if t in dividends.columns]
    if not tickers:
        return pd.DataFrame(index=dividends.index)

    # fillna(0) : un NaN = jour sans dividende, pas une donnée manquante
    div = dividends[tickers].fillna(0.0)
    ttm_div = div.rolling(window=window, min_periods=1).sum()

    price = prices_unadj[tickers].copy()
    price = price.replace(0.0, float("nan"))

    div_yield = ttm_div / price                # adimensionnel
    carry = div_yield.sub(cash_rate, axis=0)   # broadcast sur les colonnes
    return carry


def bond_carry_raw(
    rates: pd.DataFrame,
    bond_carry_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Carry brut pour les ETF obligataires.

        Carry = (taux_instrument − DTB3) / 100

    Parameters
    ----------
    rates         : DataFrame FRED en pourcentage (ex : 5.25 = 5.25 %).
                    Colonnes attendues : DTB3, DGS7, DGS20, BAMLC0A0CMEY.
    bond_carry_map: mapping instrument → colonne FRED.
    """
    if bond_carry_map is None:
        bond_carry_map = BOND_CARRY_MAP

    cash_pct = rates["DTB3"]
    result: dict[str, pd.Series] = {}
    for ticker, fred_col in bond_carry_map.items():
        if fred_col in rates.columns:
            result[ticker] = (rates[fred_col] - cash_pct) / 100.0
    return pd.DataFrame(result, index=rates.index)


def carry_raw(
    dividends: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    rates: pd.DataFrame,
    equity_tickers: list[str] | None = None,
    bond_carry_map: dict[str, str] | None = None,
    window: int = 252,
) -> pd.DataFrame:
    """
    Concatène les carries equity et obligataires.

    Returns
    -------
    DataFrame avec une colonne par instrument éligible, indexé sur `dividends.index`.
    """
    cash_rate = rates["DTB3"] / 100.0

    eq = equity_carry_raw(
        dividends,
        prices_unadj,
        cash_rate,
        window=window,
    )
    bo = bond_carry_raw(rates, bond_carry_map)

    combined = pd.concat([eq, bo], axis=1)
    combined = combined.reindex(dividends.index)  # aligne sur le calendrier actions
    return combined


# ── Normalisation & lissage ───────────────────────────────────────────────────

def normalize_carry(
    carry_brut: pd.DataFrame,
    vol_annual: pd.DataFrame,
) -> pd.DataFrame:
    """
    carry_normalisé = carry_brut / σ_annual.

    Seules les colonnes présentes dans les deux DataFrames sont traitées.
    """
    common = carry_brut.columns.intersection(vol_annual.columns)
    denom = vol_annual[common].replace(0.0, float("nan"))
    return carry_brut[common] / denom


def smooth_carry(
    normalized: pd.DataFrame,
    span_medium: int = 5,
    span_slow: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lisse le carry normalisé par EMA.

    Les vitesses viennent du lissage du signal, PAS de fenêtres de volatilité
    différentes (évite de réintroduire le ghost effect du bloc 2).

    Returns
    -------
    (carry_medium, carry_slow) — mêmes colonnes que `normalized`.
    """
    medium = normalized.ewm(span=span_medium, min_periods=span_medium).mean()
    slow   = normalized.ewm(span=span_slow,   min_periods=span_slow).mean()
    return medium, slow


# ── Point d'entrée principal ──────────────────────────────────────────────────

def all_carry_signals(
    dividends: pd.DataFrame,
    prices_unadj: pd.DataFrame,
    rates: pd.DataFrame,
    vol_annual: pd.DataFrame,
    equity_tickers: list[str] | None = None,
    bond_carry_map: dict[str, str] | None = None,
    window: int = 252,
) -> dict[str, pd.DataFrame]:
    """
    Pipeline complet : brut → normalisé → lissé.

    Parameters
    ----------
    dividends    : dividendes bruts (tickers en colonnes).
    prices_unadj : prix bruts.
    rates        : taux FRED (DTB3, DGS7, DGS20, BAMLC0A0CMEY).
    vol_annual   : volatilité annualisée (bloc 2).

    Returns
    -------
    dict avec clés "carry_medium" et "carry_slow".
    Chaque DataFrame contient uniquement les colonnes des instruments éligibles.
    """
    brut = carry_raw(dividends, prices_unadj, rates, equity_tickers, bond_carry_map, window)
    norm = normalize_carry(brut, vol_annual)
    medium, slow = smooth_carry(norm)
    return {"carry_medium": medium, "carry_slow": slow}
