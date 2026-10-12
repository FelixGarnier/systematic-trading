# Architecture — Systematic Trading Framework

Framework de trading systématique basé sur la méthodologie de Robert Carver
(*Systematic Trading*, *Advanced Futures Trading Strategies*).

Ce document est la référence de conception. Chaque bloc est validé avant toute implémentation.

**Principe non négociable : aucun look-ahead bias.** Toute grandeur utilisée à la date `t`
n'est calculée qu'avec des informations disponibles à la clôture de `t`.

---

## Sommaire

| # | Bloc | Statut |
|---|---|---|
| 0 | Objectifs & contraintes | Validé |
| 1a | Univers | Validé |
| 1b | Données | Validé |
| 2 | Volatilité | Validé |
| 3 | Règles de trading | Validé |
| 4 | Scaling & capping | Validé |
| 5 | Combinaison & FDM | Validé |
| 6 | Position sizing | Validé |
| 7 | Portefeuille & IDM | Validé |
| 8 | Buffer & coûts | Validé |
| 9 | Backtest & évaluation | En discussion |
| 10 | Exécution live & risque | À faire |

---

## Bloc 0 — Objectifs & contraintes

| Paramètre | Valeur |
|---|---|
| Objectif | Maximiser le Sharpe ratio à niveau de risque fixé ex ante |
| Vol target | 13 % annualisé (provisoire, recalibré au bloc 9) |
| Drawdown max toléré | ~25 % |
| Short | Autorisé, frais d'emprunt modélisés |
| Levier brut max | 2x |
| Capital | 100 000 USD, sizing sur le capital courant |
| Fréquence | Quotidienne : signal à la clôture `t`, exécution à `t+1` |
| Broker (live) | Alpaca (paper), derrière une `BrokerInterface` abstraite |
| Benchmarks | SPY buy & hold, 60/40, même univers long-only au même vol target |

**Justifications**

- *Vol target* : critère de Kelly (`σ* = SR`), on prend half-Kelly car le SR du backtest
  surestime le SR futur. Contrainte additionnelle : le drawdown attendu est à peu près
  proportionnel à la vol (Magdon-Ismail & Atiya, 2004). Simulation (rendements normaux,
  5 000 trajectoires) : à 20 % de vol et SR = 0,5, drawdown max médian sur 10 ans ≈ 39 %.
  À 13 %, ≈ 25 %.
- *Benchmark long-only au même vol target* (forecast constant = +10) : isole la valeur
  ajoutée des signaux de celle de l'allocation et du vol targeting.

---

## Bloc 1a — Univers

**Critères de sélection**

1. Coût faible (spread serré, volume quotidien élevé).
2. Apport de diversification (pas de doublon à corrélation ~0,95).
3. Historique suffisant (idéalement > 10 ans ; entrée en cours de backtest autorisée).
4. Shortable (easy-to-borrow). Short en actions entières uniquement chez Alpaca (à vérifier).
5. Structure saine : pas d'ETF à levier/inverses, pas d'ETN, pas d'ETP de volatilité.

**Univers V1 — 5 classes d'actifs**

| Classe | Instruments | Règles |
|---|---|---|
| Actions US | SPY, IWM | Trend + Carry |
| Actions internationales | VGK, EWJ, EEM, EPP, EWC | Trend + Carry |
| Obligations | IEF, TLT, TIP, LQD | Trend + Carry (sauf TIP : trend-only) |
| Matières premières | GLD, DBC | Trend uniquement |
| Devises | UUP | Trend uniquement |

SHY retiré (vol ~1,5 % / an → Volatility Scalar ~138x le capital, incompatible avec levier 2x, cf. bloc 6).
TIP : carry écarté (rendement réel structurellement négatif 2011–2022, signal biaisé) → trend-only.
Les corrélations seront revues au bloc 7 ; certains instruments pourront être fusionnés ou retirés.

**Biais reconnus**

- Biais de sélection / survie : univers choisi en 2026 parmi des ETF qui existent encore.
- Historique inégal (SPY 1993, TLT/IEF 2002, GLD 2004, DBC 2006, UUP 2007) :
  backtest pleinement représentatif à partir de ~2007.

---

## Bloc 1b — Données

**Sources**

| Donnée | Source | Usage |
|---|---|---|
| Prix EOD (brut, ajusté, dividendes, splits) | Tiingo (principal), yfinance (secours) | Backtest et live |
| Taux cash | FRED `DTB3` (T-bill 3 mois) | Carry, coût de financement |
| Proxys de rendement obligataire | FRED (voir ci-dessous) | Carry obligataire |
| Bid/ask intraday, exécution | Alpaca | Live uniquement |

Alpaca n'est pas utilisé pour l'historique (données depuis ~2016 seulement : pas de 2008,
pas assez d'historique pour estimer scalars et corrélations). Tiingo reste la source des
clôtures en live pour garantir la cohérence backtest / live.

**Deux séries de prix**

- Prix ajusté (total return) → rendements, volatilité, signaux. L'ajustement est multiplicatif :
  les ratios (rendements, EWMAC normalisé) sont invariants, donc pas de look-ahead.
  Un niveau de prix ajusté n'est jamais utilisé seul.
- Prix brut → conversion dollars ↔ nombre d'actions.
- P&L : `PnL_t = N_{t-1} × P_brut_{t-1} × r_total_t` (dividendes inclus, payés par les shorts).

**Données de carry (point-in-time)**

| Instrument | Carry brut |
|---|---|
| ETF actions | Dividendes des 12 derniers mois / prix brut − taux cash |
| IEF | `DGS7` − taux cash |
| TLT | `DGS20` − taux cash |
| LQD | `BAMLC0A0CMEY` − taux cash |

Toutes les séries FRED sont décalées d'un jour ouvré (délai de publication).

**Intégrité et stockage**

- Contrôles (signalés, jamais corrigés silencieusement) : jours manquants vs calendrier NYSE,
  prix figés, rendements extrêmes (> 8σ), prix ≤ 0, cohérence brut/ajusté vs dividendes et splits.
- Couche brute immuable (avec date de téléchargement) + couche nettoyée en parquet alignée NYSE.

---

## Bloc 2 — Volatilité

**Estimateur principal : EWMA span 35**

$$\sigma^2_t = \lambda\,\sigma^2_{t-1} + (1-\lambda)\,r_t^2, \quad \lambda \approx 0{,}944$$

$$\sigma^{\text{annuel}}_t = \sigma_t \times \sqrt{256} = 16\,\sigma_t$$

Préféré à l'écart-type simple sur fenêtre glissante (effet falaise / ghost effect).

**Mélange avec la volatilité de long terme (AFTS)**

$$\sigma^{\text{final}}_t = 0{,}7\;\sigma^{\text{EWMA}}_t + 0{,}3\;\bar{\sigma}^{\,10\text{ ans}}_t$$

Protège contre la sous-estimation de la volatilité après une longue période calme.
Utilise l'historique disponible (< 10 ans en début de backtest).

**Plancher de volatilité** : paramètre à fixer au bloc 6 (5e percentile sur 500 jours,
méthode pysystemtrade).

**Look-ahead** : σ_t calculé sur les rendements jusqu'à la clôture de t inclus.
Minimum 10 observations avant de produire une estimation.

---

## Bloc 3 — Règles de trading

**V1 : Trend (EWMAC ×3) + Carry (×2)**

**EWMAC**

$$\text{EWMAC}_{f,s,t} = \text{EMA}_f(P^{\text{adj}}_t) - \text{EMA}_s(P^{\text{adj}}_t)$$

$$\text{Signal normalisé}_t = \frac{\text{EWMAC}_{f,s,t}}{P^{\text{adj}}_{t-1} \times \sigma^{\text{daily}}_t}$$

| Variation | f | s | Scalaire Carver (référence) |
|---|---|---|---|
| Rapide | 8 | 32 | ~5,3 |
| Moyenne | 16 | 64 | ~3,75 |
| Lente | 32 | 128 | ~2,65 |

Variation 64/256 écartée (corrélation ~0,75 avec 32/128, gain marginal insuffisant).

**Carry**

$$\text{Carry normalisé}_t = \frac{\text{Carry brut}_t}{\sigma^{\text{annual}}_t}$$

$$\text{Carry Medium}_t = \text{EMA}_{5}(\text{Carry normalisé})_t \qquad \text{Carry Slow}_t = \text{EMA}_{20}(\text{Carry normalisé})_t$$

Une seule estimation de volatilité (EWMA span 35, bloc 2) pour tous les signaux.
Les deux vitesses viennent du lissage du signal normalisé, pas de fenêtres de vol différentes.

**Poids des règles par instrument**

| | EWMAC 8/32 | EWMAC 16/64 | EWMAC 32/128 | Carry Medium | Carry Slow |
|---|---|---|---|---|---|
| Trend + Carry | 0,233 | 0,233 | 0,233 | 0,150 | 0,150 |
| Trend only | 0,333 | 0,333 | 0,333 | — | — |

Lecture : 70 % trend / 30 % carry pour tous les instruments qui ont les deux règles.
Poids fixes (handcrafting Carver, pas d'optimisation sur les données).

Instruments trend-only : GLD, DBC, UUP, TIP.

---

## Bloc 4 — Scaling & capping

**Objectif** : ramener tous les forecasts à une amplitude moyenne de 10.

$$\text{Scaled Forecast}_t = \text{FS} \times \text{Signal normalisé}_t$$

$$\text{FS} = \frac{10}{\mathbb{E}_{\text{expansif, poolé}}\big[|\text{Signal normalisé}|\big]}$$

$$\text{Capped Forecast}_t = \text{clip}(\text{Scaled Forecast}_t,\; -20,\; +20)$$

**Méthode de calibrage du FS** : fenêtre expansive poolée sur tous les instruments.
Après normalisation correcte (÷ P×σ pour EWMAC, ÷ σ pour carry), les signaux vivent dans
le même espace statistique → un scalaire unique par règle est justifié et plus stable
(~56 000 obs en tout vs ~8 000 par classe d'actifs). Validation : vérifier que le scalaire
par classe ne s'écarte pas de plus de 20–30 % du scalaire global.

| Règle | Scalaire Carver (référence de validation) |
|---|---|
| EWMAC 8/32 | ~5,3 |
| EWMAC 16/64 | ~3,75 |
| EWMAC 32/128 | ~2,65 |
| Carry | ~30 |

Cap : ±20 (coupe ~5 % des valeurs extrêmes, protection, non négociable).

---

## Bloc 5 — Combinaison & FDM

**Objectif** : combiner les capped forecasts d'un instrument en un signal unique.

$$\text{Average Forecast}_t = \sum_i w_i \times \text{CappedForecast}_{i,t}$$

$$\text{Combined Forecast}_t = \text{clip}(\text{FDM} \times \text{Average Forecast}_t,\; -20,\; +20)$$

$$\text{FDM} = \frac{1}{\sqrt{w^\top \Sigma\, w}}$$

où $\Sigma$ est la matrice de corrélation entre capped forecasts, estimée sur données passées.

**Deux FDM distincts** (imposé par la structure des règles, pas de l'overfitting) :

| Type | Règles | FDM estimé |
|---|---|---|
| Trend + Carry (11 instruments) | EWMAC ×3 + Carry ×2 | ~1,19 |
| Trend only (4 instruments) | EWMAC ×3 | ~1,08–1,10 |

Au sein de chaque groupe : corrélations poolées entre instruments (plus stable).

**Précautions anti-instabilité**
- Corrélations négatives ramenées à 0 avant calcul (évite FDM artificiellement élevé).
- Plafond FDM à 2,5.
- Lissage EWMA long sur la matrice de corrélation (évite les sauts brusques de position).

---

## Bloc 6 — Position sizing

**Formule centrale (Carver, *Systematic Trading*)**

$$N_t = \frac{\text{Capital}_t \times \tau \times F_t}{10 \times \sigma^{\text{annual}}_t \times P^{\text{brut}}_t}$$

- $F_t$ : Combined Forecast $\in [-20, +20]$, $\mathbb{E}[|F|] = 10$
- $\tau = 13\%$ : volatility target
- $\sigma^{\text{annual}}_t$ : volatilité annualisée (bloc 2), = $16 \times \sigma^{\text{daily}}_t$
- $P^{\text{brut}}_t$ : prix brut (pas ajusté)
- $\text{Capital}_t$ : valeur du compte **courant** (pas initial)
- Le "10" = $\mathbb{E}[|F|]$ : à $F=10$, la position contribue exactement $\tau$ de vol annuelle

Réécriture explicite du Volatility Scalar :

$$N_t = \underbrace{\frac{\text{Capital}_t \times \tau}{\sigma^{\text{annual}}_t \times P^{\text{brut}}_t}}_{\text{Volatility Scalar (actions)}} \times \frac{F_t}{10}$$

**Sizing sur capital courant — protection contre le compounding des pertes**

Après un drawdown de 20 % (capital = 80 k$), les positions se réduisent automatiquement
de 20 %. La vol en % du capital reste constante à τ quelle que soit la taille du compte.
Avec capital initial fixe, le levier effectif monterait à 1,25x après -20 %, ce qui
amplifie les pertes suivantes et mène à la ruine.

**Plancher de volatilité**

$\sigma^{\text{annual}}_t \leftarrow \max(\sigma^{\text{annual}}_t,\; \sigma^{\text{floor}})$
où $\sigma^{\text{floor}}$ = 5e percentile sur 500 jours. Évite les positions explosives
en période de volatilité anormalement basse.

**SHY retiré de l'univers**

Vol ~1,5 % / an → Volatility Scalar ≈ 163 000 actions → notionnel ~138x le capital.
Incompatible avec la contrainte de levier 2x. Univers obligations : IEF, TLT, TIP, LQD.

**Contrainte de levier**

$$\text{Levier brut}_t = \frac{\sum_i |N_{i,t} \times P^{\text{brut}}_{i,t}|}{\text{Capital}_t} \leq 2$$

Si dépassé : rescaling uniforme de toutes les positions. Appliqué après le buffer (bloc 8).

---

## Bloc 7 — Portefeuille & IDM

**Pourquoi pas Markowitz**

1. Estimation error : poids très sensibles aux espérances de rendement (paramètre le plus bruité).
2. Instabilité : poids optimaux changent radicalement selon la période → turnover élevé.
3. Overfitting : backtest Markowitz trop beau car poids calibrés sur les mêmes données.

**Handcrafting hiérarchique — poids provisoires**

Niveau 1 (entre classes) puis niveau 2 (au sein de chaque classe) — égal au sein de chaque niveau.

| Classe | Poids classe | Instruments | Poids instrument |
|---|---|---|---|
| Actions US | 20 % | SPY, IWM | 10 % chacun |
| Actions internationales | 20 % | VGK, EWJ, EEM, EPP, EWC | 4 % chacun |
| Obligations | 25 % | IEF, TLT, TIP, LQD | 6,25 % chacun |
| Matières premières | 25 % | GLD, DBC | 12,5 % chacun |
| Devises | 10 % | UUP | 10 % |

Justification des poids classes : obligations à 25 % (décorrélation avec actions la plus forte
et stable) ; matières premières à 25 % (protection inflation) ; devises à 10 % (un seul instrument,
poids réduit pour éviter de surpondérer UUP ; V2 ajoutera des instruments).
À ajuster au bloc 7 si corrélations intra-classe trop fortes (EEM/EPP notamment).

**IDM**

$$\text{IDM} = \frac{1}{\sqrt{w^\top \Sigma^{\text{instruments}}\, w}}, \quad \text{plafonné à 2,5}$$

où $\Sigma^{\text{instruments}}$ = matrice de corrélation des rendements des sous-systèmes,
estimée sur fenêtre expansive (pas de bootstrapping global : look-ahead).
La formule analytique est équivalente au bootstrap sous hypothèse de normalité, sans bruit
Monte Carlo. Mêmes précautions que FDM : $\rho < 0 \to 0$, lissage EWMA.

**Position finale**

$$N^{\text{final}}_{i,t} = N_{i,t} \times w_i \times \text{IDM}$$

---

## Bloc 8 — Buffer & coûts

**Buffer (filtre de décision)**

$$\text{Trader seulement si } \left|\frac{N^{\text{actuel}}_{i,t} - N^*_{i,t}}{N^*_{i,t}}\right| > \delta = 0{,}10$$

Si hors zone : ramener au **bord de la zone** (pas à $N^*$), ce qui réduit le turnover.
Zone tampon : $\big[N^*_{i,t} \times 0{,}90,\; N^*_{i,t} \times 1{,}10\big]$.

**Ordre des opérations (important)**

Signal → position cible $N^*$ → buffer → position à trader → vérification levier → ordre.

Le buffer doit précéder le contrôle de levier : le rescaling de levier (mécanisme de risque
global) ne doit pas polluer la décision de signal. Inverser l'ordre ferait déclencher le buffer
sur des mouvements de rescaling sans changement de signal réel.

**Modèle de coûts**

Coût d'un trade = spread bid-ask ÷ prix. Coût en Sharpe = coût × turnover annuel ÷ σ.

Speed limit de Carver : les coûts de transaction ≤ ⅓ du Sharpe brut attendu de la règle.
Utilisée pour valider que chaque règle est rentable nette de coûts sur chaque instrument.

Commissions : 0 $ chez Alpaca (ETF US). Paramétrable pour robustesse.

Dividendes payés sur les shorts : implicites dans la formule P&L (r_total négatif si short + dividende).

**Taux de borrow (shorts, valeurs conservatrices)**

| Instrument | Taux annuel |
|---|---|
| SPY, IWM | 0,3 % |
| VGK, EWJ | 0,5 % |
| EEM, EPP, EWC | 1,5 % |
| IEF, TLT, TIP, LQD | 0,3 % |
| GLD | 0,5 % |
| DBC | 0,8 % |
| UUP | 0,5 % |

À revérifier à l'implémentation (IB Securities Lending ou équivalent).

---

## Bloc 9 — Backtest & évaluation

*À détailler.*

---

## Bloc 10 — Exécution live & risque

*À détailler.*

---

## Backlog V2

- Instruments de volatilité (VIX, V2X, VXN) via futures VX/VXM/FVS → nécessite IBKR et un module de roll.
- Immobilier (VNQ) et autres classes.
- Carry des matières premières (nécessite les courbes de futures).
- Carry des devises (UUP) : différentiel de taux USD vs panier (taux étrangers mensuels, publiés avec retard).
- Migration éventuelle vers IBKR (futures) via une nouvelle implémentation de `BrokerInterface`.
- Règle Breakout (BRKOUT) : skew positif, corrélation ~0,5 avec EWMAC, candidat prioritaire.
- Règle Accélération (ACCEL) : corrélation ~0,1 avec EWMAC, instable, candidat secondaire.
- Momentum cross-sectionnel : alpha différent, mais implique une architecture de signal différente.
