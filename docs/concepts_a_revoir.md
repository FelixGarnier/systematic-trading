# Concepts à revoir

Liste des notions à approfondir. Cocher quand le concept est maîtrisé.

## Bloc 0 — Objectifs & contraintes

- [ ] Critère de Kelly et half-Kelly : pourquoi la vol optimale égale le Sharpe ratio,
      et pourquoi on prend la moitié.
- [ ] Relation drawdown / volatilité : pourquoi, à Sharpe donné, le drawdown attendu
      est à peu près proportionnel à la vol (Magdon-Ismail & Atiya, 2004).

## Bloc 2 — Volatilité

- [ ] EWMA : formule de mise à jour récursive, λ, et pourquoi c'est mieux que l'écart-type simple.
- [ ] Effet falaise (ghost effect) : pourquoi l'écart-type simple crée des sauts artificiels.
- [ ] Mélange 70/30 avec la vol de long terme : intuition et formule.

## Bloc 8 — Buffer & coûts

- [ ] Speed limit de Carver : comment exprimer un coût de transaction en unités de Sharpe.
- [ ] Turnover : comment le mesurer (nombre de "rotations" complètes par an).
- [ ] Pourquoi le buffer précède le contrôle de levier (séparation signal vs risque global).
- [ ] Taux de borrow : mécanisme économique du short (emprunt de titres, dividendes payés).

## Bloc 7 — Portefeuille & IDM

- [ ] Markowitz : pourquoi les poids sont instables (sensibilité aux espérances de rendement, estimation error).
- [ ] Handcrafting hiérarchique : intuition "budget de risque de haut en bas".
- [ ] IDM formule analytique = bootstrap sous normalité : pourquoi c'est équivalent et préférable.
- [ ] IDM vs FDM : même mécanique, niveaux différents (portefeuille vs instrument).

## Bloc 6 — Position sizing

- [ ] Formule N = Capital × τ × F / (10 × σ_annual × P) : dériver chaque terme depuis le critère "à F=10, vol = τ".
- [ ] Volatility Scalar : comment l'interpréter comme le nombre d'actions à "pleine cible".
- [ ] Sizing sur capital courant vs capital initial : pourquoi le second mène à la ruine (compounding des pertes).
- [ ] Plancher de volatilité : lien avec l'effet krach suivi de calme (ex. mars 2020 → été 2020).

## Bloc 5 — Combinaison & FDM

- [ ] Pourquoi la combinaison pondérée affaiblit le signal (variance de la somme < somme des variances).
- [ ] FDM = 1/√(w'Σw) : intuition géométrique (diversification entre règles).
- [ ] Pourquoi deux FDM et pas un seul : la composition des règles est structurellement différente.
- [ ] Corrélations négatives → 0 : pourquoi c'est conservateur et pas naïf.

## Bloc 4 — Scaling & capping

- [ ] Fenêtre expansive poolée : pourquoi "expansive" évite le look-ahead et pourquoi "poolée" stabilise l'estimation.
- [ ] Pourquoi la normalisation (÷ P×σ pour EWMAC, ÷ σ pour carry) rend les scalaires cohérents entre instruments et classes d'actifs.
- [ ] Capping à ±20 : d'où vient ce chiffre (distribution des signaux, percentile coupé).
- [ ] Scalaire unique vs par classe : pourquoi la normalisation fait déjà le travail, et comment valider (écart < 20-30 % entre groupes).
- [ ] Valeurs de référence Carver (EWMAC 8/32 ≈ 5.3, 16/64 ≈ 3.75, 32/128 ≈ 2.65, Carry ≈ 30) : comment les utiliser comme contrôle de cohérence.

## Bloc 3 — Règles de trading

- [ ] Pourquoi le skew positif du trend est une "assurance naturelle" en crise.
- [ ] Pourquoi le skew négatif du carry est l'opposé, et comment les deux se complètent.
- [ ] Différence entre trend time-series (EWMAC) et momentum cross-sectionnel.
- [ ] Normalisation EWMAC : pourquoi diviser par (P × σ_daily) rend le signal sans unité.
- [ ] Carry normalisé = ratio de Sharpe instantané : interprétation économique.
- [ ] Pourquoi les deux vitesses de carry viennent du lissage du signal, pas de fenêtres de vol différentes.

## Bloc 1b — Données

- [ ] Prix ajusté (total return) vs prix brut : à quoi sert chacun.
- [ ] Pourquoi l'ajustement multiplicatif ne crée pas de look-ahead sur les ratios
      (rendements, EWMAC), mais rend faux un niveau de prix ajusté pris isolément.
- [ ] Formule du P&L : `PnL_t = N_{t-1} × P_brut_{t-1} × r_total_t`, et pourquoi
      elle gère automatiquement les dividendes, y compris ceux payés par les shorts.
- [ ] Données "point-in-time" et décalage de publication (lag FRED d'un jour).
