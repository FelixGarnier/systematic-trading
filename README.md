# Systematic Trading

Implémentation pré-enregistrée du cadre de Robert Carver sur quatorze ETF américains, puis boucle de paper trading Alpaca. Les paramètres n'ont pas été modifiés après le backtest.

**Pre-registered Carver framework on US ETFs.** Sharpe 0.41 from 2009 to October 2026, against 0.88 for SPY, 0.99 for a 60/40, and 1.00 for a long-only book in the same engine. The paper is [`docs/research_paper.pdf`](docs/research_paper.pdf).

| Livre | Sharpe | CAGR | Volatilité | Drawdown max |
|---|---:|---:|---:|---:|
| Stratégie | 0,41 | 4,1 % | 11,6 % | −29,9 % |
| SPY | 0,88 | 15,2 % | 17,9 % | −33,7 % |
| 60/40 | 0,99 | 10,2 % | 10,4 % | −21,0 % |
| Long-only, forecast +10 | 1,00 | 10,1 % | 10,2 % | −19,8 % |

Fenêtre : 2 janvier 2009 – 13 octobre 2026. Le livre à forecast constant +10 isole les signaux : même univers, mêmes coûts, même moteur.

## Paper trading

Compte Alpaca paper uniquement. Un agent macOS appelle `python -m systrade.execution tick` toutes les cinq minutes. Le tick est calé sur New York : clôture 20 minutes après la fin de séance, envoi à l'ouverture, re-peg puis market dans la demi-heure qui suit. S'il a manqué le soir, il rattrape au passage suivant.

```bash
pip install -e ".[live]"
python -m systrade.execution close --dry-run   # aucun appel Alpaca
```

Les clés vont dans `.env` (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`). Le journal est `data/live/trading.db`, le log `data/live/scheduler.log`.

## Tests

```bash
pytest
```

250 tests. Le design est dans [`docs/architecture.md`](docs/architecture.md).
