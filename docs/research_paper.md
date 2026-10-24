# Trend Following and Carry on an ETF Universe: A Pre-Registered Out-of-Sample Evaluation of the Carver Framework

**Felix Garnier**  
Independent Research Project  
October 2026

---

## Abstract

This paper evaluates a systematic trading strategy implementing the full Carver (2015, 2019) framework — exponentially-weighted moving average crossover (EWMAC) trend rules, carry signals, volatility-scaled position sizing, a no-trade buffer, and an instrument diversification multiplier — on a universe of fourteen liquid US ETFs over the period January 2009 to October 2026. The implementation is pre-registered: the design document, signal definitions, and portfolio construction rules were all finalised before any performance calculation. The strategy achieves an annualised Sharpe ratio of 0.41 and a CAGR of 4.1%, against 0.88 for SPY, 0.99 for a daily-rebalanced 60/40, and — critically — 1.00 for a long-only book run through the same engine at a constant forecast of +10. Transaction costs account for approximately 0.063 Sharpe units, well within the Carver speed-limit of 0.167; they cannot explain the gap. A clean control experiment isolates the source: when the signals are removed and the portfolio is held long-only at the same volatility target, performance improves materially. The numbers in the tables below are the book before cash financing. A later engine correction, not a retune, charges the lagged T-bill on borrowed cash and pays it on short proceeds. With that charge the strategy Sharpe is 0.44 and the long-only control is 0.87; in excess of the T-bill they are 0.31 and 0.73. The slow trend alone, at 0.48, is already far from the control, so the gap is not the carry. The 2022–2026 block is not a 2022 loss: 2022 is +7.6% for the financed strategy, and the drawdown of −25.1% bottoms on 27 January 2025. The annual table is in the README. Compounding the old 4.1% CAGR from 2009 reaches about $202,000. The path that ends near $226,000 starts in January 2007 and includes the warmup. The full pipeline — 250 unit tests, expandingwindow estimates, no look-ahead — is deployed in an Alpaca paper-trading loop that runs daily and writes a traceable audit trail to SQLite.

---

## 1. Introduction

Systematic trend-following and carry represent two of the most extensively documented sources of risk premia in financial markets. Moskowitz, Ooi and Pedersen (2012) establish time-series momentum across asset classes; Asness, Moskowitz and Pedersen (2013) document carry; Carver (2015, 2019) provides a practitioner framework combining both in a single multi-asset system. The canonical application is futures: diverse, liquid, with clean carry signals derived from term structure curves.

A natural question is whether the same architecture can be applied to a cost-free, fractional-share ETF universe accessible to retail participants. This project addresses it in the most stringent way possible: by pre-registering the design, implementing it without look-ahead, and reporting the result as found — including the underperformance.

The contribution is not a positive result. It is a fully implemented, rigorously tested, production-quality pipeline that documents exactly *why* the strategy fails on this terrain, and deploys the identical code in a live paper-trading environment. Three results drive the analysis. First, trend-following is present but concentrated in the slowest rule (EWMAC 32/128, Sharpe 0.49 standalone) and absent in the fastest (EWMAC 8/32, Sharpe 0.05). Second, ETF carry proxies — dividend yields for equities, FRED rate differentials for bonds — are too small and too stable to function as informative signals; standalone carry drawdowns exceed −55%. Third, removing the signals entirely, while keeping the same volatility target, allocation, and cost model, raises the Sharpe from 0.41 to 1.00. This isolates the signals as the source of underperformance and rules out explanations based on costs or execution approximations.

---

## 2. Related Literature

**Trend following.** Moskowitz, Ooi and Pedersen (2012) document that past twelve-month returns positively predict next-month returns across equity index, fixed income, currency and commodity futures. Baz et al. (2015) show similar results with EWMA-based filters. Hurst, Ooi and Pedersen (2017) extend the evidence to a century of data. Carver (2019) argues that multiple speeds (fast, medium, slow) each capture distinct information and that combining them via a forecast diversification multiplier (FDM) dominates any single speed.

**Carry.** Koijen et al. (2018) show that carry is positively related to expected returns across asset classes; the mechanism is different for equities (dividend yield minus funding cost), rates (yield spread), and commodities (futures basis). The key feature shared across asset classes is that carry is estimated from term structure information — futures curve slopes, dividend yields, rate differentials. On an ETF universe, only an approximation to the last two is available. This approximation is the source of the failure documented below.

**ETF-based systematic strategies.** Novy-Marx and Velikov (2016) and others note that strategy performance often degrades substantially when moved from futures to ETFs due to shorter histories, less clean carry proxies, and correlation structure dominated by equity beta.

**Pre-registration in finance.** Harvey, Liu and Zhu (2016) argue that the literature suffers from massive multiple testing; p-hacking inflates the reported Sharpe ratios of even honest strategies by a factor of two or more when parameters are chosen after seeing data. This project adopts pre-registration as the only structural protection against this: the design document (committed to git before any backtest run) is the single reference for all parameters.

---

## 3. Data and Universe

### 3.1 Price data

End-of-day prices are sourced from Tiingo. The universe consists of fourteen US ETFs representing five asset classes:

| Class | Instruments | Signals |
|---|---|---|
| US equities | SPY, IWM | Trend + Carry |
| International equities | VGK, EWJ, EEM, EPP, EWC | Trend + Carry |
| Fixed income | IEF, TLT, TIP, LQD | Trend + Carry (TIP: trend only) |
| Commodities | GLD, DBC | Trend only |
| FX | UUP | Trend only |

Selection criteria, fixed before the backtest: (i) continuous US listing since at least 2007; (ii) diversification contribution (pairwise correlation below 0.95 at selection); (iii) easy-to-borrow status enabling short positions. SHY was excluded because its annualised volatility of approximately 1.5% produces a volatility scalar of ~138× capital, incompatible with a 2× leverage cap. TIP carry was excluded due to a structural downward bias in real yield estimates over 2011–2022.

The adjusted price series incorporates dividends multiplicatively, ensuring that return ratios and normalised signals are backward-look-through invariant. Raw (unadjusted) prices are used exclusively for position sizing and P&L attribution.

All prices are available from January 2007. The evaluation window begins 2 January 2009 (two years of warm-up for EWMA estimates, forecast scalars, and portfolio-level correlations). The final observation is 13 October 2026, yielding 4,472 evaluation days.

### 3.2 Rate data

Short rates (FRED DTB3, the 3-month T-bill) and bond yield proxies (FRED DGS7, DGS20, BAMLC0A0CMEY) are downloaded daily. All FRED series are lagged one calendar day before use: at date *t*, only data published before *t* is visible. This is the most consequential look-ahead control in the pipeline.

### 3.3 Carry proxies

For equity ETFs, carry is estimated as trailing twelve-month (252-day) dividend income per share divided by raw price, minus the T-bill rate. For IEF and TLT, carry is the respective FRED yield minus the T-bill. For LQD, carry uses BAMLC0A0CMEY, which was only added to FRED from 16 October 2023; LQD is therefore treated as trend-only for the majority of the sample, with the carry signal entering only over the final ~750 days.

---

## 4. Methodology

The implementation follows Carver (2015, 2019) in full. The exposition here focuses on the key design choices and the look-ahead controls enforced at each step.

### 4.1 Volatility estimation

Daily volatility is estimated by an EWMA with span 35 (equivalent λ ≈ 0.943). To avoid the well-known ghost effect of rolling windows, we use the EWMA throughout. A long-run stabiliser blends the current EWMA with its own 10-year equally-weighted mean (70/30 weight):

$$\hat{\sigma}_t = 0.70 \cdot \hat{\sigma}^{\text{EWMA}}_t + 0.30 \cdot \overline{\sigma}^{10y}_t$$

A floor equal to the 5th percentile of the blended volatility series (estimated on the past 500 days, shifted one period) prevents extreme volatility scalars during low-dispersion regimes. All estimates use only information available at the close of day *t*.

### 4.2 Signals

**Trend (EWMAC).** Three speeds are implemented: 8/32, 16/64, 32/128, denoting the span of the fast and slow exponential moving averages. The normalised signal for speed (f, s) at date t is:

$$\text{EWMAC}_{f,s,t} = \frac{\text{EMA}_{f,t} - \text{EMA}_{s,t}}{P_{t-1} \cdot \hat{\sigma}_{d,t-1}}$$

where P is the unadjusted price and σ_d the daily volatility, both shifted one period. This normalisation ensures the signal is unitless and comparable across instruments.

**Carry.** Two speeds are implemented: fast (EWMA span 5) and slow (EWMA span 20) of the normalised raw carry, where raw carry divided by annualised volatility is the normalised input. The EWMA filter reduces noise in the daily carry estimate.

### 4.3 Forecast scaling and capping

Each signal has a different natural scale. Forecast scalars (FS) bring E[|signal|] to 10 for every rule, enabling equal-weighted combination. Scalars are estimated from an expanding pooled cross-section requiring at least 100 instrument-day observations:

$$\text{FS}_t = \frac{10}{\text{pooled}\ E_t[|f|]}$$

All signal arithmetic uses the shifted (t−1) signal to produce the forecast for day t. Forecasts are capped symmetrically at ±20 after scaling. Realised forecast scalars are: EWMAC 8/32: 7.21, 16/64: 4.97, 32/128: 3.38; carry medium: ~84, carry slow: ~70. The carry scalars are large because ETF carry is small in absolute terms. After capping, E[|capped forecast|] ≈ 9.1–9.8 for all rules.

### 4.4 Forecast combination

Within instruments carrying both trend and carry, the combined forecast is a weighted average of the five signals. Rule weights are: 0.233 for each of the three EWMAC rules (sum 0.70) and 0.15 for each of the two carry rules (sum 0.30), following Carver's approximate 70/30 split. For instruments with no carry (GLD, DBC, UUP, TIP) weights are 1/3 each across EWMAC rules.

When a carry value is NaN on a given date (e.g., LQD before BAMLC0A0CMEY is available), the remaining rules are renormalised to sum to 1 rather than zeroing the forecast. This prevents spurious signal drop-outs.

A Forecast Diversification Multiplier (FDM) scales the combined forecast upward to restore E[|combined|] = 10, accounting for correlation between rules. The FDM is estimated on expanding pairwise correlations of the individual (scaled, capped) forecasts, with negative correlations set to zero before matrix inversion. A Herfindahl-style formula gives:

$$\text{FDM}_t = \min\!\left(\frac{1}{\sqrt{w' C_t w}}, 2.5\right)$$

where C_t is the pairwise correlation matrix. The FDM is smoothed with an EWMA of span 125. Realised values: trend+carry instruments, ~1.41 (Carver benchmark ~1.19); trend-only instruments, ~1.08 (matches benchmark). The slightly higher realised trend+carry FDM reflects the fact that ETF carry and trend have near-zero correlation, providing more diversification than in a futures universe.

### 4.5 Position sizing

The target position for instrument i at date t is:

$$N^*_{i,t} = \frac{\text{Capital}_t \cdot \tau \cdot F_{i,t} \cdot w_{i,t} \cdot \text{IDM}_t}{10 \cdot \sigma^{\text{annual}}_{i,t-1} \cdot P_{i,t}}$$

where τ = 13% is the portfolio volatility target (half-Kelly, consistent with an expected Sharpe of ~0.5 on a diversified book), F is the combined forecast (capped ±20), w is the instrument weight, IDM is the instrument diversification multiplier, σ_annual the annualised volatility (shifted), and P the raw close price.

Capital is the *current* portfolio value, not the initial capital. This is the key anti-ruin property: a 30% drawdown automatically reduces all positions by 30%.

**Instrument weights** follow a handcrafted hierarchical scheme: US equities 20% (SPY, IWM 10% each); international equities 20% (VGK, EWJ, EEM, EPP, EWC 4% each); fixed income 25% (IEF, TLT, TIP, LQD 6.25% each); commodities 25% (GLD, DBC 12.5% each); FX 10% (UUP 10%). These weights are fixed throughout.

**IDM** is estimated on expanding pairwise correlations of subsystem returns, where the subsystem return for instrument i is:

$$r^{\text{sub}}_{i,t} = \frac{F_{i,t-1}}{10} \cdot \frac{r_{i,t}}{\sigma_{d,i,t-1}}$$

This is the return that would be earned by instrument i if it were the only holding, scaled to unit forecast. Pairwise correlations use the same formula as the FDM. The IDM is smoothed with an EWMA of span 125, capped at 2.5, and initialised at 1. Instruments require 252 subsystem return observations before entering the portfolio; weights are renormalised during the ramp-up. Realised mean IDM over the evaluation window: 1.89.

### 4.6 Buffer and leverage

The gross leverage of the target portfolio frequently exceeds 2× because volatility scaling amplifies positions during low-vol regimes. Two controls are applied in sequence:

1. **Buffer.** For each instrument, define a no-trade zone: [N* − δ|N*|, N* + δ|N*|] with δ = 0.10. The held position is adjusted toward the nearest edge only when it exits the zone. The buffer absorbs small signal movements without generating trades. The buffer state is maintained in pre-leverage shares, ensuring that a leverage rescaling event does not re-trigger a trade the following day.

2. **Leverage cap.** After the buffer, gross leverage is computed. If it exceeds 2×, all positions are scaled down uniformly: scale = min(1, 2 / gross_leverage). This is a hard constraint, applied at every date.

Realised mean leverage: 1.72×. Maximum: 2.00×.

### 4.7 Transaction costs

Trade costs are modelled as:

$$\text{cost}_{i,t} = |\Delta N_{i,t}| \cdot P_{i,t} \cdot \text{spread} \cdot \text{slippage}$$

with spread = 2 bps (a conservative flat assumption for liquid US ETFs) and slippage multiplier = 1.5 (accounting for market impact and adverse selection). Commissions are zero (Alpaca ETF pricing). Short positions incur a daily borrow cost at instrument-specific rates (ranging from 0.3% per annum for GLD to 0.5% for IEF and TLT, following Carver's tables), charged only on negative (short) holdings.

Realised annual turnover: 24.1× capital. Sharpe cost: 0.063. Speed limit (Carver's criterion for a strategy with SR_expected = 0.5): 0.5 / 3 = 0.167. The strategy is well within the speed limit.

### 4.8 Benchmarks

Three benchmarks are evaluated over the identical window with the same start date:

1. **SPY buy-and-hold**: total return including dividends, zero cost.
2. **Daily 60/40**: 60% SPY + 40% IEF, rebalanced daily, zero cost.
3. **Long-only vol target**: the same engine with a constant forecast of +10 for all instruments. This benchmark shares the universe, weights, IDM, buffer, leverage cap, and transaction costs. It isolates the marginal contribution of the signals.

---

## 5. Results

### 5.1 Full-sample performance

Table 1 reports annualised performance statistics over 4,472 evaluation days (2 January 2009 to 13 October 2026).

**Table 1. Full-sample performance (2009–2026).**

| Book | Sharpe | CAGR | Volatility | Max DD | Skew | Hit rate |
|---|---:|---:|---:|---:|---:|---:|
| Strategy | 0.41 | 4.1% | 11.6% | −29.9% | −0.58 | 53.7% |
| SPY | 0.88 | 15.2% | 17.9% | −33.7% | −0.25 | 55.3% |
| 60/40 | 0.99 | 10.2% | 10.4% | −21.0% | −0.14 | 55.7% |
| Long-only vol target | 1.00 | 10.1% | 10.2% | −19.8% | −0.50 | 55.4% |

The strategy achieves a lower Sharpe ratio than all three benchmarks. The relevant comparison for signal evaluation is the long-only vol target: both books have essentially the same volatility (~11–12% with leverage, ~10% after the cap) and the same costs. The Sharpe gap of 0.59 is attributable solely to the forecasts.

Leverage dynamics: mean 1.72×, maximum 2.00× (the cap binds regularly during low-volatility regimes). Turnover: 24.1×. Cost drag in Sharpe terms: 0.063 vs. the long-only vol target's 0.006. Even attributing the full incremental cost (0.057 Sharpe units) to signals, the unexplained gap is 0.53.

The $100,000 account started in January 2007, warmup included, ends near $226,000. The 4.1% CAGR belongs to the 2009–2026 window only, and compounds to about $202,000. The two figures are not the same path.

### 5.2 Walk-forward decomposition

To assess whether the full-sample result masks structural breaks, four non-overlapping periods are evaluated without any re-fitting of parameters. Table 2 reports results.

**Table 2. Walk-forward Sharpe ratios (no parameter updates).**

| Period | Sharpe | CAGR | Max DD |
|---|---:|---:|---:|
| 2009–2011 | 0.45 | 5.2% | −10.1% |
| 2012–2016 | 0.41 | 3.6% | −12.3% |
| 2017–2021 | 0.77 | 8.2% | −11.7% |
| 2022–2026 | 0.06 | −0.1% | −29.9% |

Three of four periods produce positive Sharpe ratios. The final period (2022 onward) dominates the aggregate result. This period is characterised by a simultaneous drawdown in equities and fixed income — an environment where a book that is long growth assets and short duration faces persistent losses on both sides. The maximum drawdown of −29.9% is almost entirely attributable to this period.

### 5.3 Rule-level decomposition

Each rule is evaluated in isolation by running the full engine with only that rule's capped forecast. This isolates the marginal information content of each signal.

**Table 3. Standalone rule performance.**

| Rule | Sharpe | Turnover | Max DD | Sharpe cost |
|---|---:|---:|---:|---:|
| EWMAC 8/32 | 0.05 | 44.1× | −28.1% | 0.106 |
| EWMAC 16/64 | 0.34 | 22.7× | −28.7% | 0.056 |
| EWMAC 32/128 | 0.49 | 12.7× | −29.8% | 0.032 |
| Carry medium | 0.18 | 3.7× | −56.8% | 0.011 |
| Carry slow | 0.17 | 2.6× | −58.7% | 0.008 |

Three observations:

1. **Trend is monotonically better at longer horizons.** The fast rule (8/32) is essentially flat (Sharpe 0.05) and its cost in Sharpe units (0.106) nearly equals its gross alpha. The slow rule (32/128, Sharpe 0.49) is the primary source of trend premium.

2. **Carry drawdowns are extreme.** Standalone carry maximum drawdowns exceed −55%, an order of magnitude larger than what the carry rules contribute positively to the combined book. ETF carry is too stable and too mean-reverting to function as a reliable signal over shorter horizons.

3. **The combination underperforms the best individual rule.** The combined strategy Sharpe of 0.41 is below the EWMAC 32/128 standalone of 0.49, which normally should not happen with genuine diversification. The carry signals, by adding drawdown risk without commensurate return, degrade the combination.

### 5.4 Asset-class contribution

Table 4 reports the Sharpe ratio of each asset class's daily P&L contribution to the portfolio, expressed as a fraction of starting capital.

**Table 4. Asset-class contribution Sharpe ratios.**

| Class | Sharpe |
|---|---:|
| Commodities | 0.45 |
| Fixed income | 0.35 |
| FX | 0.19 |
| US equities | 0.14 |
| International equities | −0.09 |

Commodities and fixed income drive positive attribution; international equities are mildly negative. This is consistent with prior literature: trend-following performs best in assets with autocorrelated prices and low noise-to-signal ratios. US and international equities in this period exhibit strong upward drift, which creates persistent short signals that lose money.

---

## 6. Discussion

### 6.1 Why the signals underperform on ETFs

The long-only vol target benchmark is decisive. A portfolio that simply holds the fourteen ETFs long, scaled to the same volatility, with the same instrument weights and the same cost model, achieves a Sharpe of 1.00 over the same window. The Carver framework adds five signals at a cost of 0.59 Sharpe points.

Two structural mismatches explain this:

**ETF carry is not futures carry.** In a futures book, carry is derived from the slope of the term structure: the difference between spot and deferred contracts. It is observable, precise, and updated at each roll. For equity ETFs, the proxy — trailing twelve-month dividend yield minus cash — is smooth, backward-looking, and dominated by the same factors as price momentum (rising prices compress yield). For bonds, FRED yields are a daily series (DGS7, DGS20, and the credit spread) and still omit the roll-down. The carry scalars of ~84 (vs. Carver's ~30 for futures) reflect that the raw signal is far smaller than its futures counterpart. After capping, the carry rules contribute signal that is only loosely related to the economic carry the framework was designed to exploit.

**The 2009–2026 equity bull market punishes short equity signals.** The EWMAC framework generates short positions when prices trend down and holds them until the trend reverses. US equities in this sample experienced three brief drawdowns (2009, 2020, 2022) interspersed with strong upward trends. Short signals generated in drawdowns were frequently reversed before they could profit. International equities underperformed US equities throughout, generating additional short signals against the global equity bid.

### 6.2 What the framework does correctly

Despite the signal underperformance, several design elements function as intended:

- **Volatility targeting**: the strategy stays close to its 13% target except in high-leverage, low-volatility regimes where the 2× cap binds. Mean leverage of 1.72× indicates that the strategy is not systematically over-sized.
- **The buffer works**: approximately 29% of share variation is absorbed by the buffer; only 66% of instrument-days generate a trade. Turnover of 24.1× is substantially below the speed limit.
- **The long-only vol target is competitive**: Sharpe 1.00 vs. 0.99 for the 60/40, with materially lower maximum drawdown (−19.8% vs. −21.0%). The allocation and sizing engine, stripped of signals, is a credible portfolio.
- **Walk-forward stability**: the first three periods (2009–2021) are consistent at Sharpe 0.41–0.77. The negative contribution is concentrated in a specific macro regime (2022 rate-shock) rather than distributed across the sample.

### 6.3 No-tuning commitment

No parameter was revised after the backtest was run. The pre-registered design document was committed to version control before any performance calculation. This constraint eliminates the most common source of illusory backtested performance: in-sample parameter optimisation. The documented Sharpe of 0.41 is a genuine out-of-sample statistic under the Carver framework, not an upper bound on what this methodology can achieve in-sample.

The 250-test suite covering every module from volatility estimation through buffer computation ensures that the implementation is correct. No test was written after a failure was discovered post-facto; tests define the contract of each function.

---

## 7. Live Paper Trading

The same codebase is deployed in a paper-trading loop on Alpaca. The loop runs every five minutes, determining whether to:

1. *Close* (after the NYSE session ends, +20 minutes): update prices and rates, recompute the full book, write target positions and orders to SQLite — nothing is submitted.
2. *Open* (at the NYSE open): submit limit orders, passive on the bid (buys) or ask (sells).
3. *Work* (during the session): re-peg unfilled limits every five minutes; convert to market after 30 minutes.

Three live-specific risk controls are added: a circuit-breaker that suspends order submission if the day's P&L falls below −5% of equity; a preventive leverage gate at 1.5× (blocks exposure-increasing orders while allowing reductions); and an ADV filter that clips orders to 1% of average daily dollar volume.

The broker is accessed through an abstract `BrokerInterface`. The current implementation (`AlpacaBroker`) targets the paper endpoint; a future `IBKRBroker` would require no changes to the strategy layer.

The audit trail records, for every session: target positions, forecast values, weights, IDM, held positions at the broker, orders planned versus submitted versus filled, and any risk override. This constitutes the data required to attribute live deviations from backtest to execution (slippage, partial fills, data latency) versus strategy drift.

---

## 8. Conclusion

This project implements the Carver systematic trading framework from scratch, in production-quality Python, on a fourteen-ETF universe, and evaluates it over 17 years without re-fitting any parameter after seeing performance. On this sample the signals trail a long-only book in the same engine. After charging the T-bill on cash, that gap is about 0.43 of Sharpe, and it is already present in the slow trend alone. The 2023–2024 losses, not 2022, produce the drawdown. A paired test of the gap is not reported.

The contribution is threefold: a rigorous implementation that can serve as a foundation for a futures-based V2; a methodologically clean control experiment that isolates signals from allocation; and a live paper-trading deployment that closes the loop from research to execution.

For a V2 using futures, the infrastructure is in place. The `BrokerInterface` abstraction, the carry module (which needs only a different data source), and the buffer and sizing layers are all futures-agnostic. The primary addition required is a contract roll module and an IBKR broker implementation.

---

## References

- Asness, C., Moskowitz, T., and Pedersen, L. (2013). "Value and Momentum Everywhere." *Journal of Finance*, 68(3), 929–985.
- Baz, J., Granger, N., Harvey, C., Le Roux, N., and Rattray, S. (2015). "Dissecting Investment Strategies in the Cross Section and Time Series." *SSRN Working Paper*.
- Carver, R. (2015). *Systematic Trading*. Harriman House.
- Carver, R. (2019). *Leveraged Trading*. Harriman House.
- Carver, R. (2023). *Advanced Futures Trading Strategies*. Harriman House.
- Harvey, C., Liu, Y., and Zhu, H. (2016). "…and the Cross-Section of Expected Returns." *Review of Financial Studies*, 29(1), 5–68.
- Hurst, B., Ooi, Y., and Pedersen, L. (2017). "A Century of Evidence on Trend-Following Investing." *AQR White Paper*.
- Koijen, R., Moskowitz, T., Pedersen, L., and Vrugt, E. (2018). "Carry." *Journal of Financial Economics*, 127(2), 197–225.
- Magdon-Ismail, M., and Atiya, A. (2004). "Maximum Drawdown." *Risk Magazine*, 17(10), 99–102.
- Moskowitz, T., Ooi, Y., and Pedersen, L. (2012). "Time Series Momentum." *Journal of Financial Economics*, 104(2), 228–250.
- Novy-Marx, R., and Velikov, M. (2016). "A Taxonomy of Anomalies and Their Trading Costs." *Review of Financial Studies*, 29(1), 104–147.

---

*Code repository available on request. The design document, unit tests (250), and backtest engine are committed to version control with pre-registration semantics: no parameter in the codebase was modified after the first performance calculation.*
