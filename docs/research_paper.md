# Trend Following and Carry on an ETF Universe: An Evaluation of the Carver Framework

**Felix Garnier**  
Independent Research Project  
October 2026

---

## Abstract

This paper evaluates a systematic trading strategy implementing the full Carver (2015, 2019) framework — exponentially-weighted moving average crossover (EWMAC) trend rules, carry signals, volatility-scaled position sizing, a no-trade buffer, and an instrument diversification multiplier — on a universe of fourteen liquid US ETFs over the period January 2009 to October 2026. The rules, the weights, and the volatility target were written down before the first performance number and were not changed after it. The strategy achieves an annualised Sharpe ratio of 0.41 and a CAGR of 4.1%, against 0.88 for SPY, 0.99 for a daily-rebalanced 60/40, and 1.00 for a long-only book run through the same engine at a constant forecast of +10. Spread and stock-borrow account for about 0.063 Sharpe units, inside the speed limit of 0.167. Tables 1 to 4 are this book. They do not charge the T-bill on cash. Section 5.5 does. On excess returns the financed Sharpes are 0.31 for the strategy and 0.73 for the long-only book. A block bootstrap of that gap does not exclude zero at 95%. The evaluation window is 2 January 2009 to 13 October 2026, 4,472 days, which is 17.5 years at 256 days a year.

---

## 1. Introduction

Systematic trend-following and carry represent two of the most extensively documented sources of risk premia in financial markets. Moskowitz, Ooi and Pedersen (2012) establish time-series momentum across asset classes; Asness, Moskowitz and Pedersen (2013) document carry; Carver (2015, 2019) provides a practitioner framework combining both in a single multi-asset system. The canonical application is futures: diverse, liquid, with clean carry signals derived from term structure curves.

A natural question is whether the same architecture can be applied to an ETF universe a retail account can trade. The rules were frozen before the first performance number. The result is reported as it came out, including the underperformance.

The contribution is a complete implementation of the framework, a control that removes the signals, and a paper-trading loop that runs the same book. Three facts organise the tables. The slow trend, as its own book, has a Sharpe of 0.49, and the fast trend has 0.05. Standalone carry drawdowns exceed −55%. Removing the signals, and keeping the volatility target, the weights, and the spread and borrow costs, raises the Sharpe from 0.41 to 1.00. That comparison does not yet charge the T-bill on borrowed cash. Section 5.5 does, and it is the comparison used in the conclusion.

---

## 2. Related Literature

**Trend following.** Moskowitz, Ooi and Pedersen (2012) document that past twelve-month returns positively predict next-month returns across equity index, fixed income, currency and commodity futures. Baz et al. (2015) show similar results with EWMA-based filters. Hurst, Ooi and Pedersen (2017) extend the evidence to a century of data. Carver (2019) argues that multiple speeds (fast, medium, slow) each capture distinct information and that combining them via a forecast diversification multiplier (FDM) dominates any single speed.

**Carry.** Koijen et al. (2018) show that carry is positively related to expected returns across asset classes; the mechanism is different for equities (dividend yield minus funding cost), rates (yield spread), and commodities (futures basis). The key feature shared across asset classes is that carry is estimated from term structure information — futures curve slopes, dividend yields, rate differentials. On an ETF universe the futures basis is not observed. Equity carry is a dividend yield minus the bill, which is close to the carry of an equity-index future. Bond carry is a yield spread without the roll-down. Section 5.5 shows that this is not, by itself, the gap versus the long-only book.

**ETF-based systematic strategies.** Novy-Marx and Velikov (2016) and others note that strategy performance often degrades substantially when moved from futures to ETFs due to shorter histories, less clean carry proxies, and correlation structure dominated by equity beta.

**Design frozen before the results.** Harvey, Liu and Zhu (2016) argue that the literature suffers from massive multiple testing. The rules, the weights, and the volatility target in this project were committed before the first performance calculation and were not changed after it. A git history is not a third-party timestamp. Some universe choices used the sample: TIP carry was dropped because real yields looked biased over 2011–2022, and the list is of ETFs still listed in 2026. Those choices are stated in Section 3.1. They are not part of the frozen rule set.

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

SHY was excluded because its annualised volatility of about 1.5% produces a volatility scalar near 138× capital, which cannot sit under a 2× leverage cap. That reason does not require the backtest. TIP carry was excluded because real-yield estimates looked biased over 2011–2022. That reason uses the sample. The list itself is of funds still listed in 2026. Closed funds are absent.

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

with spread = 2 bps and a slippage multiplier of 1.5. Commissions are zero. Short positions pay an instrument-specific stock-borrow rate, from 0.3% a year on GLD to 0.5% on IEF and TLT, and only on the short holding. Section 5.5 adds the lagged T-bill on the cash balance. Tables 1 to 4 are struck before that charge.

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

The strategy has a lower Sharpe than the three benchmarks in this cost model. The long-only vol target is the comparison that removes the forecasts and keeps the rest. The gap of 0.59 is the forecasts, before any charge for borrowed cash. Spread and stock-borrow, 0.063 against 0.006 for the long-only book, leave about 0.53. Section 5.5 puts the T-bill on the cash and narrows the gap to 0.43 on total returns, and to 0.42 on excess returns.

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

Three of four periods produce a positive Sharpe. The block labelled 2022–2026 contains the published maximum drawdown of −29.9%. It is one block, not a year. Section 5.5 splits it. The label "2009–2012" in the code ends on 31 December 2011. No parameter is refit between blocks, so these are subperiods.

### 5.3 Rule-level decomposition

Each rule is evaluated by running the full engine on that rule's capped forecast alone. The Sharpe is the Sharpe of that book. It is not the marginal effect of adding the rule to the combination.

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

2. **Standalone carry drawdowns are large.** Maximum drawdowns exceed −55%. That is a description of those two books. It is not a test that the carry signal is mean-reverting, and it is not the contribution of carry inside the combined book. Section 5.5 measures that contribution.

3. **The combined book is below the best single rule.** The combined Sharpe of 0.41 sits below the standalone EWMAC 32/128 at 0.49. The difference is not the cost of carry. Carry's own books, in this table, have Sharpes of 0.18 and 0.17 and drawdowns past −55%. The slow trend, alone, is already 0.51 below the long-only control.

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

Commodities and fixed income drive positive attribution; international equities are mildly negative. These are shares of the portfolio P&L, not standalone sleeves.

### 5.5 Cash financing, added after the tables

Tables 1 to 4 do not charge the T-bill on cash. A long position above the account's equity was borrowing for free. Short proceeds earned nothing. The rate was already in the data, lagged one day, and used by the carry rule. Putting it on the cash balance changes no forecast, no weight, and no volatility target. The sample in this section runs to 23 October 2026, ten days past Table 1. That is not what moves the Sharpes.

Cash is equity minus the marked value of the positions. A negative balance pays the lagged three-month bill. A positive balance, which is what a short leaves behind, earns it. The excess Sharpe then subtracts the same bill from the daily return, so a book that only holds cash scores zero. SPY and the 60/40 are unlevered. Their total-return Sharpe is the one in Table 1. Only the excess column charges them for the bill.

**Table 5. Same books after the cash charge.**

| Book | Total Sharpe | Excess Sharpe | CAGR | Volatility | Max DD |
|---|---:|---:|---:|---:|---:|
| Strategy | 0.44 | 0.31 | 4.5% | 11.5% | −25.1% |
| Long-only vol target | 0.87 | 0.73 | 8.7% | 10.2% | −20.3% |
| SPY | 0.87 | 0.80 | 15.0% | 17.8% | −33.7% |
| 60/40 | 0.99 | 0.85 | 10.2% | 10.4% | −21.0% |

Table 6 is the financed book by year. It is a presentation cut, not a new estimation. It is there so the fourth block is not read as "2022".

**Table 6. Annual total return of the financed book.**

| Year | Strategy | Long-only | SPY |
|---|---:|---:|---:|
| 2009 | 2.3% | 15.2% | 26.4% |
| 2010 | 8.7% | 21.2% | 15.1% |
| 2011 | 4.4% | 9.3% | 1.9% |
| 2012 | −0.9% | 12.0% | 16.0% |
| 2013 | 6.9% | −2.6% | 32.3% |
| 2014 | 10.5% | 6.0% | 13.5% |
| 2015 | −4.4% | −8.7% | 1.3% |
| 2016 | 6.4% | 14.0% | 12.0% |
| 2017 | 6.2% | 15.6% | 21.7% |
| 2018 | 4.8% | −7.5% | −4.6% |
| 2019 | 7.5% | 26.5% | 31.2% |
| 2020 | 15.1% | 9.4% | 18.4% |
| 2021 | 6.5% | 14.1% | 28.7% |
| 2022 | 7.6% | −14.0% | −18.2% |
| 2023 | −8.1% | 9.4% | 26.2% |
| 2024 | −5.5% | 12.0% | 24.9% |
| 2025 | 9.6% | 20.7% | 17.7% |
| 2026 | 3.7% | 9.2% | 12.3% |

The control falls from 1.00 to 0.87. It was the book that borrowed and did not pay. The strategy moves from 0.41 to 0.44. On excess returns the figures that matter are 0.31 and 0.73. A total-return Sharpe counts the bill the book receives. An excess Sharpe takes it back out.

**Table 7. Excess Sharpe of each rule as its own book, at three rebates on short proceeds.** The debit that funds longs always pays the full bill. Idle cash always earns it. Only the interest on short proceeds changes. A retail stock account often receives little or none of that rebate.

| Rule | Rebate 100% | Rebate 50% | Rebate 0% |
|---|---:|---:|---:|
| Strategy | 0.31 | 0.28 | 0.25 |
| Long-only | 0.73 | 0.73 | 0.73 |
| EWMAC 8/32 | −0.03 | −0.06 | −0.09 |
| EWMAC 16/64 | 0.24 | 0.21 | 0.18 |
| EWMAC 32/128 | 0.36 | 0.34 | 0.31 |
| Carry medium | 0.27 | 0.20 | 0.14 |
| Carry slow | 0.26 | 0.19 | 0.13 |

The carry books move the most. Part of their rise from the 0.18 in Table 3 to about 0.40 in total return is the bill earned on short proceeds, not a stronger signal. At a zero rebate their excess Sharpes are 0.14 and 0.13. The long-only book does not change with the rebate, because it holds no shorts.

A book that keeps only EWMAC 16/64 and 32/128, with those two weights renormalised and the same FDM machinery, has an excess Sharpe of 0.32. Dropping the fast trend and both carry rules does not close the gap to 0.73.

The position that earns the next day's return is linear in the forecast before the buffer and the leverage cap. Inside the combined book, each rule's capped forecast, times its renormalised weight, times the FDM of that day, is its share of the instrument forecast. The traded position's price P&L is split in that proportion, using the forecast from the previous close. The pieces add up to the price P&L. They are not standalone books.

**Table 8. Contribution to the combined book's price P&L, compounded within the year.**

| Rule | 2022 | 2023 | 2024 |
|---|---:|---:|---:|
| EWMAC 8/32 | +4.4% | −0.8% | −2.1% |
| EWMAC 16/64 | +3.2% | −2.8% | +0.4% |
| EWMAC 32/128 | +4.1% | −4.7% | +3.7% |
| Carry medium | −2.5% | −1.8% | −2.8% |
| Carry slow | −2.5% | −1.9% | −2.8% |

In 2023 the three trend rules contribute about −8.3 percentage points and the two carry rules about −3.7. The slow trend is the largest single piece, not the whole loss. In 2024 the two carry rules contribute about −5.6 points, the slow trend +3.7, and the fast trend −2.1. The 2024 price loss is the carry. The class split of the same combined book, which includes costs, is unchanged: 2023 is negative in every class, and 2024 is negative in equities and bonds and positive in commodities and FX.

What is measured about equities, and only this, is the sign of the slow carry. From 2023 it is negative on essentially every day for SPY, IWM, VGK, EWJ, EEM, EPP and EWC. The rule was short those equities. Table 6 shows the trend book making money in 2020 and in 2022. A general claim that short equity trend lost the bull market is not the result.

The excess-return Sharpe of the strategy minus the long-only book is −0.42. `block_bootstrap_sharpe_difference`, 2,000 circular block draws with seed 0, gives a 95% interval of about [−1.03, +0.17] at a 21-day block and [−0.96, +0.10] at a 63-day block. Both include zero. The fraction of draws in which the strategy is ahead is about 0.08 and 0.05. The point estimate is a gap. It is not separated from zero at 95%.

A regression of the strategy's daily excess return on the long-only book's, with Newey-West standard errors and 21 lags, has a beta of 0.12 (t = 1.54) and an annualised alpha of +2.7% (t = 1.08), on 4,479 days. The strategy is not a scaled copy of the control. The alpha is not distinguishable from zero either.

The evaluation window is 17.5 years when a year is 256 trading days. Calendar time from January 2009 to October 2026 is a few months longer. The standard error of an annual Sharpe over 17.5 years is about 0.24, which is why the bootstrap interval is wide.

---

## 6. Discussion

### 6.1 Why the signals underperform on ETFs

The long-only vol target in Table 1 has a Sharpe of 1.00. The strategy has 0.41. The difference is the forecasts, on a book that does not yet charge the T-bill for borrowed cash. Section 5.5 charges it. Equity carry in this implementation is the trailing dividend yield minus the T-bill, which is also, roughly, the carry of an equity-index future. Bond carry uses a daily FRED yield and has no roll-down. Commodity and FX carry are absent. A slow trend book at 0.49 does not need a weak carry proxy to sit far below the control.

**The bull market is not, by itself, a demonstrated short-trend loss.** Table 6 shows the financed strategy up in 2020 and in 2022, two years in which equities were not a smooth bull market. What is measured is the carry sign from 2023, in Section 5.5. International equities are the negative class contribution in Table 4. That is a contribution, not a test that short trend was the mechanism.

### 6.2 What the framework does correctly

Despite the signal underperformance, several design elements function as intended:

- **Volatility targeting and the buffer do what they were built to do.** Mean leverage is 1.72× and the cap binds at 2×. Turnover is 24.1×. About 29% of share variation is absorbed by the buffer.
- **The long-only book in Table 1** has a Sharpe of 1.00, next to 0.99 for the 60/40, with a smaller drawdown (−19.8% against −21.0%). That comparison does not charge borrowed cash. In Section 5.5 the same long-only book has a total Sharpe of 0.87, below the unlevered 60/40 at 0.99. The allocation stops looking like a free improvement once the bill is paid.

### 6.3 No-tuning commitment

No rule, weight, or volatility target was revised after the backtest was run. The design document was committed before the first performance calculation. A git commit is not a third-party timestamp, and Section 3.1 records the universe choices that used the sample. The Sharpe of 0.41 in Table 1 is the number produced by that frozen rule set under the original cost model. It is not a claim that the same number would survive every omitted cost. Section 5.5 is the check that was missing, and it is labelled as a correction of the engine.

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

The rules were frozen before the first performance number and were not refit. On the original cost model the signals trail the long-only book, 0.41 against 1.00. Once borrowed cash pays the T-bill, the excess-return Sharpes are 0.31 and 0.73. The difference is −0.42, and a block bootstrap does not exclude zero at 95%. A regression on the long-only book has a beta of 0.12 and an annualised alpha of +2.7%, neither distinguishable from zero at conventional levels. Inside the combined book, 2023's price loss is spread across the trend rules, and 2024's is the carry. 2022 is a positive year.

The implementation, the control, and the paper-trading loop are the deliverable. A futures version needs a roll module and another broker. It does not need a new sizing engine.

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

*The rules, the weights, and the volatility target were not modified after the first performance calculation. Section 3.1 records the universe choices that used the sample. Section 5.5 is an engine correction, not a new rule.*
