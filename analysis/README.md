# Stable vs. volatile stocks: is there an exploitable relationship?

Data: daily adjusted closes from Yahoo Finance, Jan 2014 – Sep 2026, for 50 US stocks plus SPY, ^VIX, TLT and GLD.

Method: every pattern is **found on 2014–2020 (train)** and **tested unchanged on 2021–Sep 2026 (test)**. Strategy returns include 5 bps of cost per unit of turnover.

```bash
pip install -r requirements.txt
python fetch_data.py   # -> data/prices.csv
python analyze.py      # -> output/results.json, output/*.png, output/*.csv
```

## Buckets (assigned by realized volatility in the train window only)

| Stable (15 lowest vol, ~18–21% ann.) | Volatile (15 highest vol, ~54–131% ann.) |
|---|---|
| VZ RSG PG KO JNJ WM PEP CL BRK-B COST KMB XEL GIS AEP WMT | TSLA FCX SHOP APA ETSY AMD TWLO SEDG W ROKU CVNA PLUG ENPH RIOT MARA |

Betas to SPY over the full period: stable basket **0.51**, volatile basket **1.72**.

## Findings

### 1. The two groups barely move together, except in a crisis
- Average pairwise correlation: stable–stable 0.46, volatile–volatile 0.29, **stable–volatile 0.10**.
- Basket-level correlation is 0.25 overall. The rolling 63-day value swings between −0.57 and +0.74 (`output/rolling_correlation.png`).
- **Correlation depends on the regime:** it is 0.05 when VIX < 15, 0.09 when VIX is 15–25, and **0.49 when VIX ≥ 25**. The diversification is gone in exactly the periods when it is needed. This is the most robust result in the study. It helps with risk management but is not a trading edge by itself.

### 2. Lead-lag: a statistically "significant" signal that isn't real
- In train, the stable basket's return today shows a correlation of −0.12 with the volatile basket's return tomorrow. Granger tests give p ≈ 1e-9.
- The effect comes **entirely from a handful of March 2020 crash days.** Excluding Feb 15–May 15 2020, the lag-1 correlation is −0.005. The rank (Spearman) correlation is 0.01 in train and 0.007 in test.
- In test, every lag sits inside the noise band (`output/lead_lag.png`).
- Pair level: 450 stock-to-stock lag-1 tests produce 101 that stay significant after Benjamin-Hochberg correction. **Only 53 of the 101 keep the same sign out of sample**, which is coin-flip odds. Train and test correlations across all pairs correlate at 0.007 (`output/pair_replication.png`).

### 3. Cointegration (225 stable/volatile pairs)
26 pairs reach raw p < 0.05, about what chance predicts (~11). **None survives multiple-testing correction.** Trading the top 5 out of sample loses money.

### 4. Strategy backtests (test window 2021–2026, after costs)

| Strategy (parameters fixed on train) | Train Sharpe | **Test Sharpe** | Test ann. return | Test max DD |
|---|---:|---:|---:|---:|
| Lead-lag: trade volatile basket off yesterday's stable move | −0.34 | **−0.34** | −23.4% | −83% |
| Rotation: 252-day momentum on stable-minus-volatile (best of 12 on train) | 1.00 | **0.49** | 12.6% | −47% |
| VIX switch: volatile basket when VIX < 14.5, else stable | 1.07 | **0.50** | 7.8% | −22% |
| Beta-neutral long stable / short volatile ("betting against beta") | −0.60 | **0.22** | 1.0% | −52% |
| Cointegration pairs (top 5) | 1.07 | **−0.67** | −10.1% | −50% |
| *Buy & hold SPY* | 0.78 | ***0.91*** | 14.8% | −25% |
| *Buy & hold stable basket* | 0.91 | *0.71* | 8.5% | −14% |
| *Buy & hold volatile basket* | 1.46 | *0.52* | 14.3% | −71% |

**No strategy beats buying and holding SPY, or even the stable basket alone, on a risk-adjusted basis out of sample.** Every in-sample Sharpe above 1 roughly halved or turned negative. No test-period t-stat reaches 2.

## Bottom line

Public daily price data shows **no exploitable pattern** between stable and volatile stocks. The real effects are structural: low unconditional correlation, correlations that jump in stress, and very different betas. These matter for portfolio construction, e.g. assuming diversification will fail when VIX spikes. They do not support a trading signal. The apparent signals are artifacts of a few extreme days, a large number of tests, and parameter selection.

## Caveats
- **Survivorship and selection bias:** the candidate lists were chosen in 2026 from stocks that still trade. Delisted high-volatility names are missing, which flatters the volatile basket.
- Daily close-to-close data only. Intraday lead-lag (minutes) can exist between liquid and illiquid names, but exploiting it needs tick data and low-latency execution.
- Costs are modeled as a flat 5 bps. Borrow costs for shorting volatile names (often high for MARA, RIOT, CVNA and PLUG) are excluded, which makes the short-volatile results look better than they would be.
- This is research, not investment advice.
