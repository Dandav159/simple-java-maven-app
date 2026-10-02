# Stable vs. volatile stocks: is there an exploitable relationship?

Data: daily adjusted closes from Yahoo Finance, Jan 2014 – Sep 2026, for 50 US stocks plus SPY, ^VIX, TLT and GLD.

Method: every pattern is **found on 2014–2020 (train)** and **tested unchanged on 2021–Sep 2026 (test)**. Strategy returns include 5 bps of cost per unit of turnover.

```bash
pip install -r requirements.txt
python fetch_data.py   # -> data/prices.csv
python analyze.py      # -> output/results.json, output/*.png, output/*.csv

python fetch_sp500.py      # -> data/sp500_prices.csv, data/sp500_members.csv
python analyze_factors.py  # -> output/factors_results.json, output/factors_*.png
python compression.py      # -> output/compression_results.json (needs sp500 data)
python predict_compression.py  # -> output/predict_compression_results.json, .png
python grassmann.py        # -> output/grassmann_results.json, output/grassmann_*.png
python ica_pca.py          # -> output/ica_pca_results.json, output/ica_pca.png
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

## Part 2: economic structure as a prior (`analyze_factors.py`)

Instead of searching for patterns, this part uses signals whose direction is set **in advance** by published economic research. The model ranks ~400 S&P 500 members each month and buys the top 20% / shorts the bottom 20%, with signals neutralized within each sector. Costs are 10 bps per unit of turnover. Months run Jan 2015 – Sep 2026 (train 2015–2020, test 2021–2026).

| Factor | Prior direction | Economic story | Source |
|---|---|---|---|
| 12-1 month momentum | + | Investors under-react to news | Jegadeesh & Titman 1993 |
| 1-month reversal | − | Liquidity shocks revert | Jegadeesh 1990 |
| Low volatility | + | Leverage-constrained investors overpay for risky stocks | Frazzini & Pedersen 2014 |
| Near 52-week high | + | Anchoring delays the reaction to good news | George & Hwang 2004 |

Three ways to weight the factors: **prior** (literature signs, equal weights, nothing fitted), **data** (Fama-MacBeth slopes estimated on train), and **bayes** (a prior premium of 0.15%/month per z-score, shrunk toward the train estimate).

### Results

| Weights | Train Sharpe | **Test Sharpe** | Test t-stat | Test ann. return (long-short) |
|---|---:|---:|---:|---:|
| prior | −0.44 | **−0.50** | −1.19 | −7.6% |
| data | 0.41 | **0.52** | 1.24 | +6.8% |
| bayes | −0.15 | **−0.08** | −0.20 | −1.4% |

- **The priors did not hold in large-cap US stocks over 2015–2026.** No factor has a monthly rank IC with |t| > 1.3 in either period (`output/factors_ic.png`).
- Low volatility worked **backwards** in both periods: risky stocks beat safe ones through the mega-cap tech rally. Momentum and 52-week-high flipped sign between train and test.
- The "data" weights earned +6.8% a year out of sample, but only by betting on high volatility. Its t-stat of 1.24 is not significant: that is the 2021–2026 market rewarding risk, not a found edge.
- This matches the literature. Published anomalies shrink by roughly half after publication (McLean & Pontiff 2016), and most are concentrated in small caps, not the S&P 500 (Hou, Xue & Zhang 2020).

### The one prior that held: volatility clusters

Scaling SPY exposure by target volatility divided by trailing 21-day realized volatility (Moreira & Muir 2017), with the target set on train:

| | Train Sharpe | Test Sharpe | Test ann. return | Test max DD |
|---|---:|---:|---:|---:|
| SPY buy & hold | 0.78 | 0.91 | 14.8% | −24.5% |
| Vol-managed, max 1x | 0.95 | **0.97** | 10.7% | **−13.6%** |
| Vol-managed, max 2x | 0.89 | 0.85 | 9.9% | −14.6% |

The Sharpe improvement is small. The real benefit is **cutting the worst drawdown roughly in half**, at the cost of lower raw return. This is risk control, not a source of extra return: later studies find vol-managed portfolios often fail to beat buy-and-hold after realistic constraints (Cederburg et al. 2020).

## Part 3: prediction as compression (`compression.py`)

If daily moves contain predictable patterns, a sequence of moves should compress better than the same moves in shuffled order. Shuffling keeps symbol frequencies but destroys order. The test needs no trading rule: it measures the information in the sequence directly.

Each of ~500 stocks' ~3,100 daily returns (1.54M days in total) becomes symbols, which are then coded with an adaptive context model looking back 0–5 days. This is an exact arithmetic-coding length, so it is a genuine online predictor. The model runs per stock and also as one shared ("pooled") model across all stocks. lzma/bz2 serve as a sanity check.

| What is predicted | Max bits/day | Best pooled bits/day | Saving vs shuffled |
|---|---:|---:|---:|
| Direction (up/down) | 1.000 | 0.9985 | **0.03%** |
| Size (quartile of the day's move) | 2.000 | 1.9755 | **1.66%** |

- **Direction barely compresses.** The pattern found is a slight one-day reversal: P(up after a down day) = 52.9% vs P(up after an up day) = 51.4%. As a predictor, it implies **~51% accuracy**, worth roughly **0.026% per trade** against an average move of 1.35%. That is less than the cost of a round trip, and it is an in-sample upper bound.
- **Size compresses ~60× better than direction.** Volatility clusters, which is the same structure that made the volatility-managed overlay in Part 2 work.
- **Sharing one model across stocks helps.** Per stock, models that look back 2+ days do *worse* than looking back 1 day because each context sees too few examples. Pooled, longer memory keeps improving (1.66% at 5 days). Shared structure is what makes "learning from less" work.

### Part 3b: the compressor as a live forecaster (`predict_compression.py`)

The context model is turned into a next-day predictor and traded. It learns from 2014–2020, then each day of 2021–2026 it predicts every stock before seeing that day and learns from the day afterwards. Each symbol combines the stock's direction, whether its move was bigger than usual, and the market's direction (8 symbols). The model is pooled across all stocks. Each day it buys the top 10% by predicted P(up) and shorts the bottom 10%.

| Looks back | Accuracy | Always-guess-"up" accuracy | Before costs | **After 5 bps costs** |
|---|---:|---:|---:|---:|
| 1 day | 51.53% | 51.60% | +2.0%/yr (Sharpe 0.21) | **−35.2%/yr** |
| 2 days | 51.44% | 51.60% | +7.2%/yr (Sharpe 0.57) | **−31.3%/yr** |
| 3 days | 51.41% | 51.60% | +3.3%/yr (Sharpe 0.32) | **−33.6%/yr** |

Over 715,000 out-of-sample predictions, the compressor is **less accurate than always guessing "up"**. Ranking stocks earns a small pre-cost return, but none of it is statistically significant (the best case has t ≈ 1.4), and replacing most of the book every day turns it into a ~90% loss after costs.

## Part 4: market structure on a Grassmannian (`grassmann.py`)

The top-5 principal directions of ~450 stocks' standardized returns span a subspace: a point on Gr(5, n). The script measures its quarter-over-quarter rotation (geodesic distance from principal angles) and uses the subspace to separate shared moves from stock-specific residuals.

**A. Does rotation warn of turbulence?** No. Forecasting next month's SPY volatility out of sample:

| Inputs | Test R² |
|---|---:|
| Current volatility only | 0.303 |
| + subspace rotation | 0.301 |
| + absorption ratio (variance share of top 5) | 0.258 |

The rotation spiked in March 2020 *together with* the crash, not before it (`output/grassmann_rotation.png`). Distances sit around 2.0–2.6 radians (max ≈ 3.5), so directions 2–5 are largely re-estimated noise each quarter; only the dominant market direction is stable.

One lead, not a finding: faster rotation correlated with a *higher* next-month SPY return in the test period (r = 0.30, t ≈ 2.6) but not in train (r = 0.09). It was not specified in advance and was one of several correlations computed, so it needs fresh data before it means anything.

**B. Residual reversal (statistical arbitrage).** Weekly: remove the shared 5-D subspace, buy the decile whose residuals fell most, short the decile that rose most, hold 5 days.

| | Train before costs | Train after costs | Test before costs | **Test after costs** |
|---|---:|---:|---:|---:|
| Grassmannian residual reversal | +7.7% (SR 0.77) | −1.5% | +1.6% (SR 0.20) | **−7.2%** |
| Plain 5-day reversal (no geometry) | +21.0% (SR 0.72) | +10.8% | +6.8% (SR 0.40) | **−2.2%** |

Removing the shared subspace made the strategy **worse** than the plain control by 8.4%/yr in test (t = −0.96, i.e. not distinguishable). Neither is significant after 2020, and both lose money after costs.

## Part 5: is some blend of stocks predictable? (`ica_pca.py`)

Maybe no single stock is predictable but some portfolio of them is. Three rotations of 452 stocks' standardized returns, fit on 2014–2020 within the top 20 principal components (moves capped at 4 sd while fitting, otherwise March 2020 dominates the fit):

- **PCA:** the blends that move the most
- **ICA (FastICA):** statistically independent blends, the candidate "hidden drivers"
- **Box-Tiao (1977):** the blends whose tomorrow is *most predictable* from today, found by a generalized eigenproblem on a VAR(1)

The best Box-Tiao blend explains 25% of next-day variance in train. In test, **no blend from any method exceeds a 0.07 day-to-day correlation**. 7 of 60 fall outside the ±2/√n noise band, against ~3 expected by chance, all with |r| < 0.07 (`output/ica_pca.png`). Trading the 3 most predictable blends per method by the sign of today's move:

| Method | Train before costs | Test before costs | **Test after costs** |
|---|---:|---:|---:|
| PCA | +1.6% (SR 0.28) | +0.8% (SR 0.19, t 0.44) | **−11.1%** |
| ICA | +2.3% (SR 0.89) | +1.0% (SR 0.46, t 1.11) | **−11.2%** |
| Box-Tiao | +3.3% (SR 1.64) | +0.0% (SR 0.02, t 0.04) | **−12.2%** |

The method built to find predictability found the most in-sample and kept none of it, which is the signature of fitting noise. ICA held up best (all 3 picks kept their sign), but the edge is ~1%/yr, insignificant, and an order of magnitude smaller than the cost of trading it daily.

## Caveats
- **Parts 2–5 universe:** only current S&P 500 members could be downloaded. Each stock enters only from its index-inclusion date, but stocks that were removed (often losers) are still missing. This likely biases the result against low volatility, because high-volatility stocks that survived to 2026 are the winners.
- Part 2 uses price-based factors only. Value, profitability and quality need point-in-time fundamentals, which this data source doesn't provide.
- **Survivorship and selection bias:** the candidate lists were chosen in 2026 from stocks that still trade. Delisted high-volatility names are missing, which flatters the volatile basket.
- Daily close-to-close data only. Intraday lead-lag (minutes) can exist between liquid and illiquid names, but exploiting it needs tick data and low-latency execution.
- Costs are modeled as a flat 5 bps. Borrow costs for shorting volatile names (often high for MARA, RIOT, CVNA and PLUG) are excluded, which makes the short-volatile results look better than they would be.
- This is research, not investment advice.
