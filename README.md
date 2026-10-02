# VaR Backtesting on an Indian Stock + FX Portfolio

Computes 1-day 99% Value-at-Risk for an equal-weight portfolio (9 NSE stocks + USD/INR) four ways,
then tests how reliable each method was.

## Methods (250-day rolling window; day t uses only data before day t)
- **Historical simulation** - 1st percentile of the last 250 portfolio returns
- **Parametric** - assumes normal returns: mean + z x standard deviation
- **Monte Carlo** - 5,000 simulated scenarios from the covariance matrix, with fat tails (Student-t, 5 d.o.f.)
- **Filtered historical (EWMA)** - historical simulation on returns divided by EWMA volatility (lambda = 0.94), then scaled by today's volatility forecast. Meant to fix the problem that plain historical VaR reacts slowly when volatility jumps

## Backtest
- Exception = a day when the actual loss was bigger than the VaR
- **Kupiec test** - is the exception rate close to the expected 1%?
- **Christoffersen test** - do exceptions come in clusters?
- **Basel traffic light** - Green 0-4, Yellow 5-9, Red 10+ exceptions in the last 250 days
- **Exception analysis** - exceptions split into low / medium / high volatility days
- **Stress test** - worst day, worst 10-day period, and that period at 1.5x and 2x severity

## How to run
```
pip install -r requirements.txt
python get_real_data.py           # downloads real data to prices.csv (needs internet)
python var_backtest.py --csv prices.csv
python var_backtest.py --test     # fake data, only checks that the code runs
```
Creates `var_report.xlsx` and `var_backtest.png`.

To download real data first: `pip install polars pyarrow` and `python get_real_data.py`. It saves `prices.csv` (stocks + USDINR) and `usdinr.csv`.

## Results (real data: NSE adjusted prices from TejHQ + USD/INR, backtest 2016 to 1 Oct 2026, 2,641 days)
| Method | Exceptions (expected 26.4) | Rate | Kupiec | Christoffersen p | Basel zone (last 250d) |
|---|---|---|---|---|---|
| Historical | 36 | 1.36% | Pass (p = 0.076) | 0.013 (clustered) | Yellow (5) |
| Parametric | 43 | 1.63% | Reject (p = 0.003) | 0.195 | Yellow (7) |
| Monte Carlo (Student-t) | 27 | 1.02% | Pass (p = 0.908) | 0.280 | Green (3) |
| Filtered historical (EWMA) | 32 | 1.21% | Pass (p = 0.290) | 0.405 | Green (3) |

- Parametric VaR (normal assumption) was rejected: it had the most exceptions, 25 of its 43 on high-volatility days.
- Plain historical VaR reacted slowly in the March 2020 crash: 7 exceptions between February and April 2020, and its exceptions were clustered (Christoffersen p = 0.013). The EWMA version had 4 in the same months.
- The EWMA version has more exceptions on low-volatility days (12 vs 4 for historical): its VaR is lower than the plain historical estimate when markets are calm, which is the trade-off.
- Worst day: -11.5% (23 March 2020). Worst 10-day period: -26.7%, ending 23 March 2020. The same period at 1.5x and 2x severity gives -37.3% and -46.3%.
- Average 99% VaR: 2.36% (historical), 2.06% (parametric), 2.32% (Monte Carlo), 2.44% (EWMA).
- Data check: the largest one-day moves match known events (SBIN +27.7% on 25 Oct 2017, INFY -16% on 22 Oct 2019, COVID crash on 23 March 2020), so the adjusted prices look clean.
- Full tables: `results/var_report.xlsx`, chart: `results/var_backtest.png`. Random seed is fixed, so running the script again gives the same numbers.

## Limitations
- Equal weights, stocks and FX only (no options or bonds), 1-day VaR only
- Normal and Student-t assumptions may miss extreme crashes
- Stock prices: TejHQ Indian-markets dataset (NSE official end-of-day data), adjusted close. USD/INR from Yahoo Finance or FRED. Days when FX and equity markets are not both open use the previous FX rate
