"""
Project 1: VaR backtesting on an Indian stock + FX portfolio
Computes 1-day 99% VaR four ways, then checks how well each one worked.

Run:  python var_backtest.py                      (real data from Yahoo Finance)
      python var_backtest.py --csv prices.csv     (your own file: Date column + one close-price column per stock)
      python var_backtest.py --test               (fake data, only to test the code)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
           "SBIN.NS", "ITC.NS", "LT.NS", "BHARTIARTL.NS", "USDINR=X"]
START = "2015-01-01"
CONF = 0.99          # confidence level
WINDOW = 250         # days of history used for each VaR number
N_SIMS = 5000        # Monte Carlo simulations
DF = 5               # degrees of freedom of the Student-t used in Monte Carlo
LAMBDA = 0.94        # decay factor for the EWMA volatility (RiskMetrics value)


# ---------------------------------------------------------------- 1. data
def read_csv_prices(path):
    """Read a wide CSV (first column = date, other columns = close prices).
    Column names can be like RELIANCE, RELIANCE.NS, USDINR or USDINR=X."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    short = {t.replace(".NS", "").replace("=X", ""): t for t in TICKERS}
    renamed = {}
    for col in df.columns:
        key = str(col).upper().replace(".NS", "").replace("=X", "").replace("/", "").strip()
        if key in short:
            renamed[col] = short[key]
    df = df[list(renamed)].rename(columns=renamed)
    missing = [t for t in TICKERS if t not in df.columns]
    if missing:
        print("Columns not found in the CSV, leaving these out:", missing)
    return df.ffill().dropna()

def get_returns(use_fake_data):
    if "--csv" in sys.argv:
        prices = read_csv_prices(sys.argv[sys.argv.index("--csv") + 1])
        return np.log(prices / prices.shift(1)).dropna(), False
    if not use_fake_data:
        try:
            import yfinance as yf
            prices = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]
            prices = prices.ffill().dropna()
            if len(prices) > 500:
                return np.log(prices / prices.shift(1)).dropna(), False
        except Exception as e:
            print("Download failed:", e)
        print("Using fake data instead.")
    # fake data: random returns with some volatility clustering (for testing only)
    np.random.seed(1)
    n, k = 2500, len(TICKERS)
    vol = 0.01
    data = np.zeros((n, k))
    for t in range(n):
        market = np.random.standard_t(5)
        data[t] = vol * (0.6 * market + 0.8 * np.random.standard_t(5, k))
        vol = np.sqrt(2e-6 + 0.09 * data[t].mean() ** 2 + 0.89 * vol ** 2)
    dates = pd.bdate_range(START, periods=n)
    return pd.DataFrame(data, index=dates, columns=TICKERS), True


# ---------------------------------------------------------------- 2. VaR methods
# Each function takes past returns of the assets (window x assets) and portfolio weights.
# VaR is returned as a positive number (the loss as a fraction of portfolio value).

def historical_var(window_returns, w):
    port = window_returns @ w
    return -np.percentile(port, (1 - CONF) * 100)

def parametric_var(window_returns, w):
    port = window_returns @ w
    z = stats.norm.ppf(1 - CONF)
    return -(port.mean() + z * port.std())

def monte_carlo_var(window_returns, w):
    mean = window_returns.mean(axis=0)
    cov = np.cov(window_returns, rowvar=False)
    # simulate asset returns from a multivariate normal, then make the tails fatter (Student-t)
    sims = np.random.multivariate_normal(np.zeros(len(w)), cov, N_SIMS)
    chi = np.random.chisquare(DF, N_SIMS)
    sims = sims * np.sqrt((DF - 2) / chi)[:, None] + mean
    return -np.percentile(sims @ w, (1 - CONF) * 100)


def ewma_volatility(port_ret):
    """EWMA volatility forecast for every day. Entry t only uses returns BEFORE day t."""
    var = np.zeros(len(port_ret))
    var[0] = port_ret[:30].var()
    for t in range(1, len(port_ret)):
        var[t] = LAMBDA * var[t - 1] + (1 - LAMBDA) * port_ret[t - 1] ** 2
    return np.sqrt(var)

def filtered_historical_var(t, port_ret, sigma):
    """Historical simulation on returns divided by their volatility at the time,
    then multiplied by today's volatility forecast. Reacts faster when markets get volatile."""
    past = port_ret[t - WINDOW:t] / sigma[t - WINDOW:t]
    return -np.percentile(past, (1 - CONF) * 100) * sigma[t]


# ---------------------------------------------------------------- 3. backtest
def kupiec_test(n_exceptions, n_days):
    """Kupiec test: is the share of exceptions close to the expected 1%?"""
    p = 1 - CONF
    x, n = n_exceptions, n_days
    phat = x / n
    # log-likelihood under H0 (rate = p) and under the observed rate
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + (x * np.log(phat) if x > 0 else 0)
    lr = -2 * (ll0 - ll1)
    p_value = 1 - stats.chi2.cdf(lr, 1)
    return lr, p_value

def christoffersen_test(hits):
    """Christoffersen test: do the exceptions come one after another (clusters)?
    hits is a list of 0/1, 1 = exception."""
    hits = np.array(hits, dtype=int)
    prev, now = hits[:-1], hits[1:]
    n00 = np.sum((prev == 0) & (now == 0))
    n01 = np.sum((prev == 0) & (now == 1))
    n10 = np.sum((prev == 1) & (now == 0))
    n11 = np.sum((prev == 1) & (now == 1))
    if n01 + n11 == 0 or n10 + n11 == 0:
        return np.nan                      # too few exceptions to test
    p01 = n01 / (n00 + n01)                # chance of an exception after a normal day
    p11 = n11 / (n10 + n11)                # chance of an exception after an exception
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    def ll(prob, zeros, ones):             # log-likelihood of zeros/ones with probability prob
        return (zeros * np.log(1 - prob) if zeros > 0 else 0) + (ones * np.log(prob) if ones > 0 else 0)
    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return 1 - stats.chi2.cdf(lr, 1)

def basel_zone(exceptions_in_250_days):
    if exceptions_in_250_days <= 4:
        return "Green"
    elif exceptions_in_250_days <= 9:
        return "Yellow"
    return "Red"


# ---------------------------------------------------------------- 4. main
def main():
    np.random.seed(42)                      # same random numbers every run, so results can be repeated
    returns, fake = get_returns("--test" in sys.argv)
    weights = np.ones(returns.shape[1]) / returns.shape[1]    # equal weights
    port_ret = returns.values @ weights
    dates = returns.index

    # rolling VaR: the VaR for day t uses only the 250 days BEFORE day t
    methods = {"Historical": historical_var, "Parametric": parametric_var, "Monte Carlo": monte_carlo_var}
    names = list(methods) + ["Filtered historical (EWMA)"]
    sigma = ewma_volatility(port_ret)
    var = {name: [] for name in names}
    for t in range(WINDOW, len(returns)):
        window = returns.values[t - WINDOW:t]
        for name, f in methods.items():
            var[name].append(f(window, weights))
        var["Filtered historical (EWMA)"].append(filtered_historical_var(t, port_ret, sigma))
    var = pd.DataFrame(var, index=dates[WINDOW:])
    actual = pd.Series(port_ret[WINDOW:], index=dates[WINDOW:])

    # exceptions = days when the actual loss was bigger than the VaR
    rows = []
    for name in names:
        hits = (-actual) > var[name]
        n_days, n_exc = len(hits), int(hits.sum())
        lr, pval = kupiec_test(n_exc, n_days)
        rows.append({"Method": name, "Days": n_days, "Exceptions": n_exc,
                     "Expected": round(n_days * (1 - CONF), 1),
                     "Exception rate %": round(100 * n_exc / n_days, 2),
                     "Kupiec p-value": round(pval, 3),
                     "Kupiec": "Reject" if pval < 0.05 else "Pass",
                     "Christoffersen p-value": round(christoffersen_test(hits.values), 3),
                     "Basel zone (last 250d)": basel_zone(int(hits.iloc[-250:].sum()))})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # exception analysis: do exceptions happen more when volatility is high?
    vol20 = actual.rolling(20).std().shift(1)            # volatility known before the day starts
    regime = pd.qcut(vol20, 3, labels=["Low vol", "Medium vol", "High vol"])
    reg_rows = []
    for name in names:
        hits = (-actual) > var[name]
        for r in ["Low vol", "Medium vol", "High vol"]:
            mask = regime == r
            reg_rows.append({"Method": name, "Regime": r, "Days": int(mask.sum()),
                             "Exceptions": int(hits[mask].sum())})
    regimes = pd.DataFrame(reg_rows)
    print("\n", regimes.to_string(index=False))

    # stress test: worst 10-day period in the data, and what if it was 1.5x / 2x worse
    ten_day = pd.Series(port_ret, index=dates).rolling(10).sum().dropna()
    worst_end = ten_day.idxmin()
    worst = ten_day.min()
    def to_pct_loss(log_ret):
        return round(100 * (np.exp(log_ret) - 1), 2)     # log return -> % change in portfolio value
    stress = pd.DataFrame({
        "Scenario": ["Worst single day", "Worst 10-day period (ends " + str(worst_end.date()) + ")",
                     "Worst 10-day period x1.5", "Worst 10-day period x2"],
        "Portfolio loss %": [to_pct_loss(port_ret.min()), to_pct_loss(worst),
                             to_pct_loss(1.5 * worst), to_pct_loss(2 * worst)]})
    print("\n", stress.to_string(index=False))

    # chart
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for ax, name in zip(axes, names):
        hits = (-actual) > var[name]
        ax.plot(actual.index, 100 * actual, color="grey", lw=0.6, label="Daily return")
        ax.plot(var.index, -100 * var[name], color="red", lw=1, label="-VaR 99%")
        ax.scatter(actual.index[hits], 100 * actual[hits], color="black", s=12, label="Exceptions")
        ax.set_title(name + (" (FAKE DATA)" if fake else ""))
        ax.set_ylabel("%")
        ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig("var_backtest.png", dpi=150)

    # excel report
    with pd.ExcelWriter("var_report.xlsx") as xl:
        summary.to_excel(xl, sheet_name="Backtest", index=False)
        regimes.to_excel(xl, sheet_name="Exceptions by regime", index=False)
        stress.to_excel(xl, sheet_name="Stress", index=False)
        (100 * var).round(3).to_excel(xl, sheet_name="Daily VaR %")
    print("\nSaved var_report.xlsx and var_backtest.png")
    if fake:
        print("WARNING: fake data used. Do not quote these numbers anywhere.")


if __name__ == "__main__":
    main()
