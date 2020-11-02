import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, objective_functions
import datetime as dt

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def equal_value_weighted_ptf(stocks, data, mu):
    evw_ptf_value = [1]
    evw_equal_weights = [(1 / len(stocks)) / data.iloc[0]]

    for index in np.arange(1, len(data)):
        evw_gain = np.dot(evw_equal_weights[index - 1], data.iloc[index])
        if index < len(df):
            evw_equal_weights.append((evw_gain / len(stocks)) / data.iloc[index])
            evw_change = evw_equal_weights[index] - evw_equal_weights[index - 1]
            evw_abs_change = [abs(number) for number in evw_change]
            evw_trans_cost = evw_ptf_value[index - 1] * mu * np.sum(evw_abs_change)
            evw_ptf_value.append(evw_gain - evw_trans_cost)
        else:
            evw_ptf_value.append(evw_gain)
    return evw_ptf_value


def equal_weighted_ptf(stocks, data):
    ew_norm_ptf_value = [1]
    ew_equal_weights = 1 / len(stocks)
    ew_ptf_value = [np.sum(ew_equal_weights * data.iloc[0])]

    for index in np.arange(1, len(df)):
        ew_ptf_value.append(np.sum(ew_equal_weights * data.iloc[index]))
        ew_norm_ptf_value.append(ew_ptf_value[index] / ew_ptf_value[0])
    return ew_norm_ptf_value


def minimum_variance_ptf(data_est, mu, n):
    daily_ret = data_est[0:n].pct_change()
    mv_mu = daily_ret[0:n].mean()
    mv_cov_daily = data_est[0:n].cov()

    ef = EfficientFrontier(mv_mu, mv_cov_daily, weight_bounds=(0, 1))
    ef.min_volatility()
    mv_weights = [pd.Series(ef.weights) / np.sum(pd.Series(ef.weights))]
    mv_ptf_value = [pd.Series(np.dot(mv_weights, data_est.iloc[n]))]
    mv_norm_ptf_value = [1]

    for index in np.arange(n + 1, len(data_est)):
        mv_gain = np.dot(mv_weights[index - (n + 1)], data_est.iloc[index])
        if index < len(data_est):
            mv_cov_daily = df_est[(index - n):index].cov()
            ef = EfficientFrontier(mv_mu, mv_cov_daily, weight_bounds=(0, 1))
            ef.min_volatility()
            mv_weights.append(pd.Series(ef.weights) / np.sum(pd.Series(ef.weights)))
            mv_change = mv_weights[index - n] - mv_weights[index - (n + 1)]
            mv_abs_change = [abs(number) for number in mv_change]
            mv_trans_cost = mv_ptf_value[index - (n + 1)] * mu * np.sum(mv_abs_change)
            mv_ptf_value.append(pd.Series(mv_gain - mv_trans_cost))
            mv_norm_ptf_value.append(mv_ptf_value[index - n] / mv_ptf_value[0])
        else:
            mv_ptf_value.append(pd.Series(mv_gain))
            mv_norm_ptf_value.append(mv_ptf_value[index - n] / mv_ptf_value[0])
    return mv_norm_ptf_value


def min_risk_giv_ret_ptf(data_est, mu, n):
    daily_ret = data_est[0:n].pct_change()
    mrr_mu = daily_ret[0:n].mean()
    mrr_cov_daily = data_est[0:n].cov()

    ef = EfficientFrontier(mrr_mu, mrr_cov_daily, weight_bounds=(0, 1))
    ef.add_objective(objective_functions.L2_reg, gamma=10)
    ef.efficient_return(target_return=0.0007)
    mrr_weights = [pd.Series(ef.weights) / np.sum(pd.Series(ef.weights))]
    mrr_ptf_value = [pd.Series(np.dot(mrr_weights, data_est.iloc[n]))]
    mrr_norm_ptf_value = [1]

    for index in np.arange((n + 1), len(df_est)):
        mrr_gain = np.dot(mrr_weights[index - (n + 1)], data_est.iloc[index])
        if index < len(df_est):
            mrr_cov_daily = data_est[(index - n):index].cov()
            ef = EfficientFrontier(mrr_mu, mrr_cov_daily, weight_bounds=(0, 1))
            ef.add_objective(objective_functions.L2_reg, gamma=10)
            ef.efficient_return(target_return=0.0007)
            mrr_weights.append(pd.Series(ef.weights) / np.sum(pd.Series(ef.weights)))
            mrr_change = mrr_weights[index - n] - mrr_weights[index - (n + 1)]
            mrr_abs_change = [abs(number) for number in mrr_change]
            mrr_trans_cost = mrr_ptf_value[index - (n + 1)] * mu * np.sum(mrr_abs_change)
            mrr_ptf_value.append(pd.Series(mrr_gain - mrr_trans_cost))
            mrr_norm_ptf_value.append(mrr_ptf_value[index - n] / mrr_ptf_value[0])
        else:
            mrr_ptf_value.append(pd.Series(mrr_gain))
            mrr_norm_ptf_value.append(mrr_ptf_value[index - n] / mrr_ptf_value[0])
    return mrr_norm_ptf_value


stocks = ["AAPL", "AMZN", "CVX", "FB", "GOOG",
          "JNJ", "JPM", "KO", "MA", "MRK",
          "MSFT", "PG", "PFE"]

# Import data from yahoo finance ( df = dataset di partenza,
# df_est = dataset per le stime, ha n giorni in più)

start_df = dt.datetime(2017, 3, 24)
end_df = dt.datetime(2020, 6, 1)
df = yf.download(stocks, start_df, end_df)
df = df['Adj Close']
start_df_est = dt.datetime(2013, 4, 5)
end_df_est = dt.datetime(2020, 6, 1)
df_est = yf.download(stocks, start_df_est, end_df_est)
df_est = df_est['Adj Close']
n = len(df_est) - len(df)

# Equal Value-Weighted Portfolio + Equal Weighted Portfolio

evw_mu_tc = 5 * 10 ** (-4)
evw_ptf_value = equal_value_weighted_ptf(stocks, df, evw_mu_tc)
ew_norm_ptf_value = equal_weighted_ptf(stocks, df)


# Minimum Variance Portfolio

mv_mu_tc = 5 * 10 ** (-4)
mv_norm_ptf_value = minimum_variance_ptf(df_est, mv_mu_tc, n)

# Minimise risk for a given return

# mrr_mu_tc = 5 * 10 ** (-4)
# mrr_norm_ptf_value = minimum_variance_ptf(df_est, mrr_mu_tc, n)

# Plot

plt.style.use('seaborn')
p1 = plt.plot(df.index, ew_norm_ptf_value, "-b", label="Equal Weight")
p2 = plt.plot(df.index, evw_ptf_value, "-r", label="Equal Value-Weight")
p3 = plt.plot(df.index, mv_norm_ptf_value, "-g", label="Minimum Variance")
# p4 = plt.plot(df.index, mrr_norm_ptf_value, "-k", label="Minimise risk for a given Return")
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend(loc="upper left")
plt.show()
