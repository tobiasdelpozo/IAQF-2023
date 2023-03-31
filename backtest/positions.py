import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import RANSACRegressor, HuberRegressor

# Robust market beta estimation for ratio sizing.


def ransac_beta(asset_1, asset_2, mkt):
    # Calculate each asset's beta to the market.
    # Use RANSAC to remove outliers.
    log_rets_a1 = (np.log(asset_1) - np.log(asset_1.shift(1))).cumsum().dropna()
    log_rets_a2 = (np.log(asset_2) - np.log(asset_2.shift(1))).cumsum().dropna()
    log_rets_mkt = (np.log(mkt) - np.log(mkt.shift(1))).cumsum().dropna()
    m1 = RANSACRegressor(random_state=0)
    m1.fit(np.array(log_rets_mkt).reshape(-1, 1), log_rets_a1)
    beta_1 = m1.estimator_.coef_
    m2 = RANSACRegressor(random_state=0)
    m2.fit(np.array(log_rets_mkt).reshape(-1, 1), log_rets_a2)
    beta_2 = m2.estimator_.coef_
    dollar_beta_1 = beta_1[0] * asset_1[-1]
    dollar_beta_2 = beta_2[0] * asset_2[-1]
    hedge_ratio = dollar_beta_1 / dollar_beta_2
    return hedge_ratio


def huber_beta(asset_1, asset_2, mkt):
    # Calculate each asset's beta to the market.
    # Use Huber loss function to remove outliers.
    log_rets_a1 = (np.log(asset_1) - np.log(asset_1.shift(1))).cumsum().dropna()
    log_rets_a2 = (np.log(asset_2) - np.log(asset_2.shift(1))).cumsum().dropna()
    log_rets_mkt = (np.log(mkt) - np.log(mkt.shift(1))).cumsum().dropna()
    m1 = HuberRegressor()
    m1.fit(np.array(log_rets_mkt).reshape(-1, 1), log_rets_a1)
    beta_1 = m1.coef_
    m2 = HuberRegressor()
    m2.fit(np.array(log_rets_mkt).reshape(-1, 1), log_rets_a2)
    beta_2 = m2.coef_
    dollar_beta_1 = beta_1[0] * asset_1[-1]
    dollar_beta_2 = beta_2[0] * asset_2[-1]
    hedge_ratio = dollar_beta_1 / dollar_beta_2
    return hedge_ratio


def beta_calc(asset_1, asset_2, mkt=None, mod=3):
    # Calculate each asset's beta to the market.
    # Use RANSAC to remove outliers.

    if mod in [1, 4, 5]:
        a1 = (np.log(asset_1) - np.log(asset_1.shift(1))).cumsum().dropna()
        a2 = (np.log(asset_2) - np.log(asset_2.shift(1))).cumsum().dropna()
        mkt_data = (np.log(mkt) - np.log(mkt.shift(1))).cumsum().dropna()

    if mod == 4:
        m1 = RANSACRegressor(random_state=0)
        m1.fit(np.array(a1).reshape(-1, 1), np.array(mkt_data).reshape(-1, 1))
        beta_1 = m1.estimator_.coef_[0]
        m2 = RANSACRegressor(random_state=0)
        m2.fit(np.array(a2).reshape(-1, 1), np.array(mkt_data).reshape(-1, 1))
        beta_2 = m2.estimator_.coef_[0]
        dollar_beta_1 = beta_1 * asset_1[-1]
        dollar_beta_2 = beta_2 * asset_2[-1]
        hedge_ratio = dollar_beta_1 / dollar_beta_2
        return hedge_ratio * beta_2 / beta_1

    elif mod == 5:
        m1 = RANSACRegressor(random_state=0)
        m1.fit(np.array(a1).reshape(-1, 1), np.array(mkt_data).reshape(-1, 1))
        beta_1 = m1.estimator_.coef_[0]
        m2 = RANSACRegressor(random_state=0)
        m2.fit(np.array(a2).reshape(-1, 1), np.array(mkt_data).reshape(-1, 1))
        beta_2 = m2.estimator_.coef_[0]
        dollar_beta_1 = beta_1 * asset_1[-1]
        dollar_beta_2 = beta_2 * asset_2[-1]
        hedge_ratio = dollar_beta_1 / dollar_beta_2
        return hedge_ratio

    elif mod == 1:
        global temp1, temp2
        temp1 = asset_1
        temp2 = asset_2
        return asset_1[-1] / asset_2[-1]

    if mod in [2, 3]:
        a1 = asset_1
        a2 = asset_2
        mkt_data = mkt

    if mod == 2:
        m1 = RANSACRegressor(random_state=0)
        m1.fit(np.array(mkt_data).reshape(-1, 1), np.array(a1).reshape(-1, 1))
        beta_1 = m1.estimator_.coef_[0]
        m2 = RANSACRegressor(random_state=0)
        m2.fit(np.array(mkt_data).reshape(-1, 1), np.array(a2).reshape(-1, 1))
        beta_2 = m2.estimator_.coef_[0]
        return beta_1 / beta_2

    if mod == 3:
        m1 = RANSACRegressor(random_state=0)
        m1.fit(np.array(a2).reshape(-1, 1), np.array(a1).reshape(-1, 1))
        beta = m1.estimator_.coef_[0]
        return beta


def calc_rolling_betas(asset_1, asset_2, mkt=None, mod=3, WINDOW_SIZE=10):
    # These betas SHOULD be lagged already
    betas = pd.DataFrame(index=asset_1.iloc[WINDOW_SIZE:].index, columns=["Betas"])
    start = 0
    end = WINDOW_SIZE
    for i in betas.index:
        if mkt is None:
            beta = beta_calc(
                asset_1.iloc[start : (start + WINDOW_SIZE)],
                asset_2.iloc[start : (start + WINDOW_SIZE)],
                mod=mod,
            )
        else:
            beta = beta_calc(
            asset_1.iloc[start : (start + WINDOW_SIZE)],
            asset_2.iloc[start : (start + WINDOW_SIZE)],
            mkt.iloc[start : (start + WINDOW_SIZE)],
            mod=mod,
            )
        betas.loc[i] = beta
        start = start + 1
    # Truncate beta to be 2 decimal places
    # and then scale position sizes to be the same.
    return betas.astype(float)


# if __name__ == "__main__":
#     data = pd.read_csv("../data/raw_data.csv", index_col=0, parse_dates=True)
#     prices = data.pivot_table(values="adj_close", index="date", columns="ticker")
#     data_1 = prices[["VTWV", "EES"]].dropna()
#     data_2 = prices[["SPY", "IVV"]].dropna()
#     data_3 = pd.concat([data_1, data_2], axis=1).dropna()
#     print(huber_beta(data_3["VTWV"], data_3["EES"], data_3["SPY"]))
#
