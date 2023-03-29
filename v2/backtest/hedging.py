import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import RANSACRegressor, LinearRegression

WINDOW_SIZE = 252
MOD_REGRESSION = 3


# This code is both from Brian's notebook for exponential weighted regressions
# and also from Tobias Rodriguez del Pozo's notebook for predictive regressions.


def hl_to_lambda(hl):
    # Convert half-life to lambda.
    return 2 ** (-1.0 / hl)


def calc_weights(hl, n):
    # Get the weights for the exponential regression.
    lamd = hl_to_lambda(hl)
    weights = lamd ** np.arange(n - 1, -1, -1)
    return weights[weights > 1e-6]


def calc_min_points(hl, n, epsilon=1e-2):
    # Calculate the minimum number of points needed to
    # perform the regression.
    weights = calc_weights(hl, n)
    return np.where(weights > epsilon)[0].shape[0]


def rolling_beta(data, hl, y_col, x_col):
    # Calculate the rolling regression, return the beta.
    n = data.shape[0]
    min_pts = calc_min_points(hl, n)
    wts = calc_weights(hl, n)

    def _reg():
        for i in range(min_pts, n):
            _df = data.iloc[max(0, i - wts.shape[0]) : i]
            _wt = wts[-_df.shape[0] :]
            _mod = sm.WLS(_df[y_col], _df[x_col], weights=_wt).fit()
            yield _mod.params[0]

    return pd.Series(list(_reg()), index=data.index[min_pts:])


def beta_calc(asset_1, asset_2, mkt, mod=MOD_REGRESSION):
    # Calculate each asset's beta to the market.
    # Various other hedge ratios are included.
    # Use RANSAC to remove outliers.

    if mod in [1, 4, 5]:
        a1 = (np.log(asset_1) - np.log(asset_1.shift(1))).cumsum().dropna()
        a2 = (np.log(asset_2) - np.log(asset_2.shift(1))).cumsum().dropna()
        mkt_data = (np.log(mkt) - np.log(mkt.shift(1))).cumsum().dropna()

    if mod == 4:
        m1 = RANSACRegressor(random_state=0, estimator=LinearRegression(fit_intercept=False))
        m1.fit(np.array(a1).reshape(-1, 1), np.array(mkt_data).reshape(-1, 1))
        beta_1 = m1.estimator_.coef_[0]
        m2 = RANSACRegressor(random_state=0, estimator=LinearRegression(fit_intercept=False))
        m2.fit(np.array(a2).reshape(-1, 1), np.array(mkt_data).reshape(-1, 1))
        beta_2 = m2.estimator_.coef_[0]
        dollar_beta_1 = beta_1 * asset_1[-1]
        dollar_beta_2 = beta_2 * asset_2[-1]
        hedge_ratio = dollar_beta_1 / dollar_beta_2
        return hedge_ratio * beta_2 / beta_1

    elif mod == 5:
        m1 = RANSACRegressor(random_state=0, estimator=LinearRegression(fit_intercept=False))
        m1.fit(np.array(a1).reshape(-1, 1), np.array(mkt_data).reshape(-1, 1))
        beta_1 = m1.estimator_.coef_[0]
        m2 = RANSACRegressor(random_state=0, estimator=LinearRegression(fit_intercept=False))
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
        m1 = RANSACRegressor(random_state=0, estimator=LinearRegression(fit_intercept=False))
        m1.fit(np.array(mkt_data).reshape(-1, 1), np.array(a1).reshape(-1, 1))
        beta_1 = m1.estimator_.coef_[0]
        m2 = RANSACRegressor(random_state=0, estimator=LinearRegression(fit_intercept=False))
        m2.fit(np.array(mkt_data).reshape(-1, 1), np.array(a2).reshape(-1, 1))
        beta_2 = m2.estimator_.coef_[0]
        return beta_1 / beta_2

    if mod == 3:
        m1 = RANSACRegressor(random_state=0, estimator=LinearRegression(fit_intercept=False))
        m1.fit(np.array(a2).reshape(-1, 1), np.array(a1).reshape(-1, 1))
        beta = m1.estimator_.coef_[0]
        return beta


def calc_rolling_betas(asset_1, asset_2, mkt, mod):
    # These betas SHOULD be lagged already
    betas = pd.DataFrame(index=asset_1.iloc[WINDOW_SIZE:].index, columns=["Betas"])
    start = 0
    end = WINDOW_SIZE
    for i in betas.index:
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
