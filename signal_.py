# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys

sys.path.append("../")
from copulas.copulae import (
    GumbelCopula,
    ClaytonCopula,
    FrankCopula,
    CopulaDistribution,
)
from copulas.distributions import KDEDist
from copulas.fitting import CopulaPairs

# from Notebooks.copulas.fitting import CopulaPairs


class SignalGeneration:
    def __init__(
        self,
        prices,
        pairs,
        train_start,
        oos_start,
        oos_end=None,
        n=100,
        mode="log_ma",
        copulae=None,
        verbose=False,
    ):
        # Initialize with TRAINING data.
        self.mode = mode
        self.prices = prices
        self.pairs = pairs
        self.n = n
        self.verbose = verbose
        self.data = self.input_data_calc()
        self.train_test_split(train_start, oos_start, oos_end)
        if copulae:
            self.copulae = copulae
        else:
            self.c_res = CopulaPairs(self.train_data, self.pairs)
            self.copulae = self.c_res.get_copulae()
        self.cdfs = self._fit_cdf()

    def train_test_split(self, train_start, oos_start, oos_end=None):
        if isinstance(train_start, str):
            if self.verbose:
                print(self.data.loc[train_start:oos_start].isna().sum())
            self.train_data = self.data.loc[train_start:oos_start].dropna(
                axis=1, how="any"
            )
            if oos_end:
                self.oos_data = self.data.loc[oos_start:oos_end]
            else:
                self.oos_data = self.data.loc[oos_start:]
        else:
            self.train_data = self.data.iloc[train_start:oos_start].dropna(
                axis=1, how="any"
            )
            if oos_end:
                self.oos_data = self.data.iloc[oos_start:oos_end]
            else:
                self.oos_data = self.data.iloc[oos_start:]

    def input_data_calc(self):
        if self.mode == "log_ma":
            return np.log(self.prices / self.prices.rolling(self.n).mean())
        elif self.mode == "log_cum_ret":
            self.init_prices = self.prices.iloc[0]
            return np.log(self.prices / self.init_prices)

    def _fit_cdf(self):
        # Fit KDE to the log returns
        cdfs = {}
        for pair in self.pairs:
            cdfs[pair] = (
                KDEDist(self.train_data[pair[0]]),
                KDEDist(self.train_data[pair[1]]),
            )
        return cdfs

    def calc_signals(self, oos_date=None):
        sigs = {}
        for i in self.pairs:
            copula = self.copulae[i]
            cdfs = self.cdfs[i][0].cdf(self.oos_data[i[0]]), self.cdfs[i][1].cdf(
                self.oos_data[i[1]]
            )
            probs = copula.cond_cdf(cdfs[0], cdfs[1])
            sigs[i] = probs
        return pd.DataFrame(sigs)


def generate_signals(
    df,
    upper_1=0.95,
    upper_2=0.9,
    upper_3=0.85,
    lower_1=0.05,
    lower_2=0.10,
    lower_3=0.15,
    exit_upper=0.80,
    exit_lower=0.20,
):
    """
    Converts probabilities to signals based on certain thresholds
    """
    in_trade = False
    new_signals = pd.DataFrame(index=df.index, columns=["Signal"])
    cur_signal = 0
    for i in df.index:
        p = df.loc[i].item()
        if not in_trade:
            if p >= upper_1:
                cur_signal = -3
                signal = -3
                in_trade = True
            elif p >= upper_2:
                cur_signal = -2
                signal = -2
                in_trade = True
            elif p >= upper_3:
                cur_signal = -1
                signal = -1
                in_trade = True
            elif p <= lower_1:
                cur_signal = 3
                signal = 3
                in_trade = True
            elif p <= lower_2:
                cur_signal = 2
                signal = 2
                in_trade = True
            elif p <= lower_3:
                cur_signal = 1
                signal = 1
                in_trade = True
            else:
                signal = 0
        else:
            if p <= exit_upper and cur_signal < 0:
                signal = cur_signal * -1
                in_trade = False
            elif p >= exit_lower and cur_signal > 0:
                signal = cur_signal * -1
                in_trade = False
            else:
                signal = 0
        new_signals.loc[i] = signal
    return new_signals


# %%
# TESTING CODE
if __name__ == "__main__":
    # Pick uhhhhh EES and VTWV
    data = pd.read_csv("../data/raw_data.csv", index_col=0, parse_dates=True)
    prices = data.pivot_table(values="adj_close", index="date", columns="ticker")
    data_1 = prices[["VTWV", "EES"]].dropna()
    data_2 = prices[["SPY", "IVV"]].dropna()
    data_3 = prices[["SUB", "FLOT"]].dropna()
    data_fin = pd.concat([data_1, data_2, data_3], axis=1).dropna(how="any")
    # data_train = data_fin.iloc[:1000]
    # data_test = data_fin.iloc[1000:]
    pairs = [("VTWV", "EES"), ("SPY", "IVV"), ("SUB", "FLOT")]
    signal = SignalGeneration(
        data_fin, pairs, train_start=0, oos_start=1000, mode="log_ma"
    )
    # print(signal.calc_signals(data_test))
    df = signal.calc_signals()
    df.iloc[:, 0].plot()

    # %%
    """
    INPUT STRUCTURE:

    data_dict = {
        ('SPY', 'VOO'): {'copula_type': GumbelCopula, 'theta': 8},
        ('VTWV', 'EES'): {'copula_type': ClaytonCopula, 'theta': 1},
    }

    prices (dataframe - RAW prices):
    date, SPY, VOO, VTWV, EES
    ...., ..., ..., ...., ... 

    """

    # %%
