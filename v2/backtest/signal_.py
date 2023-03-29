# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys

sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../copulas/")

# from copulas.copulae import (
#     GumbelCopula,
#     ClaytonCopula,
#     FrankCopula,
#     CopulaDistribution,
# )
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
        copula_type="mixed",
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
        self.data = self.input_data_calc().dropna(axis=0, how="any")
        self.train_test_split(train_start, oos_start, oos_end)
        self.copula_type = copula_type
        if copulae:
            self.copulae = copulae
        else:
            self.c_res = CopulaPairs(self.train_data, self.pairs, self.copula_type)
            self.copulae = self.c_res.get_copulae()
        self.cdfs = self._fit_cdf()

    def train_test_split(self, train_start, oos_start, oos_end=None):
        if isinstance(train_start, str) or isinstance(train_start, pd.Timestamp):
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

    def calc_signals(self, oos_date=None, single = True):
        sigs = {}
        sig2 = {}
        for i in self.pairs:
            copula = self.copulae[i]
            cdfs = self.cdfs[i][0].cdf(self.oos_data[i[0]]), self.cdfs[i][1].cdf(
                self.oos_data[i[1]]
            )
            sigs[i] = copula.cond_cdf(cdfs[0], cdfs[1])
            if not single:
                sig2[i] = copula.cond_cdf(cdfs[1], cdfs[0])
                
        if not single:
            return pd.DataFrame(sigs), pd.DataFrame(sig2)
        else:
            return pd.DataFrame(sigs)
           


    @staticmethod
    def generate_signals(
        sig1,
        sig2 = None,
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
        if sig2 is None:
            in_trade = False
            new_signals = pd.DataFrame(index=sig1.index, columns=["Signal"])
            cur_signal = 0
            for i in sig1.index:
                p = sig1.loc[i].item()
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
        
        else:
            in_trade = False
            new_signals = pd.DataFrame(index=sig1.index, columns=["Signal"])
            cur_signal = 0
            for i in sig1.index:
                p = sig1.loc[i].item()
                p2 = sig2.loc[i].item()
                if not in_trade:
                    if p >= upper_1 and p2 <= lower_1:
                        cur_signal = -3
                        signal = -3
                        in_trade = True
                    elif p >= upper_2 and p2 <= lower_2:
                        cur_signal = -2
                        signal = -2
                        in_trade = True
                    elif p >= upper_3 and p2 <= lower_3:
                        cur_signal = -1
                        signal = -1
                        in_trade = True
                    elif p <= lower_1 and p2 >= upper_1:
                        cur_signal = 3
                        signal = 3
                        in_trade = True
                    elif p <= lower_2 and p2 >= upper_2:
                        cur_signal = 2
                        signal = 2
                        in_trade = True
                    elif p <= lower_3 and p2 >= upper_3:
                        cur_signal = 1
                        signal = 1
                        in_trade = True
                    else:
                        signal = 0
                else:
                    if (p <= exit_upper or p2 >= exit_lower) and cur_signal < 0:
                        signal = cur_signal * -1
                        in_trade = False
                    elif (p >= exit_lower or p2 <= exit_upper) and cur_signal > 0:
                        signal = cur_signal * -1
                        in_trade = False
                    else:
                        signal = 0
                new_signals.loc[i] = signal
            return new_signals

    @staticmethod
    def calc_quantiles(spread):
        up_1 = spread.quantile(0.95).item()
        up_2 = spread.quantile(0.9).item()
        up_3 = spread.quantile(0.85).item()
        lo_1 = spread.quantile(0.05).item()
        lo_2 = spread.quantile(0.10).item()
        lo_3 = spread.quantile(0.15).item()
        exit_up = spread.quantile(0.80).item()
        exit_lo = spread.quantile(0.2).item()
        return [up_1, up_2, up_3, lo_1, lo_2, lo_3, exit_up, exit_lo]


if __name__ == "__main__":
    # Pick uhhhhh EES and VTWV
    # data = pd.read_csv(
    #     "data/monthly/prices_Mar_2022_.csv", index_col=0, parse_dates=True
    # )

    data = pd.read_csv(
        "../../data/monthly/prices_Mar_2022_.csv", index_col=0, parse_dates=True
    ).ffill()

    pairs = [
        ("L", "AFL"),
        ("DIS", "BIO"),
        ("NDAQ", "FISV"),
        ("UPS", "CARR"),
        ("STZ", "CB"),
        ("ETN", "DOV"),
        ("KDP", "BDX"),
        ("WEC", "EVRG"),
        ("O", "ACN"),
        ("ATO", "ED"),
    ]
    import time

    start = time.time()
    signal = SignalGeneration(
        prices=data,
        pairs=pairs,
        train_start="2022-03-01",
        oos_start="2022-03-02",
        oos_end="2022-03-04",
    )
    df = signal.calc_signals()
    print(df)
    print(time.time() - start)
    df.iloc[:, 0].plot()
