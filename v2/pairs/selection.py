import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.cluster import OPTICS
import statsmodels.tsa.stattools as ts


class PairSelection:
    def __init__(self, price, number_of_pairs=10, optics_min_pts=4):
        self.number_of_pairs = number_of_pairs
        self.optics_min_pts = optics_min_pts
        self.price = price
        self.price.columns.name = "sym_root"
        self.log_price = np.log(
            self.price.fillna(method="ffill").fillna(method="bfill")
        )
        self.log_returns = (
            (np.log(self.price) - np.log(self.price.shift(1))).cumsum().dropna()
        )

        self.groups = pd.DataFrame()
        self.coint_result = pd.DataFrame()

    def create_clusters(self, cum_returns=None, min_pts=None):
        if min_pts is None:
            min_pts = self.optics_min_pts
        if cum_returns is None:
            cum_returns = self.log_returns

        clustering = OPTICS(min_samples=min_pts)
        clustering.fit(cum_returns.T)

        clusters = pd.DataFrame(clustering.labels_)
        clusters.index = cum_returns.columns
        clusters.columns = ["Clusters"]

        self.groups = (
            clusters[clusters["Clusters"] != -1]
            .reset_index()
            .groupby("Clusters")
            .agg({"sym_root": ["count", ", ".join]})
        )

    def get_group(self, x):
        return self.groups.loc[x, ("sym_root", "join")].split(", ")

    def test_coint(self):
        if self.groups.shape[0] == 0:
            self.create_clusters()

        columns = ["sym1", "sym2", "cluster", "p", "t", "1%", "5%", "10%"]
        codf = pd.DataFrame(columns=columns)
        for c in self.groups.index:
            group = self.get_group(c)
            for i, j in combinations(group, 2):
                co = ts.coint(self.log_price[i], self.log_price[j])
                temp = pd.DataFrame(
                    [i, j, c, co[1], co[0], co[2][0], co[2][1], co[2][2]], index=columns
                ).T
                codf = pd.concat([codf, temp], ignore_index=True)

        codf.sort_values("p", ascending=True, inplace=True, ignore_index=True)
        self.coint_result = codf

    def get_pairs(self, number_of_pairs=None):
        if number_of_pairs is None:
            number_of_pairs = self.number_of_pairs

        if self.coint_result.shape[0] == 0:
            self.test_coint()

        final_pairs = []
        selected_tickers = set()
        for s1, s2, p in self.coint_result[["sym1", "sym2", "p"]].values:
            if len(final_pairs) >= number_of_pairs:
                break
            if s1 not in selected_tickers and s2 not in selected_tickers and p <= 0.05:
                final_pairs.append((s1, s2))
                selected_tickers.add(s1)
                selected_tickers.add(s2)

        return final_pairs
