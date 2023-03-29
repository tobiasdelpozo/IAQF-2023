# %%
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import repeat

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from copulas.copulae import (
    GaussianCopula,
    ClaytonCopula,
    GumbelCopula,
    FrankCopula,
    CopulaDistribution,
    MixedCopula,
)
from copulas.distributions import NormalDist, KDEDist


class CopulaPairs:
    def __init__(
        self, data, pairs, copula_type="mixed", marginal_type="kde", alpha=None
    ):
        self.pairs = list(pairs)
        self.data = data
        self.copula_type = copula_type
        self.marginal_type = marginal_type
        self.alpha = alpha
        self.copulae = {}

    # def get_copulae(self):
    #     for pair in self.pairs:
    #         copula_fitter = CopulaSelection(
    #             self.data, pair, self.copula_type, self.marginal_type, self.alpha
    #         )
    #         self.copulae[pair] = copula_fitter.get_copula()
    #     return self.copulae
    
    def get_copulae(self):
        n = len(self.pairs)
        with ProcessPoolExecutor(max_workers=n) as executor:
            copula_fitter = [CopulaSelection(self.data, pair, self.copula_type, self.marginal_type, self.alpha) for pair in self.pairs]
            copulae = [executor.submit(i.get_copula) for i in  copula_fitter]
            for ind, copula in enumerate(as_completed(copulae)):
                self.copulae[self.pairs[ind]] = copula.result()
        return self.copulae


class CopulaSelection:
    def __init__(
        self, data, pairs, copula_type="mixed", marginal_type="kde", alpha=None
    ):
        self.pairs = list(pairs)
        self.data = data[self.pairs]
        self.a1 = data[pairs[0]]
        self.a2 = data[pairs[1]]
        self.copula_type = copula_type
        self.marginal_type = marginal_type
        self.alpha = alpha
        self.candidate_copulas = ["gumbel", "clayton", "independence", "gaussian"]
        self.mixed_candidate_copulas = [f"{cop1}_{cop2}" for cop1, cop2 in combinations(self.candidate_copulas, 2)]
        self.get_marginals()

    def get_marginals(self):
        if self.marginal_type == "kde":
            self.cdf_a1 = KDEDist(self.a1).cdf(self.a1)
            self.cdf_a2 = KDEDist(self.a2).cdf(self.a2)
            self.marginals = [KDEDist(self.a1), KDEDist(self.a2)]
        elif self.marginal_type == "normal":
            self.cdf_a1 = NormalDist(self.a1).cdf(self.a1)
            self.cdf_a2 = NormalDist(self.a2).cdf(self.a2)
            self.marginals = [NormalDist(self.a1), NormalDist(self.a2)]
        elif self.marginal_type == "uniform":
            self.cdf_a1 = stats.rankdata(self.a1) / len(self.a1)
            self.cdf_a2 = stats.rankdata(self.a2) / len(self.a2)
            self.marginals = [stats.uniform, stats.uniform]
        else:
            raise ValueError("Marginal type not recognized.")

    def log_likelihood(self, joint_dist):
        return joint_dist.logpdf(np.array(self.data)).sum()

    def criteria(self, joint_dist):
        loglik = self.log_likelihood(joint_dist)
        num_params = 1 + sum([self.marginals[i].numargs for i in range(2)])
        num_obs = len(self.cdf_a1)

        AIC = -2 * loglik + 2 * num_params
        SIC = -2 * loglik + num_params * np.log(num_obs)
        HQIC = -2 * loglik + 2 * num_params * np.log(np.log(num_obs))
        BIC = -2 * loglik + np.log(len(self.cdf_a1) * len(self.cdf_a2)) * num_params
        return {"AIC": AIC, "SIC": SIC, "HQIC": HQIC, "BIC": BIC}

    def base_copula(self, copula_type, theta=None):
        if theta is None:
            if copula_type == "gumbel":
                copula = GumbelCopula()
            elif copula_type == "clayton":
                copula = ClaytonCopula()
            elif copula_type == "frank":
                copula = FrankCopula()
            elif copula_type == "independence":
                copula = GumbelCopula(1 + 1e-15)
            elif copula_type == "gaussian":
                copula = GaussianCopula()
            else:
                raise ValueError("Copula type not recognized.")
        else:
            if copula_type == "gumbel":
                copula = GumbelCopula(theta)
            elif copula_type == "clayton":
                copula = ClaytonCopula(theta)
            elif copula_type == "frank":
                copula = FrankCopula(theta)
            elif copula_type == "independence":
                copula = GumbelCopula(1 + 1e-15)
            elif copula_type == "gaussian":
                copula = GaussianCopula(theta)
            else:
                raise ValueError("Copula type not recognized.")
        return copula

    def fit_copula(
        self,
        copula_type,
        copula_type2=None,
        theta=None,
        theta2=None,
        alpha=None,
        joint_dist=True,
    ):
        copula = self.base_copula(copula_type)
        if theta is None:
            try:
                theta = copula.fit_corr_param(self.data.values)
            except Exception as e:
                print(e, copula_type, "theta :", theta)
                return None
        try:
            copula = self.base_copula(copula_type, theta=theta)
        except Exception as e:
            if theta < 1:
                copula = self.base_copula(copula_type, theta=1.001)
            else:
                print(e, copula_type, "theta :", theta)
                return None

        if joint_dist:  # fit distribution and return joint distribution
            try:
                copula = CopulaDistribution(copula, self.marginals)
            except:
                raise ValueError("Copula fitting failed.")

        # In case of mixed copula
        if copula_type2:
            copula2 = self.fit_copula(
                copula_type2, theta=theta2, joint_dist=joint_dist
            )  # Get second copula
            if copula2 is None:
                return None
            copula_type = copula_type + "_" + copula_type2
            if alpha:
                copula = MixedCopula(
                    copula, copula2, alpha=alpha
                )  # Fit mixed copula if alpha is given
            else:
                copula = MixedCopula(copula, copula2)
        copula.copula_type = copula_type
        return copula

    def optimize_mixed_copula(self, mixed_copula, bnds=(0, 1), tolerance = None):
        opt_func = lambda alpha: -self.log_likelihood(mixed_copula.alpha_update(alpha))
        alpha = optimize.minimize_scalar(opt_func, bounds=bnds, method="bounded", tol = 0.1).x
        return mixed_copula.alpha_update(alpha)

    def evaluate_mixed_copula(self):
        mixed_copulae = [
            self.fit_copula(cop, cop2)
            for cop, cop2 in combinations(self.candidate_copulas, 2)
        ]
        none_ = []
        res = []
        for ind, copula in enumerate(mixed_copulae):
            if copula is None:
                none_.append(self.mixed_candidate_copulas[ind])
            opt_copula = self.optimize_mixed_copula(copula)
            crit = self.criteria(opt_copula)
            crit["alpha"] = opt_copula.alpha
            res.append(crit)
        return pd.DataFrame(
            res,
            index=[cop for cop in self.mixed_candidate_copulas if cop not in none_],
        )

    def evaluate_copula(self):
        res = []
        copulae = [self.fit_copula(cop) for cop in self.candidate_copulas]
        none_ = []
        for ind, copula in enumerate(copulae):
            if copula is not None:
                res.append(self.criteria(copula))
            else:
                none_.append(self.candidate_copulas[ind])
        return pd.DataFrame(res, 
                            index=[i for i in self.candidate_copulas if i not in none_])

    def get_copula(self):
        if isinstance(self.copula_type, list):
            copula = self.fit_copula(self.copula_type[0], self.copula_type[1])
            copula_optimzed = self.optimize_mixed_copula(copula) # Joint distribution
            return self.fit_copula(self.copula_type[0], self.copula_type[1], alpha=copula_optimzed.alpha, joint_dist=False)
        
        elif self.copula_type != "mixed":
            return self.fit_copula(self.copula_type, joint_dist=False)
        else:
            table = self.evaluate_mixed_copula()
            c_name = table.drop("alpha", axis=1).mean(1).idxmin()
            alpha = table["alpha"].loc[c_name]
            c1, c2 = c_name.split("_")
            return self.fit_copula(c1, c2, alpha=alpha, joint_dist=False)


# %%
if __name__ == "__main__":
    data = pd.read_csv(
        "/Users/hun/Library/CloudStorage/OneDrive-SNU/1. University/UChicago/2022 4Q Winter/IAQF/IAQF_2023/Notebooks/data/raw_data.csv",
        index_col=0,
        parse_dates=True,
    )
    prices = data.pivot_table(values="adj_close", index="date", columns="ticker")
    data_1 = prices[["VTWV", "EES"]].dropna()
    data_2 = prices[["SPY", "IVV"]].dropna()
    data_3 = prices[["SUB", "FLOT"]].dropna()
    data_fin = (
        pd.concat([data_1, data_2, data_3], axis=1)
        .sort_index()
        .pct_change()
        .dropna(how="any")
    )
    data_fin.index = pd.to_datetime(data_fin.index)
    tickers = [("VTWV", "EES"), ("SPY", "IVV"), ("SUB", "FLOT")]
    # %%
    sample = data_fin.loc["2015":"2017"]

    # %%
    c_res = CopulaPairs(sample, tickers)
    pairs_copula = c_res.get_copulae()
    pairs_copula[tickers[0]].cond_cdf(0.02, 0.05)

    # %%
    cop = CopulaSelection(sample, pairs=tickers[-1], copula_type="mixed")
    cop.get_copula()
