import statsmodels.distributions.copula.api as smcop
from scipy import stats
import numpy as np


class CopulaDistribution(smcop.CopulaDistribution):
    """
    Filler class pointing to statsmodels CopulaDistribution class,
    used to generally clean-up imports.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MixedCopula():
    """
    MixedCopula class to allow for the mixed Archimedean copula,
    inputs can be either CopulaDistribution (joint distribution) or Copula objects.
    closed form conditional CDF.
    """

    def __init__(self, copula1, copula2, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copula1 = copula1
        self.copula2 = copula2
        self.alpha = alpha
        
    def alpha_update(self, alpha):
        self.alpha = alpha
        return self
    
    def rvs(self, iter, random_state=None):
        iter_1 = int(iter * self.alpha)
        iter_2 = iter - iter_1
        res = np.append(self.copula1.rvs(iter_1, random_state=random_state), 
                        self.copula2.rvs(iter_2, random_state=random_state), axis=0)
        return res
        
    
    def cdf(self, x):
        return self.alpha * self.copula1.cdf(x) + (1 - self.alpha) * self.copula2.cdf(x)
    
    def logpdf(self, x):
        return np.log(self.alpha * self.copula1.pdf(x) + (1 - self.alpha) * self.copula2.pdf(x))
        
    def cond_cdf(self, u, v):
        return self.alpha * self.copula1.cond_cdf(u, v) + (1 - self.alpha) * self.copula2.cond_cdf(u, v)


class GumbelCopula(smcop.GumbelCopula):
    """
    Extension of the GumbelCopula class to allow for the Archimedean,
    closed form conditional CDF.
    """

    def __init__(selfs, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gumbel_cop(self, u, v):
        logu = (-np.log(u)) ** self.theta
        logv = (-np.log(v)) ** self.theta
        inner_term = -((logu + logv) ** (1 / self.theta))
        return np.exp(inner_term)

    def cond_cdf(self, u, v):
        cop = self.gumbel_cop(u, v)
        lnu = (-np.log(u)) ** self.theta
        lnv = (-np.log(v)) ** self.theta
        in_one = (lnu + lnv) ** ((1 - self.theta) / self.theta)
        in_two = (-np.log(v)) ** (self.theta - 1) * (1 / v)
        return cop * in_one * in_two


class ClaytonCopula(smcop.ClaytonCopula):
    """
    Extension of the ClaytonCopula class to allow for the Archimedean,
    closed form conditional CDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cond_cdf(self, u, v):
        outer = v ** (-(self.theta + 1))
        inner = (u ** (-self.theta) + v ** (-self.theta) - 1) ** (-1 / self.theta - 1)
        return outer * inner


class FrankCopula(smcop.FrankCopula):
    """
    Extension of the FrankCopula class to allow for the Archimedean,
    closed form conditional CDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cond_cdf(self, u, v):
        numer = np.exp(-self.theta * v) * (np.exp(-self.theta * u) - 1)
        denom = np.exp(-self.theta) + (np.exp(-self.theta * u) - 1) * (
            np.exp(-self.theta * v) - 1
        )
        return numer / denom


class GaussianCopula(smcop.GaussianCopula):
    """
    Extension of the GaussianCopula class to allow for the Archimedean,
    closed form conditional CDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = self.corr[0,1]

    def cond_cdf(self, u, v):
        inner = (stats.norm.ppf(u) - self.theta * stats.norm.ppf(v)) / (
            np.sqrt(1 - self.theta**2)
        )
        return stats.norm.cdf(inner)


class IndependenceCopula(smcop.IndependenceCopula):
    """
    Extension of the IndependenceCopula class to allow for the Archimedean,
    closed form conditional CDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cond_cdf(self, u, v):
        return u * v

    # below doesnt work
    class StudentTCopula(smcop.StudentTCopula):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.theta_1, self.theta_2 = self.corr


# Filler function because we don't have a Joe Copula yet.
def joe_probs(theta, u, v):
    one = (1 - v) ** (theta - 1)
    two = ((1 - u) ** theta - (1 - u) ** (2 * theta) + (1 - v) ** theta) ** (
        1 / theta - 1
    )
    return one * two
