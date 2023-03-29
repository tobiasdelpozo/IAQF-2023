from scipy import stats
import numpy as np


class NormalDist(stats.rv_continuous):
    def __init__(self, data, *args, **kwargs):
        super().__init__()
        param = stats.norm.fit(data)
        self._params = {"loc": param[0], "scale": param[1]}

    def _pdf(self, x):
        return stats.norm.pdf(x, **self._params)

    def _cdf(self, x):
        return stats.norm.cdf(x, **self._params)

    def _ppf(self, q):
        return stats.norm.ppf(q, **self._params)

    def _rvs(self, size):
        return stats.norm.rvs(size, **self._params)


class KDEDist(stats.rv_continuous):
    def __init__(self, df, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kde = stats.gaussian_kde(df)
        self.__ppf_sample = self._kde.resample(int(1e6)).flatten()
        self.__ppf_sample.sort()
        self.__cdf_wrapper = np.vectorize(self.cdf_point)

    def _pdf(self, x):
        return self._kde.pdf(x)

    def cdf_point(self, x):
        return self._kde.integrate_box_1d(-np.inf, x)

    def _cdf(self, x):
        return self.__cdf_wrapper(x)

    def _ppf(self, x):
        ind = (np.asarray(x) * 1e6).astype(int)
        return self.__ppf_sample[ind]
