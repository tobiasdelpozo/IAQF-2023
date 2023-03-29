import pandas as pd
import numpy as np
from portfolio.marcenko_pastur import MarcenkoPasturCovariance
from sklearn.covariance import LedoitWolf

class Covariances:
    def __init__(self, returns):
        self.returns = returns
        self.emp_covariance = self.get_empirical_covariance()
        self.lw_covariance = self.get_ledoit_wolf_covariance()
        self.mp_covariance = self.get_denoised_covariance()

    def get_empirical_covariance(self):
        return self.returns.cov()
    
    def get_ledoit_wolf_covariance(self):
        cov = LedoitWolf().fit(self.emp_covariance).covariance_
        cov = pd.DataFrame(cov, index=self.returns.columns, columns=self.returns.columns)
        return cov
    
    def get_denoised_covariance(self):
        cov = MarcenkoPasturCovariance(self.returns).covariance
        cov = pd.DataFrame(cov, index=self.returns.columns, columns=self.returns.columns)
        return cov