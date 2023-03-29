import pandas as pd
import numpy as np
from scipy.optimize import minimize

from portfolio.different_covariances import Covariances

def get_default_weights(cov_matrix, mean_returns=None):
    """
    Gives the unconstrained weights of the portfolio.
    If mean_returns is None, gives the min variance portfolio, else gives the max Sharpe portfolio.
    """
    inv = np.linalg.inv(cov_matrix) 
    ones = np.ones(shape = (inv.shape[0], 1)) 
    if mean_returns is None:
        mean_returns = ones 
    weights = (inv @ mean_returns) / (ones.T @ inv @ mean_returns)
    return weights


def get_constrained_weights(cov_matrix, mean_returns):
    """
    Gives the constrained [0,1] weights of the portfolio by minimizing the negative Sharpe.
    """
    n_assets = len(mean_returns)
    
    # Objective function to be maximized
    def negative_sharpe_ratio(weights):
        mean = weights @ mean_returns
        std = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = -mean/std
        return sharpe_ratio
    
    # Equality constraint that weights must sum to 1
    def weight_constraint(weights):
        return np.sum(weights) - 1.0
    
    # Inequality constraints that each weight must be between 0 and 1
    bounds = tuple((0,1) for _ in range(n_assets))
    
    # Initial guess for weights
    initial_weights = np.ones(n_assets)/n_assets
    
    # Minimize negative Sharpe ratio subject to equality and inequality constraints
    optimization_result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints={'type':'eq', 'fun': weight_constraint})
    
    portfolio_weights = optimization_result.x
    
    return portfolio_weights


def get_weights(returns, constraint=True):
    """
    Returns a dictionary of weights for all the different covariances methods.
    """
    cov_obj = Covariances(returns)
    covariances = {'Empirical': cov_obj.emp_covariance, 
                   'LedoitWolf':cov_obj.lw_covariance, 
                   'MarcenkoPastur':cov_obj.mp_covariance}
    
    n_assets = returns.shape[1]
    mu = returns.mean()

    weights = {}
    
    weights['Equal'] = np.ones(n_assets)/n_assets

    for t, cov in covariances.items():
        if constraint:
            w = get_constrained_weights(cov, mu)
        else:
            w = get_default_weights(cov, mu)

        weights[t] = w

    return weights